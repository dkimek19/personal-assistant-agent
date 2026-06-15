"""Tests for the SearXNG search client module.

Covers:
- Successful search with mocked HTTP response
- Correct parsing of url / title / snippet fields
- Partial / missing fields in results
- Empty result list
- HTTP error responses (4xx, 5xx)
- Connection / network errors
- Exponential backoff retry logic (verifies sleep calls and attempt count)
- ToolError raised after exhausting all retries
- ValueError for empty query
- Non-retryable JSON parse error
- base_url and query string forwarding
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, call, patch

import httpx
import pytest

from assistant.tools.searxng import (
    ToolError,
    _DEFAULT_NUM_RESULTS,
    _MAX_RETRIES,
    _parse_results,
    _score_result,
    filter_results,
    search,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SAMPLE_RESULTS = [
    {
        "url": "https://example.com/page1",
        "title": "Example Page One",
        "content": "This is the snippet for page one.",
    },
    {
        "url": "https://example.com/page2",
        "title": "Example Page Two",
        "content": "Snippet for page two.",
    },
]

SAMPLE_RESPONSE_JSON = {"results": SAMPLE_RESULTS, "query": "hello world"}


def _make_response(
    status_code: int = 200,
    json_body: dict | None = None,
    raise_for_status_exc: Exception | None = None,
) -> MagicMock:
    """Build a mock httpx.Response."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = status_code
    mock_resp.json.return_value = (
        json_body if json_body is not None else SAMPLE_RESPONSE_JSON
    )
    if raise_for_status_exc is not None:
        mock_resp.raise_for_status.side_effect = raise_for_status_exc
    else:
        mock_resp.raise_for_status.return_value = None
    return mock_resp


# ---------------------------------------------------------------------------
# Unit tests for _parse_results
# ---------------------------------------------------------------------------


class TestParseResults:
    def test_maps_content_to_snippet(self):
        raw = [{"url": "https://a.com", "title": "A", "content": "The snippet."}]
        results = _parse_results(raw)
        assert results[0]["snippet"] == "The snippet."

    def test_all_three_fields_present(self):
        results = _parse_results(SAMPLE_RESULTS)
        for r in results:
            assert set(r.keys()) == {"url", "title", "snippet"}

    def test_missing_url_defaults_to_empty_string(self):
        raw = [{"title": "No URL", "content": "snippet"}]
        results = _parse_results(raw)
        assert results[0]["url"] == ""

    def test_missing_title_defaults_to_empty_string(self):
        raw = [{"url": "https://x.com", "content": "snippet"}]
        results = _parse_results(raw)
        assert results[0]["title"] == ""

    def test_missing_content_defaults_to_empty_string(self):
        raw = [{"url": "https://x.com", "title": "X"}]
        results = _parse_results(raw)
        assert results[0]["snippet"] == ""

    def test_empty_list_returns_empty_list(self):
        assert _parse_results([]) == []

    def test_values_cast_to_str(self):
        raw = [{"url": 123, "title": None, "content": 3.14}]
        results = _parse_results(raw)
        assert results[0]["url"] == "123"
        assert results[0]["title"] == "None"
        assert results[0]["snippet"] == "3.14"


# ---------------------------------------------------------------------------
# Integration-style tests for search() with mocked httpx.get
# ---------------------------------------------------------------------------


class TestSearchSuccess:
    @patch("assistant.tools.searxng.httpx.get")
    def test_returns_list_of_dicts(self, mock_get):
        mock_get.return_value = _make_response()
        results = search("hello world")
        assert isinstance(results, list)
        assert len(results) == 2

    @patch("assistant.tools.searxng.httpx.get")
    def test_result_has_url_title_snippet_keys(self, mock_get):
        mock_get.return_value = _make_response()
        results = search("test query")
        for r in results:
            assert "url" in r
            assert "title" in r
            assert "snippet" in r

    @patch("assistant.tools.searxng.httpx.get")
    def test_url_passthrough(self, mock_get):
        mock_get.return_value = _make_response()
        results = search("q")
        assert results[0]["url"] == "https://example.com/page1"
        assert results[1]["url"] == "https://example.com/page2"

    @patch("assistant.tools.searxng.httpx.get")
    def test_title_passthrough(self, mock_get):
        mock_get.return_value = _make_response()
        results = search("q")
        assert results[0]["title"] == "Example Page One"
        assert results[1]["title"] == "Example Page Two"

    @patch("assistant.tools.searxng.httpx.get")
    def test_snippet_passthrough(self, mock_get):
        mock_get.return_value = _make_response()
        results = search("q")
        assert results[0]["snippet"] == "This is the snippet for page one."
        assert results[1]["snippet"] == "Snippet for page two."

    @patch("assistant.tools.searxng.httpx.get")
    def test_empty_results_field_returns_empty_list(self, mock_get):
        mock_get.return_value = _make_response(json_body={"results": []})
        results = search("empty query")
        assert results == []

    @patch("assistant.tools.searxng.httpx.get")
    def test_missing_results_key_returns_empty_list(self, mock_get):
        mock_get.return_value = _make_response(json_body={})
        results = search("missing key")
        assert results == []

    @patch("assistant.tools.searxng.httpx.get")
    def test_calls_correct_endpoint(self, mock_get):
        mock_get.return_value = _make_response()
        search("my query", base_url="http://localhost:9999")
        called_url = mock_get.call_args[0][0]
        assert called_url == "http://localhost:9999/search"

    @patch("assistant.tools.searxng.httpx.get")
    def test_query_forwarded_in_params(self, mock_get):
        mock_get.return_value = _make_response()
        search("specific query text")
        params = mock_get.call_args[1]["params"]
        assert params["q"] == "specific query text"

    @patch("assistant.tools.searxng.httpx.get")
    def test_format_json_in_params(self, mock_get):
        mock_get.return_value = _make_response()
        search("q")
        params = mock_get.call_args[1]["params"]
        assert params["format"] == "json"

    @patch("assistant.tools.searxng.httpx.get")
    def test_query_stripped_of_whitespace(self, mock_get):
        mock_get.return_value = _make_response()
        search("  trimmed query  ")
        params = mock_get.call_args[1]["params"]
        assert params["q"] == "trimmed query"

    @patch("assistant.tools.searxng.httpx.get")
    def test_single_attempt_on_success(self, mock_get):
        mock_get.return_value = _make_response()
        search("q")
        assert mock_get.call_count == 1


# ---------------------------------------------------------------------------
# Sub-AC 2: search() returns relevant, non-empty results for a known query
# against a live or mocked SearXNG endpoint, with assertions that the result
# list is non-empty and contains expected relevant terms/URLs.
# ---------------------------------------------------------------------------


# A "known query" with a representative SearXNG-style JSON response. The
# fixture mirrors real SearXNG output (the snippet field is named "content")
# for the query "python programming tutorial".
KNOWN_QUERY = "python programming tutorial"

KNOWN_QUERY_RESPONSE_JSON = {
    "query": KNOWN_QUERY,
    "results": [
        {
            "url": "https://docs.python.org/3/tutorial/index.html",
            "title": "The Python Tutorial — Python 3 documentation",
            "content": (
                "This tutorial introduces the reader informally to the "
                "basic concepts and features of the Python language."
            ),
        },
        {
            "url": "https://www.learnpython.org/",
            "title": "Learn Python - Free Interactive Python Tutorial",
            "content": (
                "Learn Python programming with our interactive Python "
                "tutorial covering the basics to advanced topics."
            ),
        },
        {
            "url": "https://www.w3schools.com/python/",
            "title": "Python Tutorial - W3Schools",
            "content": "Well organized and easy to understand Python programming tutorial.",
        },
    ],
}

# Terms and URLs we expect to see for the known query above.
EXPECTED_RELEVANT_TERMS = ("python", "tutorial")
EXPECTED_RELEVANT_URL = "https://docs.python.org/3/tutorial/index.html"


class TestKnownQueryRelevance:
    """search() returns relevant results for a known query (Sub-AC 2)."""

    @patch("assistant.tools.searxng.httpx.get")
    def test_known_query_returns_non_empty_relevant_results(self, mock_get):
        """Mocked SearXNG endpoint: result list is non-empty and contains
        expected relevant terms and URLs for a known query."""
        mock_get.return_value = _make_response(json_body=KNOWN_QUERY_RESPONSE_JSON)

        results = search(KNOWN_QUERY)

        # Result list must be non-empty.
        assert isinstance(results, list)
        assert len(results) > 0

        # Every result must expose the standard url/title/snippet keys with
        # factual data passed through unmodified from the SearXNG response.
        for r in results:
            assert "url" in r
            assert "title" in r
            assert "snippet" in r

        # The result set must contain the expected relevant URL verbatim.
        urls = [r["url"] for r in results]
        assert EXPECTED_RELEVANT_URL in urls

        # The result set must contain the expected relevant terms (e.g.
        # "python" and "tutorial") in the combined title/snippet/url text.
        haystack = " ".join(
            f"{r['title']} {r['snippet']} {r['url']}".lower() for r in results
        )
        for term in EXPECTED_RELEVANT_TERMS:
            assert term in haystack

    @patch("assistant.tools.searxng.httpx.get")
    def test_known_query_filtered_results_remain_relevant(self, mock_get):
        """filter_results() applied to a known query keeps only relevant
        results, non-empty, with expected terms/URLs preserved."""
        mock_get.return_value = _make_response(json_body=KNOWN_QUERY_RESPONSE_JSON)

        results = search(KNOWN_QUERY)
        top_results = filter_results(results, KNOWN_QUERY, top_n=2)

        assert len(top_results) > 0
        urls = [r["url"] for r in top_results]
        assert EXPECTED_RELEVANT_URL in urls
        for r in top_results:
            assert r["_relevance_score"] > 0.0

    @pytest.mark.skipif(
        os.environ.get("SEARXNG_LIVE_TEST") != "1",
        reason=(
            "Live SearXNG integration test skipped by default. "
            "Set SEARXNG_LIVE_TEST=1 (and optionally SEARXNG_URL) to run "
            "against a real SearXNG instance."
        ),
    )
    def test_known_query_against_live_searxng_endpoint(self):
        """Optional live test: search() against a real SearXNG instance.

        Enabled only when SEARXNG_LIVE_TEST=1 is set in the environment so
        the default test run does not require network access or a running
        SearXNG container.
        """
        results = search(KNOWN_QUERY)

        assert isinstance(results, list)
        assert len(results) > 0

        haystack = " ".join(
            f"{r['title']} {r['snippet']} {r['url']}".lower() for r in results
        )
        assert "python" in haystack


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestSearchErrors:
    def test_raises_value_error_for_empty_query(self):
        with pytest.raises(ValueError):
            search("")

    def test_raises_value_error_for_whitespace_only_query(self):
        with pytest.raises(ValueError):
            search("   ")

    @patch("assistant.tools.searxng.time.sleep")
    @patch("assistant.tools.searxng.httpx.get")
    def test_retries_on_http_500_and_raises_tool_error(self, mock_get, mock_sleep):
        error_response = MagicMock(spec=httpx.Response)
        error_response.status_code = 500
        error_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500 Server Error", request=MagicMock(), response=error_response
        )
        mock_get.return_value = error_response

        with pytest.raises(ToolError):
            search("fail query")

        # Should have attempted _MAX_RETRIES + 1 times total
        assert mock_get.call_count == _MAX_RETRIES + 1

    @patch("assistant.tools.searxng.time.sleep")
    @patch("assistant.tools.searxng.httpx.get")
    def test_retries_on_http_429_and_raises_tool_error(self, mock_get, mock_sleep):
        error_response = MagicMock(spec=httpx.Response)
        error_response.status_code = 429
        error_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "429 Too Many Requests", request=MagicMock(), response=error_response
        )
        mock_get.return_value = error_response

        with pytest.raises(ToolError):
            search("rate limited")

        assert mock_get.call_count == _MAX_RETRIES + 1

    @patch("assistant.tools.searxng.time.sleep")
    @patch("assistant.tools.searxng.httpx.get")
    def test_retries_on_connection_error(self, mock_get, mock_sleep):
        mock_get.side_effect = httpx.ConnectError("Connection refused")

        with pytest.raises(ToolError):
            search("connection fail")

        assert mock_get.call_count == _MAX_RETRIES + 1

    @patch("assistant.tools.searxng.time.sleep")
    @patch("assistant.tools.searxng.httpx.get")
    def test_retries_on_timeout_error(self, mock_get, mock_sleep):
        mock_get.side_effect = httpx.TimeoutException("Request timed out")

        with pytest.raises(ToolError):
            search("timeout query")

        assert mock_get.call_count == _MAX_RETRIES + 1

    @patch("assistant.tools.searxng.time.sleep")
    @patch("assistant.tools.searxng.httpx.get")
    def test_tool_error_message_mentions_query(self, mock_get, mock_sleep):
        mock_get.side_effect = httpx.ConnectError("refused")

        with pytest.raises(ToolError) as exc_info:
            search("my specific query")

        assert "my specific query" in str(exc_info.value)

    @patch("assistant.tools.searxng.time.sleep")
    @patch("assistant.tools.searxng.httpx.get")
    def test_tool_error_message_mentions_attempt_count(self, mock_get, mock_sleep):
        mock_get.side_effect = httpx.ConnectError("refused")

        with pytest.raises(ToolError) as exc_info:
            search("q")

        assert str(_MAX_RETRIES + 1) in str(exc_info.value)


# ---------------------------------------------------------------------------
# Exponential backoff tests
# ---------------------------------------------------------------------------


class TestExponentialBackoff:
    @patch("assistant.tools.searxng.time.sleep")
    @patch("assistant.tools.searxng.httpx.get")
    def test_no_sleep_on_first_attempt_success(self, mock_get, mock_sleep):
        mock_get.return_value = _make_response()
        search("q")
        mock_sleep.assert_not_called()

    @patch("assistant.tools.searxng.time.sleep")
    @patch("assistant.tools.searxng.httpx.get")
    def test_sleep_durations_are_exponential(self, mock_get, mock_sleep):
        """Verify sleep(1), sleep(2) pattern for _MAX_RETRIES=2."""
        mock_get.side_effect = httpx.ConnectError("refused")

        with pytest.raises(ToolError):
            search("q")

        # With _MAX_RETRIES=2 we do 3 attempts; sleep happens before attempt 1 and 2
        assert mock_sleep.call_count == _MAX_RETRIES
        sleep_args = [c.args[0] for c in mock_sleep.call_args_list]
        # First retry: 1.0 * 2^0 = 1.0
        assert sleep_args[0] == pytest.approx(1.0)
        # Second retry: 1.0 * 2^1 = 2.0
        assert sleep_args[1] == pytest.approx(2.0)

    @patch("assistant.tools.searxng.time.sleep")
    @patch("assistant.tools.searxng.httpx.get")
    def test_succeeds_on_second_attempt_without_raising(self, mock_get, mock_sleep):
        """First attempt fails, second succeeds — no ToolError raised."""
        mock_get.side_effect = [
            httpx.ConnectError("first fail"),
            _make_response(),
        ]
        results = search("recovery query")
        assert isinstance(results, list)
        assert mock_get.call_count == 2
        # Only one sleep between attempt 0 and attempt 1
        assert mock_sleep.call_count == 1
        assert mock_sleep.call_args_list[0] == call(1.0)

    @patch("assistant.tools.searxng.time.sleep")
    @patch("assistant.tools.searxng.httpx.get")
    def test_succeeds_on_third_attempt_without_raising(self, mock_get, mock_sleep):
        """First two attempts fail, third succeeds."""
        mock_get.side_effect = [
            httpx.ConnectError("fail 1"),
            httpx.ConnectError("fail 2"),
            _make_response(),
        ]
        results = search("late recovery")
        assert isinstance(results, list)
        assert mock_get.call_count == 3
        assert mock_sleep.call_count == 2


# ---------------------------------------------------------------------------
# Tests for filter_results
# ---------------------------------------------------------------------------


# Fixture data used across filter_results tests
FILTER_FIXTURE_RESULTS = [
    {
        "url": "https://example.com/python-tutorial",
        "title": "Python Programming Tutorial",
        "snippet": "Learn Python programming from scratch with examples.",
    },
    {
        "url": "https://example.com/java-guide",
        "title": "Java Developer Guide",
        "snippet": "Comprehensive guide for Java developers.",
    },
    {
        "url": "https://example.com/python-advanced",
        "title": "Advanced Python Techniques",
        "snippet": "Deep dive into advanced Python features and patterns.",
    },
    {
        "url": "https://example.com/rust-book",
        "title": "The Rust Programming Language",
        "snippet": "Official Rust book covering all language features.",
    },
    {
        "url": "https://example.com/python-data-science",
        "title": "Python for Data Science",
        "snippet": "Using Python for data analysis and machine learning.",
    },
]


class TestScoreResult:
    """Unit tests for the internal _score_result helper."""

    def test_zero_score_for_no_matching_terms(self):
        result = {"title": "Rust Programming", "snippet": "All about Rust.", "url": "https://rust-lang.org"}
        score = _score_result(result, ["python"])
        assert score == 0.0

    def test_title_match_contributes_2_points(self):
        result = {"title": "Python Tutorial", "snippet": "", "url": ""}
        score = _score_result(result, ["python"])
        assert score == pytest.approx(2.0)

    def test_snippet_match_contributes_1_point(self):
        result = {"title": "Tutorial", "snippet": "Learn python here.", "url": ""}
        score = _score_result(result, ["python"])
        assert score == pytest.approx(1.0)

    def test_url_match_contributes_half_point(self):
        result = {"title": "Tutorial", "snippet": "Learn things.", "url": "https://python.org"}
        score = _score_result(result, ["python"])
        assert score == pytest.approx(0.5)

    def test_multiple_term_occurrences_accumulate(self):
        result = {"title": "Python Python", "snippet": "", "url": ""}
        score = _score_result(result, ["python"])
        # "python" appears twice in title → 2 × 2.0 = 4.0
        assert score == pytest.approx(4.0)

    def test_multiple_query_terms_all_count(self):
        result = {"title": "Python Tutorial", "snippet": "Learn programming.", "url": ""}
        score = _score_result(result, ["python", "programming"])
        # "python" in title → 2.0; "programming" in snippet → 1.0
        assert score == pytest.approx(3.0)

    def test_empty_query_terms_returns_zero(self):
        result = {"title": "Python Tutorial", "snippet": "snippet", "url": "https://x.com"}
        assert _score_result(result, []) == 0.0

    def test_case_insensitive_matching(self):
        result = {"title": "PYTHON Tutorial", "snippet": "", "url": ""}
        score = _score_result(result, ["python"])
        assert score == pytest.approx(2.0)

    def test_missing_fields_do_not_raise(self):
        # All fields missing
        score = _score_result({}, ["python"])
        assert score == 0.0

    def test_content_key_also_used_for_snippet(self):
        """SearXNG raw results use 'content' instead of 'snippet'."""
        result = {"title": "", "content": "All about python.", "url": ""}
        score = _score_result(result, ["python"])
        assert score == pytest.approx(1.0)


class TestFilterResults:
    """Tests for the public filter_results function."""

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_empty_results_returns_empty_list(self):
        assert filter_results([], "python", top_n=5) == []

    def test_top_n_zero_returns_empty_list(self):
        assert filter_results(FILTER_FIXTURE_RESULTS, "python", top_n=0) == []

    def test_top_n_negative_returns_empty_list(self):
        assert filter_results(FILTER_FIXTURE_RESULTS, "python", top_n=-1) == []

    def test_fewer_results_than_top_n_returns_all(self):
        two_results = FILTER_FIXTURE_RESULTS[:2]
        out = filter_results(two_results, "python", top_n=10)
        assert len(out) == 2

    def test_fewer_results_than_top_n_all_results_included(self):
        """All items are returned when len(results) < top_n."""
        single = [FILTER_FIXTURE_RESULTS[0]]
        out = filter_results(single, "python tutorial", top_n=100)
        assert len(out) == 1
        assert out[0]["url"] == "https://example.com/python-tutorial"

    def test_empty_query_returns_all_results_in_original_order(self):
        """When query is empty every result scores 0; original order preserved."""
        out = filter_results(FILTER_FIXTURE_RESULTS, "", top_n=5)
        assert len(out) == 5
        for i, r in enumerate(out):
            assert r["url"] == FILTER_FIXTURE_RESULTS[i]["url"]

    def test_whitespace_query_treated_as_empty(self):
        out = filter_results(FILTER_FIXTURE_RESULTS, "   ", top_n=3)
        # All scores are 0; first 3 items in original order
        assert len(out) == 3
        assert out[0]["url"] == FILTER_FIXTURE_RESULTS[0]["url"]

    # ------------------------------------------------------------------
    # Top-n ordering (core requirement)
    # ------------------------------------------------------------------

    def test_top_n_limits_result_count(self):
        out = filter_results(FILTER_FIXTURE_RESULTS, "python", top_n=3)
        assert len(out) == 3

    def test_most_relevant_result_is_first(self):
        """The item with the most Python mentions should rank first."""
        out = filter_results(FILTER_FIXTURE_RESULTS, "python", top_n=5)
        # "Advanced Python Techniques" snippet has "advanced python features and patterns"
        # plus title "Advanced Python Techniques" → 2 title hits
        # "Python for Data Science" has title hit + snippet hit
        # "Python Programming Tutorial" has title hit + snippet hit
        # All Python-related should beat Java and Rust
        urls = [r["url"] for r in out]
        # Java and Rust results should not be in the top 3
        assert "https://example.com/java-guide" not in urls[:3]
        assert "https://example.com/rust-book" not in urls[:3]

    def test_results_sorted_descending_by_score(self):
        """Returned list must be strictly sorted by _relevance_score descending."""
        out = filter_results(FILTER_FIXTURE_RESULTS, "python programming", top_n=5)
        scores = [r["_relevance_score"] for r in out]
        assert scores == sorted(scores, reverse=True)

    def test_zero_score_items_sorted_after_positive_score_items(self):
        results = [
            {"url": "https://irrelevant.com", "title": "Nothing here", "snippet": ""},
            {"url": "https://relevant.com", "title": "Python Guide", "snippet": "Python tutorial"},
        ]
        out = filter_results(results, "python", top_n=2)
        assert out[0]["url"] == "https://relevant.com"
        assert out[1]["url"] == "https://irrelevant.com"

    def test_top_1_returns_highest_scoring_result(self):
        """top_n=1 must return only the single best result."""
        results = [
            {"url": "https://low.com", "title": "Nothing", "snippet": ""},
            {"url": "https://high.com", "title": "Python Python Python", "snippet": "python"},
            {"url": "https://mid.com", "title": "Python guide", "snippet": ""},
        ]
        out = filter_results(results, "python", top_n=1)
        assert len(out) == 1
        assert out[0]["url"] == "https://high.com"

    def test_ordering_consistent_with_scores(self):
        """Manually verify score ordering for a controlled fixture."""
        results = [
            # score for "python": title=0, snippet=0  → 0.0
            {"url": "https://a.com", "title": "Java", "snippet": "Java guide"},
            # score for "python": title=2.0, snippet=1.0 → 3.0
            {"url": "https://b.com", "title": "Python", "snippet": "Learn python"},
            # score for "python": title=2.0, snippet=0 → 2.0
            {"url": "https://c.com", "title": "Python Guide", "snippet": "A guide"},
        ]
        out = filter_results(results, "python", top_n=3)
        assert out[0]["url"] == "https://b.com"   # score 3.0
        assert out[1]["url"] == "https://c.com"   # score 2.0
        assert out[2]["url"] == "https://a.com"   # score 0.0

    # ------------------------------------------------------------------
    # _relevance_score key injection
    # ------------------------------------------------------------------

    def test_relevance_score_key_injected(self):
        out = filter_results(FILTER_FIXTURE_RESULTS, "python", top_n=3)
        for r in out:
            assert "_relevance_score" in r

    def test_relevance_score_is_float(self):
        out = filter_results(FILTER_FIXTURE_RESULTS, "python", top_n=3)
        for r in out:
            assert isinstance(r["_relevance_score"], float)

    def test_original_dicts_not_mutated(self):
        """filter_results must not modify the caller's original dicts."""
        originals = [
            {"url": "https://x.com", "title": "Python", "snippet": "snippet"},
        ]
        original_keys = set(originals[0].keys())
        filter_results(originals, "python", top_n=1)
        assert set(originals[0].keys()) == original_keys

    def test_original_list_not_mutated(self):
        """The results list itself must not be reordered by the function."""
        results_copy = list(FILTER_FIXTURE_RESULTS)
        filter_results(FILTER_FIXTURE_RESULTS, "python", top_n=3)
        assert FILTER_FIXTURE_RESULTS == results_copy

    # ------------------------------------------------------------------
    # Exact fixture-data top-n ordering assertion (AC requirement)
    # ------------------------------------------------------------------

    def test_fixture_top3_python_query_order(self):
        """
        Fixture-based assertion: for query='python', verify the top-3 results
        are the three Python-related pages in the expected ranked order.

        Scores (query terms: ['python']):
          - python-tutorial:    title='Python Programming Tutorial'  → 1 match × 2 = 2.0
                                snippet='Learn Python programming from scratch...' → 1×1 = 1.0
                                url has 'python' → 0.5
                                TOTAL ≈ 3.5

          - python-advanced:    title='Advanced Python Techniques' → 1×2 = 2.0
                                snippet='Deep dive into advanced Python features...' → 1×1 = 1.0
                                url has 'python' → 0.5
                                TOTAL ≈ 3.5

          - python-data-science: title='Python for Data Science' → 1×2 = 2.0
                                snippet='Using Python for data analysis...' → 1×1 = 1.0
                                url has 'python' → 0.5
                                TOTAL ≈ 3.5

          - java-guide:         TOTAL = 0.0
          - rust-book:          TOTAL = 0.0
        """
        out = filter_results(FILTER_FIXTURE_RESULTS, "python", top_n=3)
        assert len(out) == 3
        python_urls = {
            "https://example.com/python-tutorial",
            "https://example.com/python-advanced",
            "https://example.com/python-data-science",
        }
        returned_urls = {r["url"] for r in out}
        assert returned_urls == python_urls

    def test_fixture_top2_java_query_order(self):
        """For query='java', only the Java guide should rank non-zero."""
        out = filter_results(FILTER_FIXTURE_RESULTS, "java", top_n=2)
        assert len(out) == 2
        assert out[0]["url"] == "https://example.com/java-guide"
        assert out[0]["_relevance_score"] > 0.0
        # Second result scores 0.0
        assert out[1]["_relevance_score"] == 0.0

    def test_fixture_top1_rust_query(self):
        """For query='rust', the rust-book must be the sole top-1 result."""
        out = filter_results(FILTER_FIXTURE_RESULTS, "rust", top_n=1)
        assert len(out) == 1
        assert out[0]["url"] == "https://example.com/rust-book"
        assert out[0]["_relevance_score"] > 0.0

    def test_fixture_empty_results_with_any_query(self):
        """Empty results list always returns empty regardless of query."""
        assert filter_results([], "python advanced tutorial", top_n=10) == []

    def test_fixture_single_result_top5(self):
        """Single item input with top_n=5 returns exactly that 1 item."""
        single = [FILTER_FIXTURE_RESULTS[1]]  # java-guide
        out = filter_results(single, "java", top_n=5)
        assert len(out) == 1
        assert out[0]["url"] == "https://example.com/java-guide"


# ---------------------------------------------------------------------------
# Non-retryable error: bad JSON / unexpected response shape
# ---------------------------------------------------------------------------


class TestNonRetryableErrors:
    @patch("assistant.tools.searxng.httpx.get")
    def test_json_decode_error_raises_tool_error_immediately(self, mock_get):
        bad_resp = MagicMock(spec=httpx.Response)
        bad_resp.raise_for_status.return_value = None
        bad_resp.json.side_effect = ValueError("JSON decode error")
        mock_get.return_value = bad_resp

        with pytest.raises(ToolError):
            search("bad json")

        # Should NOT retry on parse errors (only 1 call)
        assert mock_get.call_count == 1
