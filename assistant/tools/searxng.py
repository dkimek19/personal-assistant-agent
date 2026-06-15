"""SearXNG search client module.

Provides a `search(query)` function that queries a local SearXNG instance
and returns a list of result dicts with url, title, and snippet fields.

Retry policy: exponential backoff with up to 2 retries (3 total attempts),
then raises ToolError with an explicit user-facing error message.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEARXNG_BASE_URL: str = os.environ.get("SEARXNG_URL", "http://localhost:8888")

# Retry policy (matches project-wide constraint: 2 retries, exponential backoff)
_MAX_RETRIES: int = 2
_BACKOFF_BASE_SECONDS: float = 1.0
_REQUEST_TIMEOUT_SECONDS: float = 10.0

# Default number of results to request
_DEFAULT_NUM_RESULTS: int = 5


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ToolError(RuntimeError):
    """Raised when a tool call fails after exhausting all retries."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_results(raw_results: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Extract url, title, and snippet from raw SearXNG result objects.

    SearXNG uses the key ``content`` for the snippet text.  We map it to
    ``snippet`` so callers get a consistent interface regardless of the
    underlying search engine.

    Args:
        raw_results: List of raw result dicts from the SearXNG JSON response.

    Returns:
        List of dicts with keys ``url``, ``title``, ``snippet``.
        Missing fields are replaced with an empty string rather than raising.
    """
    parsed: list[dict[str, str]] = []
    for item in raw_results:
        parsed.append(
            {
                "url": str(item.get("url", "")),
                "title": str(item.get("title", "")),
                "snippet": str(item.get("content", "")),
            }
        )
    return parsed


def _build_params(query: str, num_results: int) -> dict[str, Any]:
    """Build the query-string parameters for the SearXNG /search endpoint."""
    return {
        "q": query,
        "format": "json",
        "pageno": 1,
        "results_count": num_results,
        # Ask for text/general category to avoid news/images noise
        "categories": "general",
        "language": "auto",
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _score_result(result: dict[str, Any], query_terms: list[str]) -> float:
    """Compute a relevance score for a single result given query terms.

    Scoring weights per term occurrence:
        - title   → 2.0 points  (most signal)
        - snippet → 1.0 point
        - url     → 0.5 points  (least signal)

    Args:
        result:      A result dict with optional keys ``title``, ``snippet``,
                     ``url`` (and any SearXNG raw keys).
        query_terms: Lowercase tokens derived from the user query.

    Returns:
        A float >= 0.0.  Zero when ``query_terms`` is empty.
    """
    if not query_terms:
        return 0.0

    title = str(result.get("title", "")).lower()
    snippet = str(result.get("snippet", "") or result.get("content", "")).lower()
    url = str(result.get("url", "")).lower()

    score = 0.0
    for term in query_terms:
        if not term:
            continue
        score += title.count(term) * 2.0
        score += snippet.count(term) * 1.0
        score += url.count(term) * 0.5

    return score


def filter_results(
    results: list[dict],
    query: str,
    top_n: int,
) -> list[dict]:
    """Rank and filter search results by relevance to the query.

    Each result is scored by counting how often query terms appear in the
    ``title`` (weight 2×), ``snippet`` / ``content`` (weight 1×), and
    ``url`` (weight 0.5×).  The ``top_n`` highest-scoring results are
    returned in descending score order.

    A ``_relevance_score`` key is injected into every returned dict so
    callers can inspect the ranking without modifying the original dicts.

    Args:
        results: Raw result dicts, typically from :func:`search` or
                 directly from a SearXNG JSON response.  Each dict should
                 have ``url``, ``title``, and ``snippet`` / ``content`` keys.
        query:   The original search query string.  Case-insensitive; leading
                 and trailing whitespace is ignored.
        top_n:   Maximum number of results to return.  If fewer results exist
                 than ``top_n``, all results are returned (ranked).

    Returns:
        A list of **at most** ``top_n`` result dicts sorted by descending
        relevance score.  Original dicts are shallow-copied; the source list
        is never mutated.

    Edge cases:
        - Empty ``results``         → returns ``[]``.
        - ``top_n <= 0``            → returns ``[]``.
        - ``len(results) < top_n``  → returns all results, ranked.
        - Empty / whitespace query  → all results score 0.0 and are
          returned in their original order (stable sort preserves order).
    """
    if not results:
        return []
    if top_n <= 0:
        return []

    query_terms: list[str] = (
        [t for t in query.lower().split() if t]
        if query and query.strip()
        else []
    )

    # Build (score, original_index, result) tuples for stable tiebreaking
    scored: list[tuple[float, int, dict]] = [
        (_score_result(r, query_terms), idx, r)
        for idx, r in enumerate(results)
    ]

    # Sort descending by score; use original index as tiebreaker (stable order)
    scored.sort(key=lambda x: (-x[0], x[1]))

    filtered: list[dict] = []
    for score, _idx, result in scored[:top_n]:
        r = dict(result)               # shallow copy — never mutate caller's data
        r["_relevance_score"] = score
        filtered.append(r)

    return filtered


def search(
    query: str,
    *,
    num_results: int = _DEFAULT_NUM_RESULTS,
    base_url: str = SEARXNG_BASE_URL,
) -> list[dict]:
    """Search SearXNG and return a list of result dicts.

    Each result dict contains:
        - ``url``     (str): The canonical URL of the result.
        - ``title``   (str): The page title.
        - ``snippet`` (str): A short excerpt / description.

    The raw tool output is **never modified** by downstream LLM logic —
    formatting only is permitted (data_fidelity principle).

    Args:
        query:       The search query string.
        num_results: Maximum number of results to return (default 5).
        base_url:    Base URL of the SearXNG instance.  Reads from the
                     ``SEARXNG_URL`` environment variable by default;
                     falls back to ``http://localhost:8888``.

    Returns:
        List of result dicts ordered by SearXNG's relevance ranking.

    Raises:
        ToolError: When all retry attempts are exhausted.  The message is
                   human-readable and safe to surface directly to the user.
    """
    if not query or not query.strip():
        raise ValueError("search() requires a non-empty query string")

    url = f"{base_url.rstrip('/')}/search"
    params = _build_params(query.strip(), num_results)

    last_error: Exception | None = None

    for attempt in range(_MAX_RETRIES + 1):  # attempts: 0, 1, 2
        if attempt > 0:
            backoff = _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                "SearXNG search attempt %d/%d failed; retrying in %.1fs",
                attempt,
                _MAX_RETRIES,
                backoff,
            )
            time.sleep(backoff)

        try:
            logger.debug("SearXNG search [attempt %d]: %r", attempt, query)
            response = httpx.get(
                url,
                params=params,
                timeout=_REQUEST_TIMEOUT_SECONDS,
                follow_redirects=True,
            )
            response.raise_for_status()

            data: dict[str, Any] = response.json()
            raw_results: list[dict[str, Any]] = data.get("results", [])
            results = _parse_results(raw_results)

            logger.debug(
                "SearXNG returned %d results for query %r", len(results), query
            )
            return results

        except httpx.HTTPStatusError as exc:
            last_error = exc
            logger.error(
                "SearXNG HTTP error %d for query %r: %s",
                exc.response.status_code,
                query,
                exc,
            )
        except httpx.RequestError as exc:
            last_error = exc
            logger.error(
                "SearXNG connection error for query %r: %s", query, exc
            )
        except (ValueError, KeyError) as exc:
            # JSON parse error or unexpected response shape — not retryable
            raise ToolError(
                f"Web search failed: unexpected response format from SearXNG — {exc}"
            ) from exc

    raise ToolError(
        f"Web search failed after {_MAX_RETRIES + 1} attempts for query {query!r}. "
        f"Please try again later. (Last error: {last_error})"
    )
