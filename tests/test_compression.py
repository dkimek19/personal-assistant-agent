"""Tests for working memory compression (assistant.compression).

Covers:
- build_compression_prompt: with/without previous summary, message content
  reproduced verbatim (data fidelity), empty message list
- summarize_messages: success (incl. /api/chat shape, empty messages
  short-circuit), HTTP/connection error retries with exponential backoff,
  non-retryable malformed-response / JSON-decode errors
- compress_working_memory: no-op below keep_recent, trims working_memory,
  folds older messages + previous summary via the summarizer, preserves
  other context keys, does not mutate input, keep_recent=0 edge case
- maybe_auto_compress: no-op below/at threshold (summarizer not called),
  compresses above threshold
- handle_compress_command: validation, no-session / too-few-messages
  responses, successful compression persists via SessionStore and reports
  the number of messages folded
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from assistant.compression import (
    build_compression_prompt,
    compress_working_memory,
    handle_compress_command,
    maybe_auto_compress,
    summarize_messages,
)
from assistant.llm import OllamaError
from assistant.session_store import SessionStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stub_summarizer(messages, previous_summary):
    return f"summary of {len(messages)} messages (prev={previous_summary!r})"


def _make_response(
    status_code: int = 200,
    json_body: dict | None = None,
    raise_for_status_exc: Exception | None = None,
) -> MagicMock:
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_body if json_body is not None else {"response": "summary text"}
    if raise_for_status_exc is not None:
        mock_resp.raise_for_status.side_effect = raise_for_status_exc
    else:
        mock_resp.raise_for_status.return_value = None
    return mock_resp


# ---------------------------------------------------------------------------
# Tests for build_compression_prompt
# ---------------------------------------------------------------------------


class TestBuildCompressionPrompt:
    def test_includes_conversation_header_and_messages_verbatim(self):
        messages = [
            {"role": "user", "content": "My birthday is March 5th"},
            {"role": "assistant", "content": "Got it, noted!"},
        ]

        prompt = build_compression_prompt(messages)

        assert "Conversation:" in prompt
        assert "user: My birthday is March 5th" in prompt
        assert "assistant: Got it, noted!" in prompt

    def test_omits_existing_summary_section_when_no_previous_summary(self):
        prompt = build_compression_prompt([{"role": "user", "content": "hi"}])

        assert "Existing summary" not in prompt

    def test_includes_previous_summary_when_provided(self):
        prompt = build_compression_prompt(
            [{"role": "user", "content": "hi"}],
            previous_summary="User likes coffee.",
        )

        assert "Existing summary of earlier conversation:" in prompt
        assert "User likes coffee." in prompt

    def test_empty_messages_still_returns_prompt(self):
        prompt = build_compression_prompt([])

        assert "Conversation:" in prompt


# ---------------------------------------------------------------------------
# Tests for summarize_messages (success)
# ---------------------------------------------------------------------------


class TestSummarizeMessagesSuccess:
    def test_empty_messages_returns_previous_summary_without_calling_llm(self):
        with patch("assistant.compression.httpx.post") as mock_post:
            result = summarize_messages([], previous_summary="Existing summary")

        assert result == "Existing summary"
        mock_post.assert_not_called()

    @patch("assistant.compression.httpx.post")
    def test_returns_summary_string(self, mock_post):
        mock_post.return_value = _make_response(json_body={"response": "User prefers email updates."})

        result = summarize_messages([{"role": "user", "content": "Email me updates"}])

        assert result == "User prefers email updates."

    @patch("assistant.compression.httpx.post")
    def test_supports_chat_response_shape(self, mock_post):
        mock_post.return_value = _make_response(
            json_body={"message": {"role": "assistant", "content": "Chat-shaped summary"}}
        )

        result = summarize_messages([{"role": "user", "content": "hi"}])

        assert result == "Chat-shaped summary"

    @patch("assistant.compression.httpx.post")
    def test_sends_request_to_api_generate_endpoint(self, mock_post):
        mock_post.return_value = _make_response()

        summarize_messages([{"role": "user", "content": "hi"}], base_url="http://localhost:11434")

        url = mock_post.call_args.args[0]
        assert url == "http://localhost:11434/api/generate"

    @patch("assistant.compression.httpx.post")
    def test_payload_contains_model_and_stream_false(self, mock_post):
        mock_post.return_value = _make_response()

        summarize_messages([{"role": "user", "content": "hi"}], model="gemma4:12b-mlx")

        payload = mock_post.call_args.kwargs["json"]
        assert payload["model"] == "gemma4:12b-mlx"
        assert payload["stream"] is False

    @patch("assistant.compression.httpx.post")
    def test_prompt_contains_message_content(self, mock_post):
        mock_post.return_value = _make_response()

        summarize_messages([{"role": "user", "content": "Remember my dog's name is Rex"}])

        payload = mock_post.call_args.kwargs["json"]
        assert "Rex" in payload["prompt"]


# ---------------------------------------------------------------------------
# Tests for summarize_messages (errors / retries)
# ---------------------------------------------------------------------------


class TestSummarizeMessagesErrors:
    @patch("assistant.compression.time.sleep", return_value=None)
    @patch("assistant.compression.httpx.post")
    def test_retries_on_http_500_and_raises_after_exhaustion(self, mock_post, mock_sleep):
        mock_post.return_value = _make_response(
            status_code=500,
            raise_for_status_exc=httpx.HTTPStatusError(
                "500 Server Error", request=MagicMock(), response=MagicMock()
            ),
        )

        with pytest.raises(OllamaError) as exc_info:
            summarize_messages([{"role": "user", "content": "hi"}], max_retries=2, base_backoff=0.01)

        assert "after 3 attempts" in str(exc_info.value)
        assert mock_post.call_count == 3

    @patch("assistant.compression.time.sleep", return_value=None)
    @patch("assistant.compression.httpx.post")
    def test_retries_on_connection_error(self, mock_post, mock_sleep):
        mock_post.side_effect = httpx.ConnectError("Connection refused")

        with pytest.raises(OllamaError):
            summarize_messages([{"role": "user", "content": "hi"}], max_retries=2, base_backoff=0.01)

        assert mock_post.call_count == 3

    @patch("assistant.compression.time.sleep", return_value=None)
    @patch("assistant.compression.httpx.post")
    def test_exponential_backoff_sleep_durations(self, mock_post, mock_sleep):
        mock_post.side_effect = httpx.ConnectError("Connection refused")

        with pytest.raises(OllamaError):
            summarize_messages([{"role": "user", "content": "hi"}], max_retries=2, base_backoff=1.0)

        calls = mock_sleep.call_args_list
        assert calls[0][0][0] == 1.0
        assert calls[1][0][0] == 2.0

    @patch("assistant.compression.time.sleep", return_value=None)
    @patch("assistant.compression.httpx.post")
    def test_malformed_response_raises_without_retry(self, mock_post, mock_sleep):
        mock_post.return_value = _make_response(json_body={"unexpected": "shape"})

        with pytest.raises(OllamaError):
            summarize_messages([{"role": "user", "content": "hi"}], max_retries=2, base_backoff=0.01)

        assert mock_post.call_count == 1
        mock_sleep.assert_not_called()

    @patch("assistant.compression.time.sleep", return_value=None)
    @patch("assistant.compression.httpx.post")
    def test_invalid_json_raises_without_retry(self, mock_post, mock_sleep):
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.side_effect = ValueError("not json")
        mock_post.return_value = mock_resp

        with pytest.raises(OllamaError):
            summarize_messages([{"role": "user", "content": "hi"}], max_retries=2, base_backoff=0.01)

        assert mock_post.call_count == 1
        mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# Tests for compress_working_memory
# ---------------------------------------------------------------------------


class TestCompressWorkingMemory:
    def test_returns_unchanged_when_at_or_below_keep_recent(self):
        context = {
            "working_memory": [{"role": "user", "content": "hi"}],
            "session_memory": {},
        }

        result = compress_working_memory(context, keep_recent=6, summarizer=_stub_summarizer)

        assert result is context

    def test_trims_working_memory_to_keep_recent(self):
        messages = [{"role": "user", "content": f"msg {i}"} for i in range(10)]
        context = {"working_memory": messages, "session_memory": {}}

        result = compress_working_memory(context, keep_recent=3, summarizer=_stub_summarizer)

        assert result["working_memory"] == messages[-3:]

    def test_summary_set_in_session_memory(self):
        messages = [{"role": "user", "content": f"msg {i}"} for i in range(10)]
        context = {"working_memory": messages, "session_memory": {}}

        result = compress_working_memory(context, keep_recent=3, summarizer=_stub_summarizer)

        assert result["session_memory"]["summary"] == "summary of 7 messages (prev='')"

    def test_summarizer_receives_previous_summary(self):
        messages = [{"role": "user", "content": f"msg {i}"} for i in range(10)]
        context = {
            "working_memory": messages,
            "session_memory": {"summary": "earlier stuff"},
        }

        result = compress_working_memory(context, keep_recent=3, summarizer=_stub_summarizer)

        assert "earlier stuff" in result["session_memory"]["summary"]

    def test_does_not_mutate_input_context(self):
        messages = [{"role": "user", "content": f"msg {i}"} for i in range(10)]
        context = {"working_memory": messages, "session_memory": {}}

        compress_working_memory(context, keep_recent=3, summarizer=_stub_summarizer)

        assert context["working_memory"] == messages
        assert "summary" not in context["session_memory"]

    def test_preserves_other_context_keys(self):
        messages = [{"role": "user", "content": f"msg {i}"} for i in range(10)]
        context = {
            "working_memory": messages,
            "session_memory": {},
            "source_interface": "telegram",
            "long_term_memory": [],
        }

        result = compress_working_memory(context, keep_recent=3, summarizer=_stub_summarizer)

        assert result["source_interface"] == "telegram"
        assert result["long_term_memory"] == []

    def test_keep_recent_zero_summarizes_all_and_keeps_none(self):
        messages = [{"role": "user", "content": f"msg {i}"} for i in range(5)]
        context = {"working_memory": messages, "session_memory": {}}

        result = compress_working_memory(context, keep_recent=0, summarizer=_stub_summarizer)

        assert result["working_memory"] == []
        assert result["session_memory"]["summary"] == "summary of 5 messages (prev='')"


# ---------------------------------------------------------------------------
# Tests for maybe_auto_compress
# ---------------------------------------------------------------------------


class TestMaybeAutoCompress:
    def test_below_threshold_returns_unchanged_without_calling_summarizer(self):
        messages = [{"role": "user", "content": f"msg {i}"} for i in range(5)]
        context = {"working_memory": messages, "session_memory": {}}
        summarizer = MagicMock()

        result = maybe_auto_compress(context, threshold=20, keep_recent=6, summarizer=summarizer)

        assert result is context
        summarizer.assert_not_called()

    def test_exactly_at_threshold_does_not_compress(self):
        messages = [{"role": "user", "content": f"msg {i}"} for i in range(20)]
        context = {"working_memory": messages, "session_memory": {}}
        summarizer = MagicMock()

        result = maybe_auto_compress(context, threshold=20, keep_recent=6, summarizer=summarizer)

        assert result is context
        summarizer.assert_not_called()

    def test_above_threshold_triggers_compression(self):
        messages = [{"role": "user", "content": f"msg {i}"} for i in range(25)]
        context = {"working_memory": messages, "session_memory": {}}
        summarizer = MagicMock(return_value="auto summary")

        result = maybe_auto_compress(context, threshold=20, keep_recent=6, summarizer=summarizer)

        assert len(result["working_memory"]) == 6
        assert result["session_memory"]["summary"] == "auto summary"


# ---------------------------------------------------------------------------
# Tests for handle_compress_command
# ---------------------------------------------------------------------------


class TestHandleCompressCommand:
    def _make_store(self, tmp_path) -> SessionStore:
        return SessionStore(db_path=tmp_path / "memory.db")

    def test_non_compress_command_raises_value_error(self, tmp_path):
        store = self._make_store(tmp_path)

        with pytest.raises(ValueError):
            handle_compress_command("user_1", "/note something", store)

    def test_no_session_returns_message(self, tmp_path):
        store = self._make_store(tmp_path)

        response = handle_compress_command("user_1", "/compress", store)

        assert "no conversation history" in response

    def test_too_few_messages_returns_message(self, tmp_path):
        store = self._make_store(tmp_path)
        store.upsert_session(
            "user_1",
            {"working_memory": [{"role": "user", "content": "hi"}], "session_memory": {}},
        )

        response = handle_compress_command("user_1", "/compress", store, keep_recent=6)

        assert "isn't enough" in response

    def test_successful_compression_persists_and_reports_count(self, tmp_path):
        store = self._make_store(tmp_path)
        messages = [{"role": "user", "content": f"msg {i}"} for i in range(10)]
        store.upsert_session("user_1", {"working_memory": messages, "session_memory": {}})

        response = handle_compress_command(
            "user_1", "/compress", store, keep_recent=3, summarizer=_stub_summarizer
        )

        assert "Compressed 7" in response

        ctx = store.get_session("user_1")
        assert ctx["working_memory"] == messages[-3:]
        assert ctx["session_memory"]["summary"] == "summary of 7 messages (prev='')"

    def test_preserves_other_session_fields_after_persist(self, tmp_path):
        store = self._make_store(tmp_path)
        messages = [{"role": "user", "content": f"msg {i}"} for i in range(10)]
        store.upsert_session(
            "user_1",
            {
                "working_memory": messages,
                "session_memory": {},
                "source_interface": "telegram",
                "long_term_memory": [],
            },
        )

        handle_compress_command("user_1", "/compress", store, keep_recent=3, summarizer=_stub_summarizer)

        ctx = store.get_session("user_1")
        assert ctx["source_interface"] == "telegram"
        assert ctx["long_term_memory"] == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
