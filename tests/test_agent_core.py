"""Tests for the shared interface-integration core (assistant.agent_core).

Covers:
- default_responder: builds the /api/chat request correctly (model, messages
  mapped to role/content only, stream=False), returns the parsed reply text,
  and wraps HTTP/parse/JSON errors as OllamaError.
- handle_user_message:
  - input validation (empty/whitespace message_text)
  - resolves the canonical session and persists the updated context
  - appends user and assistant messages to working_memory with
    source_interface tagging
  - returns the responder's reply verbatim (data-fidelity)
  - delegates to maybe_auto_compress before persisting
  - builds a default SessionStore when none is provided
  - AC6.5: cross-interface continuity -- messages sent via different
    source_interface values accumulate in the same shared working_memory
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from assistant.agent_core import default_responder, handle_user_message
from assistant.llm import OllamaError
from assistant.long_term_memory import LongTermMemoryStore
from assistant.notes import NoteStore
from assistant.session_store import SessionStore
from assistant.tools.definitions import TOOL_DEFINITIONS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(
    status_code: int = 200,
    json_body: dict | None = None,
    raise_for_status_exc: Exception | None = None,
) -> MagicMock:
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = status_code
    mock_resp.json.return_value = (
        json_body if json_body is not None else {"message": {"role": "assistant", "content": "hi"}}
    )
    if raise_for_status_exc is not None:
        mock_resp.raise_for_status.side_effect = raise_for_status_exc
    else:
        mock_resp.raise_for_status.return_value = None
    return mock_resp


def _stub_responder(reply: str):
    return lambda messages: reply


# ---------------------------------------------------------------------------
# Tests for default_responder
# ---------------------------------------------------------------------------


class TestDefaultResponder:
    @patch("assistant.agent_core.httpx.post")
    def test_returns_reply_text(self, mock_post):
        mock_post.return_value = _make_response(
            json_body={"message": {"role": "assistant", "content": "Hello there!"}}
        )

        reply = default_responder([{"role": "user", "content": "hi"}])

        assert reply == "Hello there!"

    @patch("assistant.agent_core.httpx.post")
    def test_sends_request_to_api_chat_endpoint(self, mock_post):
        mock_post.return_value = _make_response()

        default_responder([{"role": "user", "content": "hi"}], base_url="http://localhost:11434")

        url = mock_post.call_args.args[0]
        assert url == "http://localhost:11434/api/chat"

    @patch("assistant.agent_core.httpx.post")
    def test_payload_contains_model_and_stream_false(self, mock_post):
        mock_post.return_value = _make_response()

        default_responder([{"role": "user", "content": "hi"}], model="gemma4:12b-mlx")

        payload = mock_post.call_args.kwargs["json"]
        assert payload["model"] == "gemma4:12b-mlx"
        assert payload["stream"] is False

    @patch("assistant.agent_core.httpx.post")
    def test_messages_mapped_to_role_and_content_only(self, mock_post):
        mock_post.return_value = _make_response()

        default_responder(
            [{"role": "user", "content": "hi", "source_interface": "telegram"}]
        )

        payload = mock_post.call_args.kwargs["json"]
        assert payload["messages"] == [{"role": "user", "content": "hi"}]

    @patch("assistant.agent_core.httpx.post")
    def test_http_error_raises_ollama_error(self, mock_post):
        mock_post.return_value = _make_response(
            status_code=500,
            raise_for_status_exc=httpx.HTTPStatusError(
                "500 Server Error", request=MagicMock(), response=MagicMock()
            ),
        )

        with pytest.raises(OllamaError):
            default_responder([{"role": "user", "content": "hi"}])

    @patch("assistant.agent_core.httpx.post")
    def test_connection_error_raises_ollama_error(self, mock_post):
        mock_post.side_effect = httpx.ConnectError("Connection refused")

        with pytest.raises(OllamaError):
            default_responder([{"role": "user", "content": "hi"}])

    @patch("assistant.agent_core.httpx.post")
    def test_malformed_response_raises_ollama_error(self, mock_post):
        mock_post.return_value = _make_response(json_body={"unexpected": "shape"})

        with pytest.raises(OllamaError):
            default_responder([{"role": "user", "content": "hi"}])

    @patch("assistant.agent_core.httpx.post")
    def test_invalid_json_raises_ollama_error(self, mock_post):
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.side_effect = ValueError("not json")
        mock_post.return_value = mock_resp

        with pytest.raises(OllamaError):
            default_responder([{"role": "user", "content": "hi"}])


# ---------------------------------------------------------------------------
# Tests for default_responder's tool-calling loop (AC1-3, AC23)
# ---------------------------------------------------------------------------


def _tool_call(name: str, arguments: dict | str) -> dict:
    return {"function": {"name": name, "arguments": arguments}}


class TestDefaultResponderToolCalling:
    @patch("assistant.agent_core.httpx.post")
    def test_payload_includes_tool_definitions(self, mock_post):
        mock_post.return_value = _make_response(
            json_body={"message": {"role": "assistant", "content": "hi"}}
        )

        default_responder([{"role": "user", "content": "hi"}])

        payload = mock_post.call_args.kwargs["json"]
        assert payload["tools"] == TOOL_DEFINITIONS

    @patch("assistant.agent_core.httpx.post")
    def test_no_tool_calls_returns_content_and_does_not_mutate_messages(self, mock_post):
        mock_post.return_value = _make_response(
            json_body={"message": {"role": "assistant", "content": "Hi there!"}}
        )
        messages = [{"role": "user", "content": "Hello"}]

        reply = default_responder(messages)

        assert reply == "Hi there!"
        assert messages == [{"role": "user", "content": "Hello"}]

    @patch("assistant.agent_core.dispatch_tool")
    @patch("assistant.agent_core.httpx.post")
    def test_single_tool_call_dispatches_and_returns_final_reply(self, mock_post, mock_dispatch):
        mock_post.side_effect = [
            _make_response(
                json_body={
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [_tool_call("get_weather", {"location": "Seoul"})],
                    }
                }
            ),
            _make_response(
                json_body={"message": {"role": "assistant", "content": "It's 22.5C and clear in Seoul."}}
            ),
        ]
        mock_dispatch.return_value = {
            "tool_name": "get_weather",
            "tool_input": {"location": "Seoul"},
            "tool_output": {"weather": {"temperature_c": 22.5}},
            "tool_status": "success",
            "tool_retry_count": 0,
            "tool_error_message": None,
        }
        messages = [{"role": "user", "content": "What's the weather in Seoul?"}]

        reply = default_responder(messages)

        assert reply == "It's 22.5C and clear in Seoul."
        mock_dispatch.assert_called_once_with("get_weather", {"location": "Seoul"})
        assert mock_post.call_count == 2

        # The tool-call and tool-result messages were recorded for persistence.
        assert messages[1]["role"] == "assistant"
        assert messages[1]["tool_calls"] == [_tool_call("get_weather", {"location": "Seoul"})]
        assert messages[2]["role"] == "tool"
        assert messages[2]["name"] == "get_weather"
        assert json.loads(messages[2]["content"])["tool_status"] == "success"

    @patch("assistant.agent_core.dispatch_tool")
    @patch("assistant.agent_core.httpx.post")
    def test_multi_tool_chain_loops_across_iterations(self, mock_post, mock_dispatch):
        mock_post.side_effect = [
            _make_response(
                json_body={
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [_tool_call("get_calendar_events", {"start_date": "2026-06-10", "end_date": "2026-06-11"})],
                    }
                }
            ),
            _make_response(
                json_body={
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [_tool_call("get_weather", {"location": "Seoul"})],
                    }
                }
            ),
            _make_response(
                json_body={"message": {"role": "assistant", "content": "You have a meeting; bring an umbrella."}}
            ),
        ]
        mock_dispatch.side_effect = [
            {
                "tool_name": "get_calendar_events",
                "tool_input": {"start_date": "2026-06-10", "end_date": "2026-06-11"},
                "tool_output": {"events": []},
                "tool_status": "success",
                "tool_retry_count": 0,
                "tool_error_message": None,
            },
            {
                "tool_name": "get_weather",
                "tool_input": {"location": "Seoul"},
                "tool_output": {"weather": {"weather_description": "Rain"}},
                "tool_status": "success",
                "tool_retry_count": 0,
                "tool_error_message": None,
            },
        ]
        messages = [{"role": "user", "content": "What's on my calendar and do I need an umbrella?"}]

        reply = default_responder(messages)

        assert reply == "You have a meeting; bring an umbrella."
        assert mock_dispatch.call_args_list == [
            (("get_calendar_events", {"start_date": "2026-06-10", "end_date": "2026-06-11"}), {}),
            (("get_weather", {"location": "Seoul"}), {}),
        ]
        assert mock_post.call_count == 3
        # 1 user message + 2x (assistant tool-call + tool result) = 5
        assert len(messages) == 5

    @patch("assistant.agent_core.dispatch_tool")
    @patch("assistant.agent_core.httpx.post")
    def test_failed_tool_result_is_passed_to_llm_and_surfaced(self, mock_post, mock_dispatch):
        error_message = "Open-Meteo API request failed after 3 attempts"
        mock_post.side_effect = [
            _make_response(
                json_body={
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [_tool_call("get_weather", {"location": "Atlantis"})],
                    }
                }
            ),
            _make_response(
                json_body={
                    "message": {
                        "role": "assistant",
                        "content": f"Sorry, I couldn't get the weather: {error_message}",
                    }
                }
            ),
        ]
        mock_dispatch.return_value = {
            "tool_name": "get_weather",
            "tool_input": {"location": "Atlantis"},
            "tool_output": {},
            "tool_status": "failed",
            "tool_retry_count": 2,
            "tool_error_message": error_message,
        }
        messages = [{"role": "user", "content": "What's the weather in Atlantis?"}]

        reply = default_responder(messages)

        assert error_message in reply
        tool_result = json.loads(messages[2]["content"])
        assert tool_result["tool_status"] == "failed"
        assert tool_result["tool_error_message"] == error_message

    @patch("assistant.agent_core.dispatch_tool")
    @patch("assistant.agent_core.httpx.post")
    def test_tool_call_arguments_as_json_string_are_parsed(self, mock_post, mock_dispatch):
        mock_post.side_effect = [
            _make_response(
                json_body={
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [_tool_call("get_weather", '{"location": "Seoul"}')],
                    }
                }
            ),
            _make_response(json_body={"message": {"role": "assistant", "content": "Done."}}),
        ]
        mock_dispatch.return_value = {
            "tool_name": "get_weather",
            "tool_input": {"location": "Seoul"},
            "tool_output": {},
            "tool_status": "success",
            "tool_retry_count": 0,
            "tool_error_message": None,
        }

        default_responder([{"role": "user", "content": "Weather in Seoul?"}])

        mock_dispatch.assert_called_once_with("get_weather", {"location": "Seoul"})

    @patch("assistant.agent_core.dispatch_tool")
    @patch("assistant.agent_core.httpx.post")
    def test_max_tool_iterations_forces_final_tool_free_reply(self, mock_post, mock_dispatch):
        looping_response = _make_response(
            json_body={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [_tool_call("get_weather", {"location": "Seoul"})],
                }
            }
        )
        final_response = _make_response(json_body={"message": {"role": "assistant", "content": "Final answer."}})
        mock_post.side_effect = [looping_response, looping_response, final_response]
        mock_dispatch.return_value = {
            "tool_name": "get_weather",
            "tool_input": {"location": "Seoul"},
            "tool_output": {},
            "tool_status": "success",
            "tool_retry_count": 0,
            "tool_error_message": None,
        }

        reply = default_responder([{"role": "user", "content": "Weather?"}], max_tool_iterations=2)

        assert reply == "Final answer."
        assert mock_post.call_count == 3
        # The final, forced call must not offer tools.
        final_payload = mock_post.call_args_list[-1].kwargs["json"]
        assert "tools" not in final_payload


# ---------------------------------------------------------------------------
# Tests for handle_user_message
# ---------------------------------------------------------------------------


class TestHandleUserMessage:
    def _make_store(self, tmp_path) -> SessionStore:
        return SessionStore(db_path=tmp_path / "memory.db")

    def test_empty_message_raises_value_error(self, tmp_path):
        store = self._make_store(tmp_path)

        with pytest.raises(ValueError):
            handle_user_message("web_ui", "", store=store, responder=_stub_responder("hi"))

    def test_whitespace_message_raises_value_error(self, tmp_path):
        store = self._make_store(tmp_path)

        with pytest.raises(ValueError):
            handle_user_message("web_ui", "   ", store=store, responder=_stub_responder("hi"))

    def test_unknown_interface_raises_value_error(self, tmp_path):
        store = self._make_store(tmp_path)

        with pytest.raises(ValueError):
            handle_user_message("sms", "hi", store=store, responder=_stub_responder("hi"))

    def test_returns_reply_from_responder(self, tmp_path):
        store = self._make_store(tmp_path)

        reply = handle_user_message("web_ui", "Hello", store=store, responder=_stub_responder("Hi there!"))

        assert reply == "Hi there!"

    def test_appends_user_and_assistant_messages(self, tmp_path):
        store = self._make_store(tmp_path)

        handle_user_message("web_ui", "Hello", store=store, responder=_stub_responder("Hi there!"))

        ctx = store.get_session("default")
        working_memory = ctx["working_memory"]
        assert working_memory[-2] == {"role": "user", "content": "Hello", "source_interface": "web_ui"}
        assert working_memory[-1] == {
            "role": "assistant",
            "content": "Hi there!",
            "source_interface": "web_ui",
        }

    def test_responder_receives_appended_user_message(self, tmp_path):
        store = self._make_store(tmp_path)
        captured: list[dict] = []

        def responder(messages):
            captured.extend(messages)
            return "ack"

        handle_user_message("web_ui", "What's the weather?", store=store, responder=responder)

        assert captured[-1] == {
            "role": "user",
            "content": "What's the weather?",
            "source_interface": "web_ui",
        }

    def test_persists_source_interface(self, tmp_path):
        store = self._make_store(tmp_path)

        handle_user_message("telegram", "Hello", store=store, responder=_stub_responder("Hi!"))

        ctx = store.get_session("default")
        assert ctx["source_interface"] == "telegram"

    def test_calls_maybe_auto_compress_before_persisting(self, tmp_path):
        store = self._make_store(tmp_path)
        sentinel_context = {
            "working_memory": ["compressed"],
            "session_memory": {"summary": "..."},
            "source_interface": "web_ui",
        }

        with patch("assistant.agent_core.maybe_auto_compress", return_value=sentinel_context) as mock_compress:
            handle_user_message("web_ui", "Hello", store=store, responder=_stub_responder("Hi!"))

        mock_compress.assert_called_once()
        ctx = store.get_session("default")
        assert ctx["working_memory"] == ["compressed"]
        assert ctx["session_memory"]["summary"] == "..."

    @patch("assistant.agent_core.SessionStore")
    def test_creates_default_store_when_none_provided(self, mock_store_cls):
        mock_store = MagicMock()
        mock_store.get_session.return_value = {
            "working_memory": [],
            "session_memory": {},
            "long_term_memory": [],
            "_meta": {"session_id": "session-123"},
        }
        mock_store_cls.return_value = mock_store

        reply = handle_user_message("web_ui", "Hello", responder=_stub_responder("Hi!"))

        assert reply == "Hi!"
        mock_store_cls.assert_called_once_with()
        mock_store.upsert_session.assert_called_once()

    # ------------------------------------------------------------------
    # AC6.5 -- cross-interface context continuity
    # ------------------------------------------------------------------

    def test_cross_interface_continuity_shares_working_memory(self, tmp_path):
        store = self._make_store(tmp_path)

        handle_user_message("web_ui", "My name is Alex", store=store, responder=_stub_responder("Nice to meet you, Alex!"))
        handle_user_message("telegram", "What's my name?", store=store, responder=_stub_responder("Your name is Alex."))

        ctx = store.get_session("default")
        working_memory = ctx["working_memory"]

        assert len(working_memory) == 4
        assert working_memory[0] == {"role": "user", "content": "My name is Alex", "source_interface": "web_ui"}
        assert working_memory[1] == {
            "role": "assistant",
            "content": "Nice to meet you, Alex!",
            "source_interface": "web_ui",
        }
        assert working_memory[2] == {"role": "user", "content": "What's my name?", "source_interface": "telegram"}
        assert working_memory[3] == {
            "role": "assistant",
            "content": "Your name is Alex.",
            "source_interface": "telegram",
        }

    def test_cross_interface_continuity_same_session_id(self, tmp_path):
        store = self._make_store(tmp_path)

        handle_user_message("web_ui", "Hello", store=store, responder=_stub_responder("Hi!"))
        ctx_after_web = store.get_session("default")

        handle_user_message("discord", "Still there?", store=store, responder=_stub_responder("Yes!"))
        ctx_after_discord = store.get_session("default")

        assert ctx_after_web["_meta"]["session_id"] == ctx_after_discord["_meta"]["session_id"]

    def test_responder_sees_full_history_across_interfaces(self, tmp_path):
        store = self._make_store(tmp_path)
        seen_histories: list[list[dict]] = []

        def responder(messages):
            seen_histories.append([dict(m) for m in messages])
            return "ok"

        handle_user_message("web_ui", "First message", store=store, responder=responder)
        handle_user_message("telegram", "Second message", store=store, responder=responder)

        # Second call's responder should see both the first turn and the new message.
        assert len(seen_histories[1]) == 3
        assert seen_histories[1][0]["content"] == "First message"
        assert seen_histories[1][1]["content"] == "ok"
        assert seen_histories[1][2]["content"] == "Second message"


# ---------------------------------------------------------------------------
# Tests for slash-command routing (AC14-16)
# ---------------------------------------------------------------------------


class TestSlashCommandRouting:
    def _make_store(self, tmp_path) -> SessionStore:
        return SessionStore(db_path=tmp_path / "memory.db")

    def test_note_command_routed_to_note_store(self, tmp_path):
        store = self._make_store(tmp_path)
        note_store = NoteStore(db_path=tmp_path / "memory.db")
        responder = MagicMock(side_effect=AssertionError("responder should not be called"))

        reply = handle_user_message(
            "web_ui", "/note Buy milk", store=store, responder=responder, note_store=note_store
        )

        assert "saved" in reply
        responder.assert_not_called()

        notes = note_store.list_notes("default")
        assert any(n["content"] == "Buy milk" for n in notes)

    def test_note_command_appends_to_working_memory(self, tmp_path):
        store = self._make_store(tmp_path)
        note_store = NoteStore(db_path=tmp_path / "memory.db")

        reply = handle_user_message(
            "web_ui", "/note Buy milk", store=store, responder=_stub_responder("unused"), note_store=note_store
        )

        ctx = store.get_session("default")
        working_memory = ctx["working_memory"]
        assert working_memory[-2] == {"role": "user", "content": "/note Buy milk", "source_interface": "web_ui"}
        assert working_memory[-1] == {"role": "assistant", "content": reply, "source_interface": "web_ui"}

    def test_remember_command_routed_to_long_term_memory_store(self, tmp_path):
        store = self._make_store(tmp_path)
        ltm_store = LongTermMemoryStore(db_path=tmp_path / "user.db")
        responder = MagicMock(side_effect=AssertionError("responder should not be called"))

        reply = handle_user_message(
            "telegram",
            "/remember I'm allergic to peanuts",
            store=store,
            responder=responder,
            long_term_memory_store=ltm_store,
        )

        assert "remember" in reply.lower()
        responder.assert_not_called()

        memories = ltm_store.list_memories("default")
        assert any(m["content"] == "I'm allergic to peanuts" for m in memories)

    def test_compress_command_routed_to_compression(self, tmp_path):
        store = self._make_store(tmp_path)

        # Build up more than _KEEP_RECENT_MESSAGES (6) messages of history.
        for i in range(4):
            handle_user_message(
                "web_ui", f"message {i}", store=store, responder=_stub_responder(f"reply {i}")
            )

        responder = MagicMock(side_effect=AssertionError("responder should not be called"))
        with patch("assistant.compression.httpx.post") as mock_post:
            mock_post.return_value = _make_response(
                json_body={"message": {"role": "assistant", "content": "summary text"}}
            )
            reply = handle_user_message("web_ui", "/compress", store=store, responder=responder)

        assert "Compressed" in reply
        responder.assert_not_called()
        mock_post.assert_called_once()

        ctx = store.get_session("default")
        assert ctx["session_memory"]["summary"] == "summary text"
        # The most recent 6 kept messages, plus the /compress exchange itself.
        assert len(ctx["working_memory"]) == 8
        assert ctx["working_memory"][-2] == {"role": "user", "content": "/compress", "source_interface": "web_ui"}
        assert ctx["working_memory"][-1] == {"role": "assistant", "content": reply, "source_interface": "web_ui"}

    def test_compress_command_with_no_history_returns_message_without_error(self, tmp_path):
        store = self._make_store(tmp_path)

        reply = handle_user_message("web_ui", "/compress", store=store, responder=_stub_responder("unused"))

        assert "no conversation history" in reply or "isn't enough" in reply

    def test_invalid_remember_command_returns_error_message_not_exception(self, tmp_path):
        store = self._make_store(tmp_path)
        ltm_store = LongTermMemoryStore(db_path=tmp_path / "user.db")

        reply = handle_user_message(
            "web_ui", "/remember", store=store, responder=_stub_responder("unused"), long_term_memory_store=ltm_store
        )

        assert "argument" in reply

    def test_non_command_message_still_uses_responder(self, tmp_path):
        store = self._make_store(tmp_path)

        reply = handle_user_message("web_ui", "Hello there", store=store, responder=_stub_responder("Hi!"))

        assert reply == "Hi!"

    def test_lookalike_command_not_routed(self, tmp_path):
        store = self._make_store(tmp_path)

        reply = handle_user_message("web_ui", "/notebook idea", store=store, responder=_stub_responder("Hi!"))

        assert reply == "Hi!"


# ---------------------------------------------------------------------------
# Integration tests: handle_user_message + default_responder tool-calling
# ---------------------------------------------------------------------------


class TestHandleUserMessageToolCalling:
    def _make_store(self, tmp_path) -> SessionStore:
        return SessionStore(db_path=tmp_path / "memory.db")

    @patch("assistant.agent_core.dispatch_tool")
    @patch("assistant.agent_core.httpx.post")
    def test_tool_call_round_trip_persisted_to_working_memory(self, mock_post, mock_dispatch, tmp_path):
        store = self._make_store(tmp_path)
        mock_post.side_effect = [
            _make_response(
                json_body={
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [_tool_call("get_weather", {"location": "Seoul"})],
                    }
                }
            ),
            _make_response(
                json_body={"message": {"role": "assistant", "content": "It's sunny in Seoul."}}
            ),
        ]
        mock_dispatch.return_value = {
            "tool_name": "get_weather",
            "tool_input": {"location": "Seoul"},
            "tool_output": {"weather": {"weather_description": "Clear"}},
            "tool_status": "success",
            "tool_retry_count": 0,
            "tool_error_message": None,
        }

        reply = handle_user_message("web_ui", "What's the weather in Seoul?", store=store)

        assert reply == "It's sunny in Seoul."

        working_memory = store.get_session("default")["working_memory"]
        assert working_memory[0] == {
            "role": "user",
            "content": "What's the weather in Seoul?",
            "source_interface": "web_ui",
        }
        assert working_memory[1]["role"] == "assistant"
        assert working_memory[1]["tool_calls"] == [_tool_call("get_weather", {"location": "Seoul"})]
        assert working_memory[2]["role"] == "tool"
        assert working_memory[2]["name"] == "get_weather"
        assert json.loads(working_memory[2]["content"])["tool_status"] == "success"
        assert working_memory[3] == {
            "role": "assistant",
            "content": "It's sunny in Seoul.",
            "source_interface": "web_ui",
        }

    @patch("assistant.agent_core.dispatch_tool")
    @patch("assistant.agent_core.httpx.post")
    def test_failed_tool_error_message_reaches_user_reply(self, mock_post, mock_dispatch, tmp_path):
        store = self._make_store(tmp_path)
        error_message = "Open-Meteo API request failed after 3 attempts"
        mock_post.side_effect = [
            _make_response(
                json_body={
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [_tool_call("get_weather", {"location": "Atlantis"})],
                    }
                }
            ),
            _make_response(
                json_body={
                    "message": {
                        "role": "assistant",
                        "content": f"Sorry, I couldn't get the weather: {error_message}",
                    }
                }
            ),
        ]
        mock_dispatch.return_value = {
            "tool_name": "get_weather",
            "tool_input": {"location": "Atlantis"},
            "tool_output": {},
            "tool_status": "failed",
            "tool_retry_count": 2,
            "tool_error_message": error_message,
        }

        reply = handle_user_message("web_ui", "What's the weather in Atlantis?", store=store)

        assert error_message in reply


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
