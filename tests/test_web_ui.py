"""Tests for the Web UI interface adapter (assistant.interfaces.web_ui).

Covers:
- POST /chat returns the responder's reply and persists the conversation
  turn to the shared SessionStore (AC6.2).
- Repeated requests accumulate working_memory across calls.
- Empty/whitespace message -> 400.
- OllamaError from the responder -> 503.
- AC6.5: a session started via the Web UI is visible to other interfaces
  reading from the same SessionStore.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from assistant.interfaces.web_ui import NOTES_LIMIT, app, get_note_store, get_responder, get_store
from assistant.llm import OllamaError
from assistant.notes import NoteStore
from assistant.session_resolver import CANONICAL_USER_ID
from assistant.session_store import SessionStore
from assistant.tools.calendar import CalendarEvent
from assistant.tools.weather import WeatherReport


@pytest.fixture
def store(tmp_path):
    return SessionStore(db_path=tmp_path / "memory.db")


@pytest.fixture
def note_store(tmp_path):
    return NoteStore(db_path=tmp_path / "notes.db")


@pytest.fixture
def client(store, note_store):
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_responder] = lambda: (lambda messages: "stub reply")
    app.dependency_overrides[get_note_store] = lambda: note_store
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Sub-AC 1 (governed dispatch) -- GET / serves assistant/static/index.html
# ---------------------------------------------------------------------------


class TestStaticIndexServing:
    """The FastAPI backend serves assistant/static/index.html as a static
    file at the application root (``http://localhost:8000/``)."""

    def test_root_returns_200_with_html_content_type(self, client):
        response = client.get("/")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")

    def test_root_returns_dashboard_layout_structure(self, client):
        response = client.get("/")

        assert response.status_code == 200
        body = response.text

        # The dashboard's outer layout, sidebar, and main chat area
        # containers (Sub-AC 8 / AC2) are present in the served document.
        assert 'class="app-layout"' in body
        assert 'id="sidebar"' in body
        assert 'id="main"' in body
        assert 'id="chat-messages"' in body

        # The three read-only sidebar widget cards are present.
        assert 'id="widget-weather"' in body
        assert 'id="widget-calendar"' in body
        assert 'id="widget-memo"' in body

    def test_root_matches_index_html_file_on_disk(self, client):
        index_path = (
            Path(__file__).resolve().parent.parent / "assistant" / "static" / "index.html"
        )

        response = client.get("/")

        assert response.status_code == 200
        assert response.text == index_path.read_text(encoding="utf-8")


class TestChatEndpoint:
    def test_returns_reply(self, client):
        response = client.post("/chat", json={"message": "Hello"})

        assert response.status_code == 200
        assert response.json() == {"reply": "stub reply"}

    def test_persists_conversation_to_session_store(self, client, store):
        client.post("/chat", json={"message": "Hello"})

        ctx = store.get_session("default")
        assert ctx["working_memory"][-2] == {
            "role": "user",
            "content": "Hello",
            "source_interface": "web_ui",
        }
        assert ctx["working_memory"][-1] == {
            "role": "assistant",
            "content": "stub reply",
            "source_interface": "web_ui",
        }

    def test_accumulates_working_memory_across_requests(self, client, store):
        client.post("/chat", json={"message": "First"})
        client.post("/chat", json={"message": "Second"})

        ctx = store.get_session("default")
        assert len(ctx["working_memory"]) == 4

    def test_empty_message_returns_400(self, client):
        response = client.post("/chat", json={"message": ""})

        assert response.status_code == 400

    def test_whitespace_message_returns_400(self, client):
        response = client.post("/chat", json={"message": "   "})

        assert response.status_code == 400

    def test_ollama_error_returns_503(self, store):
        def failing_responder(messages):
            raise OllamaError("Chat request to Ollama failed: connection refused")

        app.dependency_overrides[get_store] = lambda: store
        app.dependency_overrides[get_responder] = lambda: failing_responder
        try:
            client = TestClient(app)
            response = client.post("/chat", json={"message": "Hello"})
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 503

    # ------------------------------------------------------------------
    # AC6.5 -- cross-interface continuity
    # ------------------------------------------------------------------

    def test_conversation_visible_to_other_interfaces_via_shared_store(self, client, store):
        client.post("/chat", json={"message": "My name is Alex"})

        # Simulate a different interface reading from the same store.
        ctx = store.get_session("default")
        assert ctx["working_memory"][0]["content"] == "My name is Alex"
        assert ctx["source_interface"] == "web_ui"


# ---------------------------------------------------------------------------
# Sub-AC 1 -- GET /api/weather
# ---------------------------------------------------------------------------

_WEATHER_REPORT = WeatherReport(
    location_name="Seoul",
    latitude=37.57,
    longitude=126.98,
    temperature_c=22.5,
    wind_speed_kmh=8.0,
    weather_code=0,
    weather_description="Clear sky",
    observation_time="2026-06-10T10:00:00Z",
)


class TestWeatherEndpoint:
    def test_returns_200_with_expected_schema_for_default_city(self, client):
        with patch(
            "assistant.tools.dispatch.get_current_weather", return_value=_WEATHER_REPORT
        ) as mock_get_weather:
            response = client.get("/api/weather")

        assert response.status_code == 200

        # Thin pass-through: dispatch_tool -> get_current_weather called
        # directly with the configured default city, no LLM round-trip.
        mock_get_weather.assert_called_once_with("Seoul")

        body = response.json()
        assert body == {
            "location_name": "Seoul",
            "latitude": 37.57,
            "longitude": 126.98,
            "temperature_c": 22.5,
            "wind_speed_kmh": 8.0,
            "weather_code": 0,
            "weather_description": "Clear sky",
            "observation_time": "2026-06-10T10:00:00Z",
        }

        # Response shape matches the weather_data ontology concept exactly.
        assert set(body.keys()) == {
            "location_name",
            "latitude",
            "longitude",
            "temperature_c",
            "wind_speed_kmh",
            "weather_code",
            "weather_description",
            "observation_time",
        }

    def test_returns_502_when_weather_tool_fails(self, client):
        with patch(
            "assistant.tools.dispatch.get_current_weather",
            side_effect=RuntimeError("Unable to retrieve weather for 'Seoul' after 3 attempts."),
        ):
            response = client.get("/api/weather")

        assert response.status_code == 502
        assert "detail" in response.json()


# ---------------------------------------------------------------------------
# Sub-AC 2 -- GET /api/calendar/today
# ---------------------------------------------------------------------------

_CALENDAR_EVENT = CalendarEvent(
    calendar_event_id="evt_001",
    calendar_event_title="Team Standup",
    calendar_event_time="2026-06-11T09:00:00+09:00",
    calendar_alert_time="2026-06-11T08:30:00+09:00",
    end_time="2026-06-11T09:30:00+09:00",
    description="Daily team sync",
    location="Zoom",
    attendees=["alice@example.com", "bob@example.com"],
    is_all_day=False,
    html_link="https://calendar.google.com/event?eid=evt_001",
)


class TestCalendarTodayEndpoint:
    def test_returns_200_with_expected_schema_for_today(self, client):
        with patch(
            "assistant.tools.dispatch.get_calendar_events", return_value=[_CALENDAR_EVENT]
        ) as mock_get_events:
            response = client.get("/api/calendar/today")

        assert response.status_code == 200

        # Thin pass-through: dispatch_tool -> get_calendar_events called
        # directly with today's date as both start and end, no LLM round-trip.
        today = date.today().isoformat()
        mock_get_events.assert_called_once_with(today, today)

        body = response.json()
        assert body == {"events": [_CALENDAR_EVENT.to_dict()]}

        # Response shape matches the calendar_events ontology concept.
        assert set(body.keys()) == {"events"}
        assert set(body["events"][0].keys()) == {
            "calendar_event_id",
            "calendar_event_title",
            "calendar_event_time",
            "calendar_alert_time",
            "end_time",
            "description",
            "location",
            "attendees",
            "is_all_day",
            "html_link",
        }

    def test_returns_200_with_empty_list_when_no_events_today(self, client):
        with patch("assistant.tools.dispatch.get_calendar_events", return_value=[]):
            response = client.get("/api/calendar/today")

        assert response.status_code == 200
        assert response.json() == {"events": []}

    def test_returns_502_when_calendar_tool_fails(self, client):
        with patch(
            "assistant.tools.dispatch.get_calendar_events",
            side_effect=RuntimeError("Unable to retrieve calendar events after 3 attempts."),
        ):
            response = client.get("/api/calendar/today")

        assert response.status_code == 502
        assert "detail" in response.json()


# ---------------------------------------------------------------------------
# Sub-AC 2 (governed dispatch) -- GET /api/calendar
# ---------------------------------------------------------------------------


class TestCalendarEndpoint:
    """``GET /api/calendar`` is a thin pass-through to
    ``calendar.get_calendar_events`` (via ``dispatch_tool``), mirroring
    ``GET /api/calendar/today``."""

    def test_returns_200_and_invokes_get_calendar_events(self, client):
        with patch(
            "assistant.tools.dispatch.get_calendar_events", return_value=[_CALENDAR_EVENT]
        ) as mock_get_events:
            response = client.get("/api/calendar")

        assert response.status_code == 200

        # Thin pass-through: dispatch_tool -> get_calendar_events called
        # directly with today's date as both start and end, no LLM round-trip.
        today = date.today().isoformat()
        mock_get_events.assert_called_once_with(today, today)

        body = response.json()
        assert body == {"events": [_CALENDAR_EVENT.to_dict()]}

        # Response shape matches the calendar_events ontology concept.
        assert set(body.keys()) == {"events"}
        assert set(body["events"][0].keys()) == {
            "calendar_event_id",
            "calendar_event_title",
            "calendar_event_time",
            "calendar_alert_time",
            "end_time",
            "description",
            "location",
            "attendees",
            "is_all_day",
            "html_link",
        }

    def test_returns_200_with_empty_list_when_no_events_today(self, client):
        with patch("assistant.tools.dispatch.get_calendar_events", return_value=[]):
            response = client.get("/api/calendar")

        assert response.status_code == 200
        assert response.json() == {"events": []}

    def test_returns_502_when_calendar_tool_fails(self, client):
        with patch(
            "assistant.tools.dispatch.get_calendar_events",
            side_effect=RuntimeError("Unable to retrieve calendar events after 3 attempts."),
        ):
            response = client.get("/api/calendar")

        assert response.status_code == 502
        assert "detail" in response.json()


# ---------------------------------------------------------------------------
# Sub-AC 3 -- GET /api/notes
# ---------------------------------------------------------------------------


class TestNotesEndpoint:
    def test_returns_200_with_expected_schema(self, client, note_store):
        note_store.add_note(CANONICAL_USER_ID, "Buy milk")
        note_store.add_note(CANONICAL_USER_ID, "Call dentist")

        response = client.get("/api/notes")

        assert response.status_code == 200

        body = response.json()

        # Response shape matches the `notes` ontology concept exactly.
        assert set(body.keys()) == {"notes"}
        assert len(body["notes"]) == 2

        # Most recently created first (NoteStore.list_notes ordering).
        assert body["notes"][0]["content"] == "Call dentist"
        assert body["notes"][1]["content"] == "Buy milk"

        for note in body["notes"]:
            assert set(note.keys()) == {"note_id", "content", "created_at", "updated_at"}
            assert isinstance(note["note_id"], int)
            assert isinstance(note["content"], str)
            assert isinstance(note["created_at"], str)
            assert isinstance(note["updated_at"], str)

    def test_returns_200_with_empty_list_when_no_notes(self, client):
        response = client.get("/api/notes")

        assert response.status_code == 200
        assert response.json() == {"notes": []}

    def test_thin_pass_through_to_note_store_list_notes(self, client, note_store):
        note_store.add_note(CANONICAL_USER_ID, "Buy milk")

        with patch.object(
            note_store, "list_notes", wraps=note_store.list_notes
        ) as mock_list_notes:
            response = client.get("/api/notes")

        assert response.status_code == 200
        # Thin pass-through: NoteStore.list_notes called directly for the
        # canonical (single-user) user_id, no LLM round-trip.
        mock_list_notes.assert_called_once_with(CANONICAL_USER_ID, limit=NOTES_LIMIT)

    def test_returns_at_most_notes_limit(self, client, note_store):
        for i in range(NOTES_LIMIT + 5):
            note_store.add_note(CANONICAL_USER_ID, f"Note {i}")

        response = client.get("/api/notes")

        assert response.status_code == 200
        assert len(response.json()["notes"]) == NOTES_LIMIT


# ---------------------------------------------------------------------------
# Sub-AC 3 (governed dispatch) -- GET /api/memos
# ---------------------------------------------------------------------------


class TestMemosEndpoint:
    def test_returns_200_with_stored_notes(self, client, note_store):
        note_store.add_note(CANONICAL_USER_ID, "Buy milk")
        note_store.add_note(CANONICAL_USER_ID, "Call dentist")

        response = client.get("/api/memos")

        assert response.status_code == 200

        body = response.json()

        # Response shape matches the `notes` ontology concept exactly.
        assert set(body.keys()) == {"notes"}
        assert len(body["notes"]) == 2

        # Most recently created first (NoteStore.list_notes ordering).
        assert body["notes"][0]["content"] == "Call dentist"
        assert body["notes"][1]["content"] == "Buy milk"

        for note in body["notes"]:
            assert set(note.keys()) == {"note_id", "content", "created_at", "updated_at"}
            assert isinstance(note["note_id"], int)
            assert isinstance(note["content"], str)
            assert isinstance(note["created_at"], str)
            assert isinstance(note["updated_at"], str)

    def test_returns_200_with_empty_list_when_no_notes(self, client):
        response = client.get("/api/memos")

        assert response.status_code == 200
        assert response.json() == {"notes": []}

    def test_thin_pass_through_to_note_store_list_notes(self, client, note_store):
        note_store.add_note(CANONICAL_USER_ID, "Buy milk")

        with patch.object(
            note_store, "list_notes", wraps=note_store.list_notes
        ) as mock_list_notes:
            response = client.get("/api/memos")

        assert response.status_code == 200
        # Thin pass-through: NoteStore.list_notes called directly for the
        # canonical (single-user) user_id, no LLM round-trip.
        mock_list_notes.assert_called_once_with(CANONICAL_USER_ID, limit=NOTES_LIMIT)

    def test_returns_at_most_notes_limit(self, client, note_store):
        for i in range(NOTES_LIMIT + 5):
            note_store.add_note(CANONICAL_USER_ID, f"Note {i}")

        response = client.get("/api/memos")

        assert response.status_code == 200
        assert len(response.json()["notes"]) == NOTES_LIMIT


# ---------------------------------------------------------------------------
# Sub-AC 4 -- GET /api/history
# ---------------------------------------------------------------------------


class TestHistoryEndpoint:
    def test_returns_200_with_empty_history_when_no_session(self, client):
        response = client.get("/api/history")

        assert response.status_code == 200
        assert response.json() == {"history": []}

    def test_returns_200_with_expected_schema(self, client, store):
        store.upsert_session(
            CANONICAL_USER_ID,
            {
                "working_memory": [
                    {"role": "user", "content": "Hello", "source_interface": "web_ui"},
                    {"role": "assistant", "content": "Hi there!", "source_interface": "web_ui"},
                ],
                "source_interface": "web_ui",
            },
        )

        response = client.get("/api/history")

        assert response.status_code == 200

        body = response.json()

        # Response shape matches the chat_history ontology concept exactly.
        assert set(body.keys()) == {"history"}
        assert body == {
            "history": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }

        for message in body["history"]:
            assert set(message.keys()) == {"role", "content"}
            assert isinstance(message["role"], str)
            assert isinstance(message["content"], str)

    def test_history_reflects_chat_endpoint_persistence(self, client):
        client.post("/chat", json={"message": "First"})
        client.post("/chat", json={"message": "Second"})

        response = client.get("/api/history")

        assert response.status_code == 200
        body = response.json()
        assert body == {
            "history": [
                {"role": "user", "content": "First"},
                {"role": "assistant", "content": "stub reply"},
                {"role": "user", "content": "Second"},
                {"role": "assistant", "content": "stub reply"},
            ]
        }

    def test_filters_out_non_user_assistant_messages(self, client, store):
        store.upsert_session(
            CANONICAL_USER_ID,
            {
                "working_memory": [
                    {"role": "user", "content": "What's the weather?", "source_interface": "web_ui"},
                    {"role": "tool", "name": "get_weather", "content": "{\"temp\": 22}"},
                    {"role": "assistant", "content": "It's 22°C.", "source_interface": "web_ui"},
                ],
                "source_interface": "web_ui",
            },
        )

        response = client.get("/api/history")

        assert response.status_code == 200
        assert response.json() == {
            "history": [
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "content": "It's 22°C."},
            ]
        }

    def test_thin_pass_through_to_session_store_get_session(self, client, store):
        store.upsert_session(
            CANONICAL_USER_ID,
            {
                "working_memory": [{"role": "user", "content": "Hi", "source_interface": "web_ui"}],
                "source_interface": "web_ui",
            },
        )

        with patch.object(store, "get_session", wraps=store.get_session) as mock_get_session:
            response = client.get("/api/history")

        assert response.status_code == 200
        # Thin pass-through: SessionStore.get_session called directly for the
        # canonical (single-user) user_id, no LLM round-trip.
        mock_get_session.assert_called_once_with(CANONICAL_USER_ID)


# ---------------------------------------------------------------------------
# Sub-AC 4 (governed dispatch) -- GET /api/session
# ---------------------------------------------------------------------------


class TestSessionEndpoint:
    def test_returns_200_with_empty_dict_when_no_session(self, client):
        response = client.get("/api/session")

        assert response.status_code == 200
        assert response.json() == {}

    def test_returns_seeded_session_data(self, client, store):
        seeded_context = {
            "working_memory": [
                {"role": "user", "content": "Hello", "source_interface": "web_ui"},
                {"role": "assistant", "content": "Hi there!", "source_interface": "web_ui"},
            ],
            "source_interface": "web_ui",
            "session_memory": {"topic": "greetings"},
        }
        store.upsert_session(CANONICAL_USER_ID, seeded_context)

        # The expected value is exactly what SessionStore.get_session returns
        # for the seeded session (including the injected `_meta` block).
        expected = store.get_session(CANONICAL_USER_ID)

        response = client.get("/api/session")

        assert response.status_code == 200
        body = response.json()
        assert body == expected
        assert body["working_memory"] == seeded_context["working_memory"]
        assert body["source_interface"] == "web_ui"
        assert body["session_memory"] == {"topic": "greetings"}
        assert "_meta" in body and "session_id" in body["_meta"]

    def test_thin_pass_through_to_session_store_get_session(self, client, store):
        store.upsert_session(
            CANONICAL_USER_ID,
            {
                "working_memory": [{"role": "user", "content": "Hi", "source_interface": "web_ui"}],
                "source_interface": "web_ui",
            },
        )

        with patch.object(store, "get_session", wraps=store.get_session) as mock_get_session:
            response = client.get("/api/session")

        assert response.status_code == 200
        # Thin pass-through: SessionStore.get_session called directly for the
        # canonical (single-user) user_id, no LLM round-trip.
        mock_get_session.assert_called_once_with(CANONICAL_USER_ID)


# ---------------------------------------------------------------------------
# Sub-AC 4 (governed dispatch) -- GET /api/widgets/session
# ---------------------------------------------------------------------------


class TestSessionWidgetEndpoint:
    def test_returns_200_with_empty_dict_when_no_session(self, client):
        response = client.get("/api/widgets/session")

        assert response.status_code == 200
        assert response.json() == {}

    def test_returns_seeded_session_data(self, client, store):
        seeded_context = {
            "working_memory": [
                {"role": "user", "content": "Hello", "source_interface": "web_ui"},
                {"role": "assistant", "content": "Hi there!", "source_interface": "web_ui"},
            ],
            "source_interface": "web_ui",
            "session_memory": {"topic": "greetings"},
        }
        store.upsert_session(CANONICAL_USER_ID, seeded_context)

        # The expected value is exactly what SessionStore.get_session returns
        # for the seeded session (including the injected `_meta` block).
        expected = store.get_session(CANONICAL_USER_ID)

        response = client.get("/api/widgets/session")

        assert response.status_code == 200
        body = response.json()
        assert body == expected
        assert body["working_memory"] == seeded_context["working_memory"]
        assert body["source_interface"] == "web_ui"
        assert body["session_memory"] == {"topic": "greetings"}
        assert "_meta" in body and "session_id" in body["_meta"]

    def test_thin_pass_through_to_session_store_get_session(self, client, store):
        store.upsert_session(
            CANONICAL_USER_ID,
            {
                "working_memory": [{"role": "user", "content": "Hi", "source_interface": "web_ui"}],
                "source_interface": "web_ui",
            },
        )

        with patch.object(store, "get_session", wraps=store.get_session) as mock_get_session:
            response = client.get("/api/widgets/session")

        assert response.status_code == 200
        # Thin pass-through: SessionStore.get_session called directly for the
        # canonical (single-user) user_id, no LLM round-trip.
        mock_get_session.assert_called_once_with(CANONICAL_USER_ID)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
