"""Web UI interface adapter (AC6.2).

Exposes a minimal FastAPI app with a single ``POST /chat`` endpoint that
delegates to :func:`assistant.agent_core.handle_user_message` with
``source_interface="web_ui"``. Conversation turns are persisted to (and
loaded from) the same :class:`~assistant.session_store.SessionStore` used
by the Telegram and Discord adapters, giving cross-interface context
continuity (AC6.5).

It also exposes a small set of read-only ``GET /api/*`` endpoints that back
the dashboard sidebar widgets (weather, calendar, notes, history). These are
thin pass-throughs to :func:`assistant.tools.dispatch.dispatch_tool` and the
existing tool implementations -- no LLM round-trip is performed for them.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from assistant.agent_core import Responder, default_responder, handle_user_message
from assistant.llm import OllamaError
from assistant.notes import NoteStore
from assistant.session_resolver import CANONICAL_USER_ID
from assistant.session_store import SessionStore
from assistant.tools.dispatch import dispatch_tool

app = FastAPI(title="Personal Assistant Agent — Web UI")

# Directory containing the single-file dashboard frontend
# (assistant/static/index.html), served at the application root (Sub-AC 1).
STATIC_DIR: Path = Path(__file__).resolve().parent.parent / "static"

# Default location used by GET /api/weather (Sub-AC 1). Configurable via the
# WEATHER_DEFAULT_CITY environment variable, falling back to "Seoul" per the
# ontology's `default_city` concept.
DEFAULT_CITY: str = os.environ.get("WEATHER_DEFAULT_CITY", "Seoul")

# Maximum number of notes returned by GET /api/notes (Sub-AC 3), most
# recently created first -- backs the ontology's `notes` ("recent notes")
# concept for the memo sidebar widget.
NOTES_LIMIT: int = 20


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


class WeatherResponse(BaseModel):
    """Response shape for ``GET /api/weather`` (ontology: ``weather_data``).

    Mirrors :meth:`assistant.tools.weather.WeatherReport.to_dict` exactly --
    factual fields are passed through unmodified.
    """

    location_name: str
    latitude: float
    longitude: float
    temperature_c: float
    wind_speed_kmh: float
    weather_code: int
    weather_description: str
    observation_time: str


class CalendarEventResponse(BaseModel):
    """Response shape for a single calendar event in ``GET /api/calendar/today``.

    Mirrors :meth:`assistant.tools.calendar.CalendarEvent.to_dict` exactly --
    factual fields are passed through unmodified.
    """

    calendar_event_id: str
    calendar_event_title: str
    calendar_event_time: str
    calendar_alert_time: str
    end_time: str
    description: str = ""
    location: str = ""
    attendees: list[str] = []
    is_all_day: bool = False
    html_link: str = ""


class CalendarTodayResponse(BaseModel):
    """Response shape for ``GET /api/calendar/today`` (ontology: ``calendar_events``)."""

    events: list[CalendarEventResponse]


class NoteResponse(BaseModel):
    """Response shape for a single note in ``GET /api/notes``.

    Mirrors the dict returned by
    :meth:`assistant.notes.NoteStore.list_notes` exactly -- factual fields
    are passed through unmodified.
    """

    note_id: int
    content: str
    created_at: str
    updated_at: str


class NotesResponse(BaseModel):
    """Response shape for ``GET /api/notes`` (ontology: ``notes``)."""

    notes: list[NoteResponse]


class HistoryMessageResponse(BaseModel):
    """Response shape for a single message in ``GET /api/history``.

    Mirrors the ``role``/``content`` keys of a ``working_memory`` entry
    (ontology: ``chat_message``) -- bookkeeping keys such as
    ``source_interface`` are dropped.
    """

    role: str
    content: str


class HistoryResponse(BaseModel):
    """Response shape for ``GET /api/history`` (ontology: ``chat_history``)."""

    history: list[HistoryMessageResponse]


def get_store() -> SessionStore:
    """FastAPI dependency providing the session store. Overridable in tests."""
    return SessionStore()


def get_responder() -> Responder:
    """FastAPI dependency providing the response generator. Overridable in tests."""
    return default_responder


def get_note_store() -> NoteStore:
    """FastAPI dependency providing the note store. Overridable in tests."""
    return NoteStore()


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the single-file dashboard frontend (assistant/static/index.html).

    Sub-AC 1: the FastAPI backend serves the dashboard UI as a static file at
    the application root (``http://localhost:8000/``) -- no templating or
    server-side rendering, just the self-contained HTML+CSS+JS document
    described in the seed contract.

    Returns:
        The contents of ``assistant/static/index.html`` with a ``text/html``
        media type.
    """
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@app.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    store: SessionStore = Depends(get_store),
    responder: Responder = Depends(get_responder),
) -> ChatResponse:
    """Handle one conversation turn from the Web UI.

    Args:
        request: The incoming chat request, containing the user's message.
        store: Session store dependency (injected).
        responder: Reply-generation dependency (injected).

    Returns:
        The assistant's reply.

    Raises:
        HTTPException: 400 if *request.message* is empty/whitespace, or 503
            if the local LLM backend is unreachable.
    """
    try:
        reply = handle_user_message("web_ui", request.message, store=store, responder=responder)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ChatResponse(reply=reply)


@app.get("/api/weather", response_model=WeatherResponse)
def get_weather() -> WeatherResponse:
    """Return current weather conditions for the configured default city.

    Sub-AC 1: a thin pass-through to
    :func:`assistant.tools.dispatch.dispatch_tool` -- which calls
    :func:`assistant.tools.weather.get_current_weather` directly -- with no
    LLM round-trip. The location is fixed to :data:`DEFAULT_CITY`
    (``WEATHER_DEFAULT_CITY`` env var, falling back to ``"Seoul"``).

    Returns:
        Current weather conditions for :data:`DEFAULT_CITY`, shaped per
        :class:`WeatherResponse` (the ontology's ``weather_data`` concept).

    Raises:
        HTTPException: 502 if the underlying weather tool fails (e.g. the
            Open-Meteo API is unreachable or the city cannot be resolved).
    """
    result = dispatch_tool("get_weather", {"location": DEFAULT_CITY})

    if result["tool_status"] != "success":
        raise HTTPException(
            status_code=502,
            detail=result["tool_error_message"] or "Failed to retrieve weather data",
        )

    return WeatherResponse(**result["tool_output"]["weather"])


@app.get("/api/calendar/today", response_model=CalendarTodayResponse)
def get_calendar_today() -> CalendarTodayResponse:
    """Return today's Google Calendar events.

    Sub-AC 2: a thin pass-through to
    :func:`assistant.tools.dispatch.dispatch_tool` -- which calls
    :func:`assistant.tools.calendar.get_calendar_events` directly -- with no
    LLM round-trip. The date range is fixed to today's date (server-local),
    using bare ``YYYY-MM-DD`` start/end dates so
    :func:`~assistant.tools.calendar._to_rfc3339_start` /
    :func:`~assistant.tools.calendar._to_rfc3339_end` expand it to cover the
    full day.

    Returns:
        Today's calendar events, shaped per :class:`CalendarTodayResponse`
        (the ontology's ``calendar_events`` concept). An empty ``events``
        list is returned (200 OK) when there are no events today.

    Raises:
        HTTPException: 502 if the underlying calendar tool fails (e.g. the
            Google Calendar API is unreachable or credentials are invalid).
    """
    today = date.today().isoformat()
    result = dispatch_tool("get_calendar_events", {"start_date": today, "end_date": today})

    if result["tool_status"] != "success":
        raise HTTPException(
            status_code=502,
            detail=result["tool_error_message"] or "Failed to retrieve calendar events",
        )

    return CalendarTodayResponse(**result["tool_output"])


@app.get("/api/calendar", response_model=CalendarTodayResponse)
def get_calendar() -> CalendarTodayResponse:
    """Return today's Google Calendar events (calendar sidebar widget).

    Sub-AC 2 (governed dispatch): a thin pass-through to
    :func:`assistant.tools.dispatch.dispatch_tool` -- which calls
    :func:`assistant.tools.calendar.get_calendar_events` directly -- with no
    LLM round-trip. Provided under the ``/api/calendar`` path to match the
    dashboard sidebar's calendar widget naming convention (mirrors how
    ``GET /api/memos`` aliases ``GET /api/notes``). Delegates directly to
    :func:`get_calendar_today`, which fixes the date range to today's date.

    Returns:
        Today's calendar events, shaped per :class:`CalendarTodayResponse`
        (the ontology's ``calendar_events`` concept). An empty ``events``
        list is returned (200 OK) when there are no events today.

    Raises:
        HTTPException: 502 if the underlying calendar tool fails (e.g. the
            Google Calendar API is unreachable or credentials are invalid).
    """
    return get_calendar_today()


@app.get("/api/notes", response_model=NotesResponse)
def get_notes(note_store: NoteStore = Depends(get_note_store)) -> NotesResponse:
    """Return the most recently created notes for the default user.

    Sub-AC 3: a thin pass-through to
    :meth:`assistant.notes.NoteStore.list_notes` -- the same store backing
    the ``/note`` slash command -- with no LLM round-trip. Notes are scoped
    to :data:`assistant.session_resolver.CANONICAL_USER_ID` (``"default"``),
    matching the single-user session model used by ``POST /chat``.

    Returns:
        Up to :data:`NOTES_LIMIT` most recent notes, most recently created
        first, shaped per :class:`NotesResponse` (the ontology's ``notes``
        concept). An empty ``notes`` list is returned (200 OK) when the user
        has no notes yet.
    """
    notes = note_store.list_notes(CANONICAL_USER_ID, limit=NOTES_LIMIT)
    return NotesResponse(notes=[NoteResponse(**note) for note in notes])


@app.get("/api/memos", response_model=NotesResponse)
def get_memos(note_store: NoteStore = Depends(get_note_store)) -> NotesResponse:
    """Return the most recently created notes for the default user (memo widget).

    Sub-AC 3 (governed dispatch): a thin pass-through to
    :meth:`assistant.notes.NoteStore.list_notes` -- the same store backing
    the ``/note`` slash command and ``GET /api/notes`` -- with no LLM
    round-trip. Provided under the ``/api/memos`` path to match the
    dashboard sidebar's memo widget naming.

    Returns:
        Up to :data:`NOTES_LIMIT` most recent notes, most recently created
        first, shaped per :class:`NotesResponse` (the ontology's ``notes``
        concept). An empty ``notes`` list is returned (200 OK) when the user
        has no notes yet.
    """
    return get_notes(note_store)


@app.get("/api/history", response_model=HistoryResponse)
def get_history(store: SessionStore = Depends(get_store)) -> HistoryResponse:
    """Return the shared conversation history for the chat panel.

    Sub-AC 4: a thin pass-through to
    :meth:`assistant.session_store.SessionStore.get_session` -- reading the
    same ``working_memory`` shared by ``POST /chat`` and the Telegram/Discord
    interfaces (AC6.5) -- with no LLM round-trip.

    Only ``user`` and ``assistant`` turns are returned, in chronological
    order, so the dashboard's chat panel can be rehydrated on page load
    without surfacing internal tool-call bookkeeping messages.

    Args:
        store: Session store dependency (injected).

    Returns:
        The conversation history shaped per :class:`HistoryResponse` (the
        ontology's ``chat_history`` concept). An empty ``history`` list is
        returned (200 OK) when no session/conversation exists yet.
    """
    ctx = store.get_session(CANONICAL_USER_ID)
    working_memory = (ctx or {}).get("working_memory", [])

    history = [
        HistoryMessageResponse(role=message["role"], content=message.get("content") or "")
        for message in working_memory
        if message.get("role") in ("user", "assistant")
    ]
    return HistoryResponse(history=history)


@app.get("/api/session")
def get_session(store: SessionStore = Depends(get_store)) -> dict[str, Any]:
    """Return the raw session context for the canonical (single) user.

    Sub-AC 4 (governed dispatch): a thin pass-through to
    :meth:`assistant.session_store.SessionStore.get_session` -- the same
    ``working_memory``/``session_memory``/``_meta`` context shared by
    ``POST /chat`` and the Telegram/Discord interfaces (AC6.5) -- with no
    LLM round-trip and no field-level transformation.

    Unlike ``GET /api/history`` (which projects ``working_memory`` down to
    ``role``/``content`` pairs for the chat panel), this endpoint returns the
    full session document untouched, suitable for debugging/inspection
    widgets that need the raw session state.

    Args:
        store: Session store dependency (injected).

    Returns:
        The full session context dict as returned by
        :meth:`SessionStore.get_session`, or ``{}`` (200 OK) if no session
        exists yet for the canonical user.
    """
    ctx = store.get_session(CANONICAL_USER_ID)
    return ctx if ctx is not None else {}


@app.get("/api/widgets/session")
def get_session_widget(store: SessionStore = Depends(get_store)) -> dict[str, Any]:
    """Return the raw session context for the canonical (single) user.

    Alias of ``GET /api/session`` retained for the dashboard widget naming
    convention. Delegates directly to :func:`get_session` -- still a thin
    pass-through to :meth:`assistant.session_store.SessionStore.get_session`
    with no LLM round-trip and no field-level transformation.

    Args:
        store: Session store dependency (injected).

    Returns:
        The full session context dict as returned by
        :meth:`SessionStore.get_session`, or ``{}`` (200 OK) if no session
        exists yet for the canonical user.
    """
    return get_session(store)


# Mount the static assets directory last, after all API routes above, so any
# future additions under assistant/static/ (e.g. /static/index.html) are
# servable without shadowing /chat, /api/*, or the root "/" route.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
