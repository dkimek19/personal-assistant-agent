"""Google Calendar tool for the Personal Assistant Agent.

Provides four public functions:
- get_calendar_events: Retrieves events from Google Calendar for a given date range.
- create_calendar_event: Creates a new event in Google Calendar and returns its ID.
- update_calendar_event: Updates fields of an existing event by event ID.
- delete_calendar_event: Deletes an existing event by event ID.

Tool-returned factual data (times, dates, titles) is never altered by the LLM —
formatting only, no content modification.

Constraint: exponential backoff with 2 retries on API failures, then explicit
error message to user.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CalendarEvent:
    """Structured representation of a single Google Calendar event.

    All factual fields (event_id, title, start_time, end_time) are stored
    exactly as returned by the API — no LLM modification allowed.
    """

    # Required ontology concepts
    calendar_event_id: str
    calendar_event_title: str
    calendar_event_time: str          # ISO 8601 start datetime as returned by API
    calendar_alert_time: str          # ISO 8601 datetime 30 min before event
    end_time: str                     # ISO 8601 end datetime as returned by API
    description: str = ""
    location: str = ""
    attendees: list[str] = field(default_factory=list)
    is_all_day: bool = False
    html_link: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict representation (safe to pass to LLM context)."""
        return {
            "calendar_event_id": self.calendar_event_id,
            "calendar_event_title": self.calendar_event_title,
            "calendar_event_time": self.calendar_event_time,
            "calendar_alert_time": self.calendar_alert_time,
            "end_time": self.end_time,
            "description": self.description,
            "location": self.location,
            "attendees": self.attendees,
            "is_all_day": self.is_all_day,
            "html_link": self.html_link,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MINUTES_BEFORE_ALERT = 30


def _compute_alert_time(event_start_iso: str) -> str:
    """Return ISO 8601 string 30 minutes before *event_start_iso*.

    Handles both datetime strings (with time) and date-only strings (all-day
    events).  For all-day events the alert is set to 09:00 local time on the
    event day minus 30 minutes = 08:30 on the same day.
    """
    # Try full datetime first
    for fmt in (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            dt = datetime.strptime(event_start_iso, fmt)
            # Normalise to UTC-aware if naive
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            from datetime import timedelta
            alert_dt = dt - timedelta(minutes=_MINUTES_BEFORE_ALERT)
            return alert_dt.isoformat()
        except ValueError:
            continue

    # Fall back: date-only all-day event  →  09:00 on the same day
    try:
        from datetime import timedelta
        d = datetime.strptime(event_start_iso, "%Y-%m-%d")
        # Default 09:00 UTC as alert anchor for all-day events
        anchor = d.replace(hour=9, minute=0, second=0, tzinfo=timezone.utc)
        alert_dt = anchor - timedelta(minutes=_MINUTES_BEFORE_ALERT)
        return alert_dt.isoformat()
    except ValueError:
        pass

    # Ultimate fallback — return the start time unchanged
    return event_start_iso


def _parse_event(raw: dict[str, Any]) -> CalendarEvent:
    """Convert a raw Google Calendar API event dict to a CalendarEvent.

    Factual data is passed through without modification.
    """
    event_id: str = raw.get("id", "")
    title: str = raw.get("summary", "(No title)")

    # Determine start/end — may be date (all-day) or dateTime
    start_obj: dict = raw.get("start", {})
    end_obj: dict = raw.get("end", {})

    is_all_day = "date" in start_obj and "dateTime" not in start_obj
    start_str: str = start_obj.get("dateTime") or start_obj.get("date", "")
    end_str: str = end_obj.get("dateTime") or end_obj.get("date", "")

    alert_time = _compute_alert_time(start_str)

    attendees: list[str] = [
        a.get("email", "") for a in raw.get("attendees", []) if a.get("email")
    ]

    return CalendarEvent(
        calendar_event_id=event_id,
        calendar_event_title=title,
        calendar_event_time=start_str,
        calendar_alert_time=alert_time,
        end_time=end_str,
        description=raw.get("description", ""),
        location=raw.get("location", ""),
        attendees=attendees,
        is_all_day=is_all_day,
        html_link=raw.get("htmlLink", ""),
    )


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def get_calendar_events(
    start_date: str,
    end_date: str,
    *,
    service: Any | None = None,
    calendar_id: str = "primary",
    max_results: int = 100,
    max_retries: int = 2,
    base_backoff: float = 1.0,
) -> list[CalendarEvent]:
    """Retrieve Google Calendar events for *start_date* to *end_date*.

    Parameters
    ----------
    start_date:
        ISO 8601 date or datetime string marking the start of the range
        (inclusive).  If a bare date string (``YYYY-MM-DD``) is given it is
        converted to ``YYYY-MM-DDT00:00:00Z``.
    end_date:
        ISO 8601 date or datetime string marking the end of the range
        (exclusive).  If a bare date string is given it is converted to
        ``YYYY-MM-DDT23:59:59Z``.
    service:
        An already-constructed Google Calendar API resource object
        (``googleapiclient.discovery.Resource``).  When *None* the function
        builds one automatically from ``GOOGLE_CALENDAR_CREDENTIALS_FILE``
        and ``GOOGLE_CALENDAR_TOKEN_FILE`` environment variables (or their
        defaults in ``~/assistant/credentials/``).
    calendar_id:
        The Calendar ID to query.  Defaults to ``"primary"``.
    max_results:
        Maximum number of events to retrieve in a single page.
    max_retries:
        Number of retry attempts on transient failures (default 2).
    base_backoff:
        Base delay in seconds for exponential backoff.

    Returns
    -------
    list[CalendarEvent]
        Ordered list of :class:`CalendarEvent` objects, sorted ascending by
        start time.  Returns an empty list when no events are found.

    Raises
    ------
    RuntimeError
        After *max_retries* failures, a ``RuntimeError`` is raised with an
        explicit human-readable message for the user.  The caller is
        responsible for forwarding this message to the user interface.
    """

    # Normalise date-only strings to RFC 3339 datetimes expected by the API
    time_min = _to_rfc3339_start(start_date)
    time_max = _to_rfc3339_end(end_date)

    # Build the API service lazily if not injected (supports unit-testing via
    # dependency injection)
    if service is None:
        service = _build_calendar_service()

    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        if attempt > 0:
            delay = base_backoff * (2 ** (attempt - 1))
            logger.warning(
                "Google Calendar API retry %d/%d after %.1fs backoff (error: %s)",
                attempt,
                max_retries,
                delay,
                last_error,
            )
            time.sleep(delay)

        try:
            events = _fetch_all_events(
                service=service,
                calendar_id=calendar_id,
                time_min=time_min,
                time_max=time_max,
                max_results=max_results,
            )
            logger.info(
                "Retrieved %d calendar events from %s to %s",
                len(events),
                start_date,
                end_date,
            )
            return events

        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.error(
                "Google Calendar API error on attempt %d: %s",
                attempt + 1,
                exc,
            )

    # All retries exhausted — surface an explicit, honest error to the user
    raise RuntimeError(
        f"Unable to retrieve calendar events after {max_retries + 1} attempts. "
        f"Last error: {last_error}. "
        "Please check your Google Calendar credentials and network connection."
    )


def _event_start_date(event: CalendarEvent) -> str:
    """Return the ``YYYY-MM-DD`` date portion of *event*'s start time.

    Works for both timed events (``calendar_event_time`` is a full ISO 8601
    datetime, e.g. ``"2026-06-15T09:00:00+09:00"``) and all-day events
    (``calendar_event_time`` is a bare date, e.g. ``"2026-06-15"``) — in both
    cases the first 10 characters are the ``YYYY-MM-DD`` date.
    """
    return event.calendar_event_time[:10]


def get_today_events(
    *,
    service: Any | None = None,
    calendar_id: str = "primary",
    max_results: int = 100,
    max_retries: int = 2,
    base_backoff: float = 1.0,
    today: str | None = None,
) -> list[CalendarEvent]:
    """Retrieve and return today's Google Calendar events.

    This is the calendar service function backing the dashboard's calendar
    sidebar widget (``GET /api/calendar/today``). It is a thin convenience
    wrapper around :func:`get_calendar_events`: it computes "today" (the
    current server-local date, unless overridden via *today* for testing),
    queries the Google Calendar API for that single-day range, and then
    additionally filters the returned events so that only events whose
    *start date* falls on that day are kept — guarding against any events
    returned at the edges of the requested range (e.g. due to timezone
    rounding in the underlying API response).

    Parameters
    ----------
    service:
        An already-constructed Google Calendar API resource object. When
        *None* (the default), :func:`get_calendar_events` builds one
        automatically. Pass a mock here for testing.
    calendar_id:
        The Calendar ID to query. Defaults to ``"primary"``.
    max_results:
        Maximum number of events to retrieve.
    max_retries:
        Number of retry attempts on transient failures (default 2).
    base_backoff:
        Base delay in seconds for exponential backoff.
    today:
        ISO 8601 date string (``YYYY-MM-DD``) to treat as "today". Primarily
        useful for testing; defaults to ``datetime.now().date().isoformat()``.

    Returns
    -------
    list[CalendarEvent]
        Events starting today, sorted ascending by start time, each
        containing the ontology's event fields (title, start/end time, id,
        description, location, attendees, etc. — see
        :meth:`CalendarEvent.to_dict`). Returns an empty list when there are
        no events today.

    Raises
    ------
    RuntimeError
        Propagated from :func:`get_calendar_events` if the Google Calendar
        API exhausts its retries.
    """
    today_str = today if today is not None else datetime.now().date().isoformat()

    events = get_calendar_events(
        today_str,
        today_str,
        service=service,
        calendar_id=calendar_id,
        max_results=max_results,
        max_retries=max_retries,
        base_backoff=base_backoff,
    )

    return [event for event in events if _event_start_date(event) == today_str]


def create_calendar_event(
    title: str,
    start_time: str,
    end_time: str,
    description: str = "",
    *,
    service: Any | None = None,
    calendar_id: str = "primary",
    max_retries: int = 2,
    base_backoff: float = 1.0,
) -> str:
    """Create a new event in Google Calendar and return the created event ID.

    Parameters
    ----------
    title:
        Event title (maps to the ``summary`` field in the Google Calendar API).
    start_time:
        ISO 8601 datetime string for the event start
        (e.g. ``"2026-06-10T09:00:00Z"``).
    end_time:
        ISO 8601 datetime string for the event end
        (e.g. ``"2026-06-10T10:00:00Z"``).
    description:
        Optional text description of the event.
    service:
        An already-constructed Google Calendar API resource object
        (``googleapiclient.discovery.Resource``).  When *None* the function
        builds one automatically using ``_build_calendar_service()``.
    calendar_id:
        The Calendar ID to create the event in.  Defaults to ``"primary"``.
    max_retries:
        Number of retry attempts on transient failures (default 2).
    base_backoff:
        Base delay in seconds for exponential backoff.

    Returns
    -------
    str
        The ``id`` of the newly created event, exactly as returned by the
        Google Calendar API — no modification.

    Raises
    ------
    RuntimeError
        After *max_retries* failures, a ``RuntimeError`` is raised with an
        explicit human-readable message for the user.  The caller is
        responsible for forwarding this message to the user interface.
    """
    if service is None:
        service = _build_calendar_service()

    # Construct the event resource payload per the Google Calendar API spec.
    # Factual data (title, times, description) is passed through without
    # alteration — the LLM must not modify these fields.
    event_body: dict[str, Any] = {
        "summary": title,
        "description": description,
        "start": {"dateTime": start_time},
        "end": {"dateTime": end_time},
    }

    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        if attempt > 0:
            delay = base_backoff * (2 ** (attempt - 1))
            logger.warning(
                "Google Calendar API create retry %d/%d after %.1fs backoff (error: %s)",
                attempt,
                max_retries,
                delay,
                last_error,
            )
            time.sleep(delay)

        try:
            response: dict[str, Any] = (
                service.events()
                .insert(calendarId=calendar_id, body=event_body)
                .execute()
            )
            event_id: str = response.get("id", "")
            logger.info(
                "Created calendar event '%s' with ID '%s'",
                title,
                event_id,
            )
            return event_id

        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.error(
                "Google Calendar API create error on attempt %d: %s",
                attempt + 1,
                exc,
            )

    # All retries exhausted — surface an explicit, honest error to the user
    raise RuntimeError(
        f"Unable to create calendar event after {max_retries + 1} attempts. "
        f"Last error: {last_error}. "
        "Please check your Google Calendar credentials and network connection."
    )


def update_calendar_event(
    event_id: str,
    *,
    title: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    description: str | None = None,
    service: Any | None = None,
    calendar_id: str = "primary",
    max_retries: int = 2,
    base_backoff: float = 1.0,
) -> CalendarEvent:
    """Update fields of an existing Google Calendar event by event ID.

    Only the fields explicitly provided are changed (a partial update via the
    Calendar API's ``patch`` method) — omitted fields retain their existing
    values on the event.

    Parameters
    ----------
    event_id:
        The ``id`` of the event to update, as returned by
        :func:`get_calendar_events` or :func:`create_calendar_event`.
    title:
        New event title (maps to ``summary``).  Unchanged if *None*.
    start_time:
        New ISO 8601 datetime string for the event start.  Unchanged if *None*.
    end_time:
        New ISO 8601 datetime string for the event end.  Unchanged if *None*.
    description:
        New text description.  Unchanged if *None*.
    service:
        An already-constructed Google Calendar API resource object.  When
        *None* the function builds one automatically using
        :func:`_build_calendar_service`.
    calendar_id:
        The Calendar ID containing the event.  Defaults to ``"primary"``.
    max_retries:
        Number of retry attempts on transient failures (default 2).
    base_backoff:
        Base delay in seconds for exponential backoff.

    Returns
    -------
    CalendarEvent
        The updated event, parsed from the Google Calendar API response —
        factual fields are passed through without modification.

    Raises
    ------
    ValueError
        If none of *title*, *start_time*, *end_time*, or *description* is
        provided (nothing to update).
    RuntimeError
        After *max_retries* failures, a ``RuntimeError`` is raised with an
        explicit human-readable message for the user.
    """
    if title is None and start_time is None and end_time is None and description is None:
        raise ValueError(
            "update_calendar_event requires at least one of: "
            "title, start_time, end_time, description"
        )

    if service is None:
        service = _build_calendar_service()

    # Construct a partial update payload — only changed fields are included.
    body: dict[str, Any] = {}
    if title is not None:
        body["summary"] = title
    if description is not None:
        body["description"] = description
    if start_time is not None:
        body["start"] = {"dateTime": start_time}
    if end_time is not None:
        body["end"] = {"dateTime": end_time}

    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        if attempt > 0:
            delay = base_backoff * (2 ** (attempt - 1))
            logger.warning(
                "Google Calendar API update retry %d/%d after %.1fs backoff (error: %s)",
                attempt,
                max_retries,
                delay,
                last_error,
            )
            time.sleep(delay)

        try:
            response: dict[str, Any] = (
                service.events()
                .patch(calendarId=calendar_id, eventId=event_id, body=body)
                .execute()
            )
            logger.info("Updated calendar event '%s'", event_id)
            return _parse_event(response)

        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.error(
                "Google Calendar API update error on attempt %d: %s",
                attempt + 1,
                exc,
            )

    # All retries exhausted — surface an explicit, honest error to the user
    raise RuntimeError(
        f"Unable to update calendar event '{event_id}' after {max_retries + 1} attempts. "
        f"Last error: {last_error}. "
        "Please check your Google Calendar credentials and network connection."
    )


def delete_calendar_event(
    event_id: str,
    *,
    service: Any | None = None,
    calendar_id: str = "primary",
    max_retries: int = 2,
    base_backoff: float = 1.0,
) -> bool:
    """Delete an existing Google Calendar event by event ID.

    Parameters
    ----------
    event_id:
        The ``id`` of the event to delete, as returned by
        :func:`get_calendar_events` or :func:`create_calendar_event`.
    service:
        An already-constructed Google Calendar API resource object.  When
        *None* the function builds one automatically using
        :func:`_build_calendar_service`.
    calendar_id:
        The Calendar ID containing the event.  Defaults to ``"primary"``.
    max_retries:
        Number of retry attempts on transient failures (default 2).
    base_backoff:
        Base delay in seconds for exponential backoff.

    Returns
    -------
    bool
        ``True`` once the event has been successfully deleted.

    Raises
    ------
    RuntimeError
        After *max_retries* failures, a ``RuntimeError`` is raised with an
        explicit human-readable message for the user.
    """
    if service is None:
        service = _build_calendar_service()

    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        if attempt > 0:
            delay = base_backoff * (2 ** (attempt - 1))
            logger.warning(
                "Google Calendar API delete retry %d/%d after %.1fs backoff (error: %s)",
                attempt,
                max_retries,
                delay,
                last_error,
            )
            time.sleep(delay)

        try:
            service.events().delete(
                calendarId=calendar_id, eventId=event_id
            ).execute()
            logger.info("Deleted calendar event '%s'", event_id)
            return True

        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.error(
                "Google Calendar API delete error on attempt %d: %s",
                attempt + 1,
                exc,
            )

    # All retries exhausted — surface an explicit, honest error to the user
    raise RuntimeError(
        f"Unable to delete calendar event '{event_id}' after {max_retries + 1} attempts. "
        f"Last error: {last_error}. "
        "Please check your Google Calendar credentials and network connection."
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_rfc3339_start(date_str: str) -> str:
    """Convert a bare date or datetime string to RFC 3339 start-of-day."""
    if "T" in date_str or " " in date_str:
        # Already a datetime; ensure Z suffix for UTC if no offset present
        if not (date_str.endswith("Z") or "+" in date_str[10:] or "-" in date_str[10:]):
            return date_str + "Z"
        return date_str
    # bare date YYYY-MM-DD → start of day UTC
    return f"{date_str}T00:00:00Z"


def _to_rfc3339_end(date_str: str) -> str:
    """Convert a bare date or datetime string to RFC 3339 end-of-day."""
    if "T" in date_str or " " in date_str:
        if not (date_str.endswith("Z") or "+" in date_str[10:] or "-" in date_str[10:]):
            return date_str + "Z"
        return date_str
    # bare date YYYY-MM-DD → end of day UTC
    return f"{date_str}T23:59:59Z"


def _fetch_all_events(
    service: Any,
    calendar_id: str,
    time_min: str,
    time_max: str,
    max_results: int,
) -> list[CalendarEvent]:
    """Fetch all pages of events from the Google Calendar API.

    Handles pagination via nextPageToken automatically.
    """
    all_events: list[CalendarEvent] = []
    page_token: str | None = None

    while True:
        kwargs: dict[str, Any] = {
            "calendarId": calendar_id,
            "timeMin": time_min,
            "timeMax": time_max,
            "maxResults": min(max_results, 2500),  # API hard limit
            "singleEvents": True,                   # expand recurring events
            "orderBy": "startTime",
        }
        if page_token:
            kwargs["pageToken"] = page_token

        response: dict[str, Any] = (
            service.events().list(**kwargs).execute()
        )

        raw_items: list[dict] = response.get("items", [])
        for raw in raw_items:
            all_events.append(_parse_event(raw))

        page_token = response.get("nextPageToken")
        if not page_token:
            break

        # Check if we have enough (respects max_results across pages)
        if len(all_events) >= max_results:
            all_events = all_events[:max_results]
            break

    return all_events


def build_calendar_service(
    token_file: "str | Path | None" = None,
    credentials_file: "str | Path | None" = None,
    scopes: "list[str] | None" = None,
) -> Any:
    """Build and return an authenticated Google Calendar API service object.

    Constructs a ``googleapiclient.discovery.Resource`` for the Google Calendar
    v3 API using OAuth2 credentials loaded from a local token file.  When the
    token does not yet exist (first run) the function initiates the standard
    OAuth2 authorisation-code flow using *credentials_file*.

    Parameters
    ----------
    token_file:
        Path to the OAuth2 token JSON file produced by the authorisation flow.
        Defaults to ``~/assistant/credentials/calendar_token.json`` (or the
        value of the ``GOOGLE_CALENDAR_TOKEN_FILE`` environment variable).
    credentials_file:
        Path to the OAuth2 client-secrets JSON file downloaded from the Google
        Cloud Console.  Only required when no valid token exists yet (i.e., the
        first authentication).  Defaults to
        ``~/assistant/credentials/credentials.json`` (or the value of the
        ``GOOGLE_CALENDAR_CREDENTIALS_FILE`` environment variable).
    scopes:
        OAuth2 scopes to request.  Defaults to
        ``["https://www.googleapis.com/auth/calendar"]`` which grants full
        read/write calendar access.

    Returns
    -------
    googleapiclient.discovery.Resource
        An authenticated Google Calendar API v3 service object ready for use.

    Raises
    ------
    RuntimeError
        If the Google API client libraries are not installed, or if the
        required credentials files are missing and no valid token exists.
    """
    import os
    from pathlib import Path as _Path

    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise RuntimeError(
            "Google API client libraries not installed. "
            "Run: pip install google-api-python-client google-auth-httplib2 "
            "google-auth-oauthlib"
        ) from exc

    # Default to full calendar scope (read + write events).
    _scopes: list[str] = scopes or ["https://www.googleapis.com/auth/calendar"]

    # Resolve paths: explicit argument > env var > default under ~/assistant/
    credentials_dir = _Path(
        os.environ.get(
            "ASSISTANT_CREDENTIALS_DIR",
            _Path.home() / "assistant" / "credentials",
        )
    )
    _token_path = _Path(token_file) if token_file is not None else _Path(
        os.environ.get(
            "GOOGLE_CALENDAR_TOKEN_FILE",
            credentials_dir / "calendar_token.json",
        )
    )
    _credentials_path = _Path(credentials_file) if credentials_file is not None else _Path(
        os.environ.get(
            "GOOGLE_CALENDAR_CREDENTIALS_FILE",
            credentials_dir / "credentials.json",
        )
    )

    creds: Any = None

    # Load existing token if available.
    if _token_path.exists():
        creds = Credentials.from_authorized_user_file(str(_token_path), _scopes)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            # Silently refresh using the stored refresh token.
            creds.refresh(Request())
        else:
            # No usable token — start the OAuth2 interactive flow.
            if not _credentials_path.exists():
                raise RuntimeError(
                    f"Google Calendar credentials file not found at {_credentials_path}. "
                    "Download credentials.json from Google Cloud Console and place it there."
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                str(_credentials_path), _scopes
            )
            creds = flow.run_local_server(port=0)

        # Persist the (new or refreshed) token for future runs.
        _token_path.parent.mkdir(parents=True, exist_ok=True)
        _token_path.write_text(creds.to_json())

    return build("calendar", "v3", credentials=creds)


def _build_calendar_service() -> Any:
    """Build and return an authenticated Google Calendar API service object.

    This is an internal convenience wrapper that forwards to the public
    :func:`build_calendar_service` using environment-variable / default-path
    resolution.  Prefer calling :func:`build_calendar_service` directly when
    explicit path control is needed (e.g., in tests).

    Raises
    ------
    RuntimeError
        If credentials cannot be loaded.
    """
    return build_calendar_service()
