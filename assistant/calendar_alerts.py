"""Calendar event alerts via Telegram (AC21).

Periodically checks upcoming Google Calendar events
(:func:`assistant.tools.calendar.get_calendar_events`) and sends a Telegram
notification via :func:`assistant.interfaces.telegram_bot.send_message`
30 minutes before each event starts -- using the ``calendar_alert_time``
already computed by :mod:`assistant.tools.calendar`.

Each event is alerted at most once. Sent alerts are tracked in a small
SQLite table (``calendar_alerts_sent``) in ``~/assistant/data/memory.db``.

:func:`run_calendar_alert_check` is the entry point intended to be invoked
periodically (e.g. every few minutes by a launchd job, see AC20).
"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from assistant.interfaces.telegram_bot import send_message
from assistant.tools.calendar import CalendarEvent, get_calendar_events

logger = logging.getLogger(__name__)

#: Database location: ~/assistant/data/memory.db (shared with SessionStore/NoteStore).
_DEFAULT_DB_DIR = Path.home() / "assistant" / "data"
_DEFAULT_DB_PATH = _DEFAULT_DB_DIR / "memory.db"

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS calendar_alerts_sent (
    event_id TEXT PRIMARY KEY,
    sent_at  TEXT NOT NULL
);
"""

#: How far ahead of *now* to look for events when checking for due alerts.
#: Must be >= the alert lead time (30 minutes) plus a buffer for the check
#: interval, so an event is seen at least once before its alert is due.
_LOOKAHEAD = timedelta(hours=1)


# ---------------------------------------------------------------------------
# Sent-alert tracking
# ---------------------------------------------------------------------------


class AlertStore:
    """Tracks which calendar event alerts have already been sent.

    Args:
        db_path: Path to the SQLite database file. Defaults to
            ``~/assistant/data/memory.db``. The parent directory is created
            automatically if it does not exist.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            db_path = _DEFAULT_DB_PATH
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE_SQL)

    def has_been_sent(self, event_id: str) -> bool:
        """Return ``True`` if an alert for *event_id* has already been sent."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM calendar_alerts_sent WHERE event_id = ?", (event_id,)
            ).fetchone()
        return row is not None

    def mark_sent(self, event_id: str, *, sent_at: str | None = None) -> None:
        """Record that an alert for *event_id* has been sent."""
        sent_at = sent_at or datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO calendar_alerts_sent (event_id, sent_at) VALUES (?, ?)",
                (event_id, sent_at),
            )


# ---------------------------------------------------------------------------
# Alert window logic
# ---------------------------------------------------------------------------


def _parse_iso(value: str) -> datetime:
    """Parse an ISO 8601 date or datetime string into a timezone-aware datetime.

    Bare dates (``YYYY-MM-DD``, as used for all-day events) and strings
    ending in ``Z`` are normalised. Naive datetimes are assumed to be UTC.

    Raises:
        ValueError: If *value* cannot be parsed.
    """
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def find_due_alerts(events: list[CalendarEvent], *, now: datetime | None = None) -> list[CalendarEvent]:
    """Return the events whose alert window currently contains *now*.

    An event's alert is due when ``calendar_alert_time <= now <
    calendar_event_time`` -- i.e. its 30-minute-before alert time has
    arrived but the event has not yet started. Events with unparsable
    times are skipped.

    Args:
        events: Calendar events to check.
        now: Reference time. Defaults to the current UTC time.

    Returns:
        The subset of *events* that are currently due for an alert, in the
        same order as *events*.
    """
    now = now or datetime.now(timezone.utc)

    due: list[CalendarEvent] = []
    for event in events:
        try:
            alert_time = _parse_iso(event.calendar_alert_time)
            event_time = _parse_iso(event.calendar_event_time)
        except ValueError:
            logger.warning("find_due_alerts: skipping event with unparsable time: %r", event)
            continue

        if alert_time <= now < event_time:
            due.append(event)

    return due


def format_alert_message(event: CalendarEvent) -> str:
    """Return the Telegram alert text for *event*."""
    message = f'Reminder: "{event.calendar_event_title}" starts at {event.calendar_event_time} (in 30 minutes).'
    if event.location:
        message += f" Location: {event.location}."
    return message


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def run_calendar_alert_check(
    *,
    chat_id: int | str | None = None,
    store: AlertStore | None = None,
    now: datetime | None = None,
    service: Any | None = None,
    token: str | None = None,
    send: Callable[..., Awaitable[None]] = send_message,
) -> dict[str, Any]:
    """Check for due calendar alerts and send any unsent ones via Telegram (AC21).

    Args:
        chat_id: The Telegram chat ID to send alerts to. Defaults to the
            ``TELEGRAM_CHAT_ID`` environment variable.
        store: Tracks which event alerts have already been sent. Defaults
            to a new :class:`AlertStore`.
        now: Reference time for determining which alerts are due. Defaults
            to the current UTC time.
        service: An already-constructed Google Calendar API service object,
            forwarded to :func:`~assistant.tools.calendar.get_calendar_events`.
        token: Telegram bot token, forwarded to *send*.
        send: The function used to deliver alert messages. Defaults to
            :func:`assistant.interfaces.telegram_bot.send_message`.

    Returns:
        A summary dict with keys ``"checked"`` (number of events fetched),
        ``"due"`` (number of events whose alert window is currently open),
        and ``"sent"`` (list of event IDs an alert was sent for in this
        call).

    Raises:
        RuntimeError: If an alert needs to be sent but no *chat_id* is
            given and ``TELEGRAM_CHAT_ID`` is not set.
    """
    now = now or datetime.now(timezone.utc)
    store = store or AlertStore()

    window_start = now.isoformat()
    window_end = (now + _LOOKAHEAD).isoformat()
    events = get_calendar_events(window_start, window_end, service=service)

    due = find_due_alerts(events, now=now)

    sent: list[str] = []
    for event in due:
        if store.has_been_sent(event.calendar_event_id):
            continue

        resolved_chat_id = chat_id if chat_id is not None else os.environ.get("TELEGRAM_CHAT_ID")
        if not resolved_chat_id:
            raise RuntimeError("TELEGRAM_CHAT_ID environment variable is not set")

        await send(resolved_chat_id, format_alert_message(event), token=token)
        store.mark_sent(event.calendar_event_id, sent_at=now.isoformat())
        sent.append(event.calendar_event_id)
        logger.info("run_calendar_alert_check: sent alert for event %s", event.calendar_event_id)

    return {"checked": len(events), "due": len(due), "sent": sent}


def main() -> None:
    """CLI entry point: run a single calendar alert check and log a summary."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = asyncio.run(run_calendar_alert_check())
    logger.info(
        "Calendar alert check complete: checked %d event(s), %d due, %d alert(s) sent",
        result["checked"],
        result["due"],
        len(result["sent"]),
    )


if __name__ == "__main__":
    main()
