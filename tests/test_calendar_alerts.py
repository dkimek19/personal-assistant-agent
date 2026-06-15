"""Tests for calendar event alerts via Telegram (assistant.calendar_alerts, AC21).

Covers:
- find_due_alerts: selects events whose alert window [calendar_alert_time,
  calendar_event_time) contains "now"; skips events not yet due, events
  already started, and events with unparsable times.
- format_alert_message: human-readable alert text, including location.
- AlertStore: has_been_sent / mark_sent round-trip via tmp_path.
- run_calendar_alert_check: fetches events, sends alerts only for due +
  unsent events, marks them sent, skips already-sent events, and raises
  when no TELEGRAM_CHAT_ID is configured but an alert needs sending.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from assistant.calendar_alerts import (
    AlertStore,
    find_due_alerts,
    format_alert_message,
    run_calendar_alert_check,
)
from assistant.tools.calendar import CalendarEvent


def _make_event(event_id: str, *, start: datetime, location: str = "") -> CalendarEvent:
    alert_time = start - timedelta(minutes=30)
    return CalendarEvent(
        calendar_event_id=event_id,
        calendar_event_title=f"Event {event_id}",
        calendar_event_time=start.isoformat(),
        calendar_alert_time=alert_time.isoformat(),
        end_time=(start + timedelta(hours=1)).isoformat(),
        location=location,
    )


_NOW = datetime(2026, 6, 10, 12, 0, 0, tzinfo=timezone.utc)


class TestFindDueAlerts:
    def test_event_within_alert_window_is_due(self):
        # Starts in 20 minutes -> alert_time was 10 minutes ago.
        event = _make_event("ev1", start=_NOW + timedelta(minutes=20))

        due = find_due_alerts([event], now=_NOW)

        assert due == [event]

    def test_event_more_than_30_minutes_away_is_not_due(self):
        event = _make_event("ev1", start=_NOW + timedelta(hours=1))

        due = find_due_alerts([event], now=_NOW)

        assert due == []

    def test_event_already_started_is_not_due(self):
        event = _make_event("ev1", start=_NOW - timedelta(minutes=5))

        due = find_due_alerts([event], now=_NOW)

        assert due == []

    def test_event_starting_exactly_now_is_not_due(self):
        event = _make_event("ev1", start=_NOW)

        due = find_due_alerts([event], now=_NOW)

        assert due == []

    def test_skips_events_with_unparsable_times(self):
        event = CalendarEvent(
            calendar_event_id="bad",
            calendar_event_title="Bad event",
            calendar_event_time="not-a-date",
            calendar_alert_time="also-not-a-date",
            end_time="not-a-date",
        )

        due = find_due_alerts([event], now=_NOW)

        assert due == []

    def test_returns_multiple_due_events_in_order(self):
        ev1 = _make_event("ev1", start=_NOW + timedelta(minutes=10))
        ev2 = _make_event("ev2", start=_NOW + timedelta(minutes=25))
        not_due = _make_event("ev3", start=_NOW + timedelta(hours=2))

        due = find_due_alerts([ev1, ev2, not_due], now=_NOW)

        assert due == [ev1, ev2]


class TestFormatAlertMessage:
    def test_includes_title_and_time(self):
        event = _make_event("ev1", start=_NOW + timedelta(minutes=20))

        message = format_alert_message(event)

        assert "Event ev1" in message
        assert event.calendar_event_time in message
        assert "30 minutes" in message

    def test_includes_location_when_present(self):
        event = _make_event("ev1", start=_NOW + timedelta(minutes=20), location="Room 42")

        message = format_alert_message(event)

        assert "Room 42" in message

    def test_omits_location_when_absent(self):
        event = _make_event("ev1", start=_NOW + timedelta(minutes=20))

        message = format_alert_message(event)

        assert "Location" not in message


class TestAlertStore:
    def test_has_not_been_sent_initially(self, tmp_path):
        store = AlertStore(db_path=tmp_path / "memory.db")

        assert store.has_been_sent("ev1") is False

    def test_mark_sent_then_has_been_sent_true(self, tmp_path):
        store = AlertStore(db_path=tmp_path / "memory.db")

        store.mark_sent("ev1")

        assert store.has_been_sent("ev1") is True

    def test_mark_sent_is_idempotent(self, tmp_path):
        store = AlertStore(db_path=tmp_path / "memory.db")

        store.mark_sent("ev1")
        store.mark_sent("ev1")

        assert store.has_been_sent("ev1") is True

    def test_other_events_unaffected(self, tmp_path):
        store = AlertStore(db_path=tmp_path / "memory.db")

        store.mark_sent("ev1")

        assert store.has_been_sent("ev2") is False


class TestRunCalendarAlertCheck:
    async def test_sends_alert_for_due_event_and_marks_sent(self, tmp_path):
        event = _make_event("ev1", start=_NOW + timedelta(minutes=20))
        store = AlertStore(db_path=tmp_path / "memory.db")
        send = AsyncMock()

        with patch("assistant.calendar_alerts.get_calendar_events", return_value=[event]) as mock_get:
            result = await run_calendar_alert_check(
                chat_id=12345, store=store, now=_NOW, token="fake-token", send=send
            )

        mock_get.assert_called_once()
        send.assert_awaited_once()
        args, kwargs = send.call_args
        assert args[0] == 12345
        assert "Event ev1" in args[1]
        assert kwargs["token"] == "fake-token"

        assert result["checked"] == 1
        assert result["due"] == 1
        assert result["sent"] == ["ev1"]
        assert store.has_been_sent("ev1") is True

    async def test_does_not_resend_already_sent_alert(self, tmp_path):
        event = _make_event("ev1", start=_NOW + timedelta(minutes=20))
        store = AlertStore(db_path=tmp_path / "memory.db")
        store.mark_sent("ev1")
        send = AsyncMock()

        with patch("assistant.calendar_alerts.get_calendar_events", return_value=[event]):
            result = await run_calendar_alert_check(chat_id=12345, store=store, now=_NOW, send=send)

        send.assert_not_awaited()
        assert result["due"] == 1
        assert result["sent"] == []

    async def test_no_due_events_sends_nothing(self, tmp_path):
        event = _make_event("ev1", start=_NOW + timedelta(hours=2))
        store = AlertStore(db_path=tmp_path / "memory.db")
        send = AsyncMock()

        with patch("assistant.calendar_alerts.get_calendar_events", return_value=[event]):
            result = await run_calendar_alert_check(chat_id=12345, store=store, now=_NOW, send=send)

        send.assert_not_awaited()
        assert result["checked"] == 1
        assert result["due"] == 0
        assert result["sent"] == []

    async def test_missing_chat_id_raises_runtime_error(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        event = _make_event("ev1", start=_NOW + timedelta(minutes=20))
        store = AlertStore(db_path=tmp_path / "memory.db")
        send = AsyncMock()

        with patch("assistant.calendar_alerts.get_calendar_events", return_value=[event]):
            with pytest.raises(RuntimeError):
                await run_calendar_alert_check(store=store, now=_NOW, send=send)

        send.assert_not_awaited()

    async def test_uses_chat_id_env_var_when_not_provided(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "999")
        event = _make_event("ev1", start=_NOW + timedelta(minutes=20))
        store = AlertStore(db_path=tmp_path / "memory.db")
        send = AsyncMock()

        with patch("assistant.calendar_alerts.get_calendar_events", return_value=[event]):
            await run_calendar_alert_check(store=store, now=_NOW, send=send)

        args, _ = send.call_args
        assert args[0] == "999"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
