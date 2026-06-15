"""Tests for assistant.tools.calendar.get_calendar_events.

All tests use a mocked Google Calendar API client with fixture data so they
run fully offline with no real credentials required.

Covers:
- Normal event retrieval returning structured CalendarEvent objects
- All-day event handling
- Pagination (multiple pages)
- Exponential backoff retry on transient API errors
- RuntimeError raised after max retries exhausted
- Alert time computation (30 minutes before event)
- Empty event list
- Data fidelity: factual fields (id, title, time) must not be altered
- Bare date string normalisation to RFC 3339
"""

from __future__ import annotations

import time
import unittest
from datetime import datetime, timezone, timedelta
from typing import Any
from unittest.mock import MagicMock, patch, call

from assistant.tools.calendar import (
    CalendarEvent,
    build_calendar_service,
    create_calendar_event,
    delete_calendar_event,
    get_calendar_events,
    get_today_events,
    update_calendar_event,
    _compute_alert_time,
    _event_start_date,
    _to_rfc3339_start,
    _to_rfc3339_end,
    _parse_event,
)


# ---------------------------------------------------------------------------
# Fixture data
# ---------------------------------------------------------------------------

FIXTURE_EVENTS_PAGE_1 = {
    "kind": "calendar#events",
    "summary": "Test Calendar",
    "nextPageToken": "TOKEN_PAGE_2",
    "items": [
        {
            "id": "evt_001",
            "summary": "Team Standup",
            "start": {"dateTime": "2026-06-10T09:00:00+09:00"},
            "end":   {"dateTime": "2026-06-10T09:30:00+09:00"},
            "description": "Daily team sync",
            "location": "Zoom",
            "attendees": [
                {"email": "alice@example.com"},
                {"email": "bob@example.com"},
            ],
            "htmlLink": "https://calendar.google.com/event?eid=evt_001",
        },
        {
            "id": "evt_002",
            "summary": "Lunch with Client",
            "start": {"dateTime": "2026-06-10T12:00:00+09:00"},
            "end":   {"dateTime": "2026-06-10T13:00:00+09:00"},
            "description": "",
            "location": "Gangnam Restaurant",
            "attendees": [],
            "htmlLink": "https://calendar.google.com/event?eid=evt_002",
        },
    ],
}

FIXTURE_EVENTS_PAGE_2 = {
    "kind": "calendar#events",
    "summary": "Test Calendar",
    "items": [
        {
            "id": "evt_003",
            "summary": "Project Review",
            "start": {"dateTime": "2026-06-10T15:00:00+09:00"},
            "end":   {"dateTime": "2026-06-10T16:00:00+09:00"},
            "description": "Q2 project review meeting",
            "location": "",
            "attendees": [{"email": "carol@example.com"}],
            "htmlLink": "https://calendar.google.com/event?eid=evt_003",
        },
    ],
}

FIXTURE_ALL_DAY_EVENT = {
    "kind": "calendar#events",
    "items": [
        {
            "id": "evt_allday",
            "summary": "Company Holiday",
            "start": {"date": "2026-06-15"},
            "end":   {"date": "2026-06-16"},
            "description": "National holiday",
            "location": "",
            "attendees": [],
            "htmlLink": "https://calendar.google.com/event?eid=evt_allday",
        },
    ],
}

FIXTURE_SINGLE_EVENT = {
    "kind": "calendar#events",
    "items": [
        {
            "id": "evt_single",
            "summary": "Doctor Appointment",
            "start": {"dateTime": "2026-06-12T14:30:00Z"},
            "end":   {"dateTime": "2026-06-12T15:30:00Z"},
            "description": "Annual check-up",
            "location": "City Hospital",
            "attendees": [],
            "htmlLink": "https://calendar.google.com/event?eid=evt_single",
        },
    ],
}

FIXTURE_EMPTY = {
    "kind": "calendar#events",
    "items": [],
}

# Events spanning multiple days, used to exercise get_today_events()'s
# date-filtering logic (only events starting on "today" -- 2026-06-15 in
# these fixtures -- should be returned).
FIXTURE_MULTI_DAY_EVENTS = {
    "kind": "calendar#events",
    "items": [
        {
            "id": "evt_today_1",
            "summary": "Morning Standup",
            "start": {"dateTime": "2026-06-15T09:00:00+09:00"},
            "end": {"dateTime": "2026-06-15T09:30:00+09:00"},
            "description": "Daily team sync",
            "location": "Zoom",
            "attendees": [],
            "htmlLink": "https://calendar.google.com/event?eid=evt_today_1",
        },
        {
            "id": "evt_today_2",
            "summary": "Afternoon Review",
            "start": {"dateTime": "2026-06-15T15:00:00+09:00"},
            "end": {"dateTime": "2026-06-15T16:00:00+09:00"},
            "description": "",
            "location": "",
            "attendees": [],
            "htmlLink": "https://calendar.google.com/event?eid=evt_today_2",
        },
        {
            "id": "evt_yesterday",
            "summary": "Yesterday's Wrap-up",
            "start": {"dateTime": "2026-06-14T17:00:00+09:00"},
            "end": {"dateTime": "2026-06-14T18:00:00+09:00"},
            "description": "",
            "location": "",
            "attendees": [],
            "htmlLink": "https://calendar.google.com/event?eid=evt_yesterday",
        },
        {
            "id": "evt_tomorrow",
            "summary": "Tomorrow's Meeting",
            "start": {"dateTime": "2026-06-16T09:00:00+09:00"},
            "end": {"dateTime": "2026-06-16T10:00:00+09:00"},
            "description": "",
            "location": "",
            "attendees": [],
            "htmlLink": "https://calendar.google.com/event?eid=evt_tomorrow",
        },
    ],
}

FIXTURE_ALL_DAY_EVENT_TODAY = {
    "kind": "calendar#events",
    "items": [
        {
            "id": "evt_allday_today",
            "summary": "Company Holiday",
            "start": {"date": "2026-06-15"},
            "end": {"date": "2026-06-16"},
            "description": "National holiday",
            "location": "",
            "attendees": [],
            "htmlLink": "",
        },
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_service(responses: list[dict]) -> MagicMock:
    """Return a mock Google Calendar API service that yields *responses* in order.

    Each call to ``.events().list(...).execute()`` pops the next response.
    """
    service = MagicMock()
    execute_mock = MagicMock(side_effect=responses)
    service.events.return_value.list.return_value.execute = execute_mock
    return service


def _make_service_with_error(error: Exception, success_response: dict | None = None) -> MagicMock:
    """Return a mock service that raises *error* once, then succeeds (if provided)."""
    service = MagicMock()
    if success_response is not None:
        execute_mock = MagicMock(side_effect=[error, success_response])
    else:
        execute_mock = MagicMock(side_effect=error)
    service.events.return_value.list.return_value.execute = execute_mock
    return service


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGetCalendarEvents(unittest.TestCase):
    """Unit tests for get_calendar_events()."""

    # ------------------------------------------------------------------
    # Basic retrieval
    # ------------------------------------------------------------------

    def test_returns_list_of_calendar_events(self):
        """Should return a non-empty list of CalendarEvent objects."""
        service = _make_service([FIXTURE_SINGLE_EVENT])
        events = get_calendar_events("2026-06-12", "2026-06-12", service=service)

        self.assertIsInstance(events, list)
        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0], CalendarEvent)

    def test_event_fields_match_fixture_data(self):
        """Data fidelity: factual fields must equal the API fixture exactly."""
        service = _make_service([FIXTURE_SINGLE_EVENT])
        events = get_calendar_events("2026-06-12", "2026-06-12", service=service)

        evt = events[0]
        self.assertEqual(evt.calendar_event_id, "evt_single")
        self.assertEqual(evt.calendar_event_title, "Doctor Appointment")
        self.assertEqual(evt.calendar_event_time, "2026-06-12T14:30:00Z")
        self.assertEqual(evt.end_time, "2026-06-12T15:30:00Z")
        self.assertEqual(evt.description, "Annual check-up")
        self.assertEqual(evt.location, "City Hospital")
        self.assertEqual(evt.attendees, [])
        self.assertFalse(evt.is_all_day)
        self.assertEqual(evt.html_link, "https://calendar.google.com/event?eid=evt_single")

    def test_alert_time_is_30_minutes_before_event(self):
        """calendar_alert_time must be exactly 30 minutes before the event start."""
        service = _make_service([FIXTURE_SINGLE_EVENT])
        events = get_calendar_events("2026-06-12", "2026-06-12", service=service)

        evt = events[0]
        # Event starts at 2026-06-12T14:30:00Z → alert at 14:00:00Z
        self.assertEqual(evt.calendar_alert_time, "2026-06-12T14:00:00+00:00")

    def test_multiple_events_returned(self):
        """Should return all events from the response."""
        service = _make_service([FIXTURE_EVENTS_PAGE_1.copy() | {"nextPageToken": None, "items": FIXTURE_EVENTS_PAGE_1["items"]}])
        # Rebuild without pagination for simplicity
        no_page_response = {
            "kind": "calendar#events",
            "items": FIXTURE_EVENTS_PAGE_1["items"],
        }
        service = _make_service([no_page_response])
        events = get_calendar_events("2026-06-10", "2026-06-10", service=service)

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].calendar_event_id, "evt_001")
        self.assertEqual(events[1].calendar_event_id, "evt_002")

    def test_attendees_list_populated(self):
        """attendees field must contain all email addresses from the API response."""
        service = _make_service([{
            "kind": "calendar#events",
            "items": [FIXTURE_EVENTS_PAGE_1["items"][0]],
        }])
        events = get_calendar_events("2026-06-10", "2026-06-10", service=service)

        self.assertEqual(events[0].attendees, ["alice@example.com", "bob@example.com"])

    # ------------------------------------------------------------------
    # Empty result
    # ------------------------------------------------------------------

    def test_empty_event_list(self):
        """Should return an empty list when there are no events."""
        service = _make_service([FIXTURE_EMPTY])
        events = get_calendar_events("2026-06-01", "2026-06-01", service=service)

        self.assertIsInstance(events, list)
        self.assertEqual(len(events), 0)

    # ------------------------------------------------------------------
    # All-day events
    # ------------------------------------------------------------------

    def test_all_day_event_detected(self):
        """Events with date (not dateTime) must have is_all_day=True."""
        service = _make_service([FIXTURE_ALL_DAY_EVENT])
        events = get_calendar_events("2026-06-15", "2026-06-15", service=service)

        self.assertEqual(len(events), 1)
        evt = events[0]
        self.assertTrue(evt.is_all_day)
        self.assertEqual(evt.calendar_event_id, "evt_allday")
        self.assertEqual(evt.calendar_event_title, "Company Holiday")
        self.assertEqual(evt.calendar_event_time, "2026-06-15")
        self.assertEqual(evt.end_time, "2026-06-16")

    def test_all_day_event_alert_time_is_8_30(self):
        """All-day event alert should be 08:30 UTC on the event day (09:00 - 30min)."""
        service = _make_service([FIXTURE_ALL_DAY_EVENT])
        events = get_calendar_events("2026-06-15", "2026-06-15", service=service)

        evt = events[0]
        # anchor = 2026-06-15T09:00:00Z, minus 30 min = 2026-06-15T08:30:00+00:00
        self.assertEqual(evt.calendar_alert_time, "2026-06-15T08:30:00+00:00")

    # ------------------------------------------------------------------
    # Pagination
    # ------------------------------------------------------------------

    def test_pagination_fetches_all_pages(self):
        """Should fetch all pages and return combined event list."""
        page1 = FIXTURE_EVENTS_PAGE_1.copy()  # has nextPageToken
        page2 = FIXTURE_EVENTS_PAGE_2.copy()  # no nextPageToken

        service = _make_service([page1, page2])
        events = get_calendar_events("2026-06-10", "2026-06-10", service=service)

        # 2 from page 1 + 1 from page 2 = 3 total
        self.assertEqual(len(events), 3)
        ids = [e.calendar_event_id for e in events]
        self.assertIn("evt_001", ids)
        self.assertIn("evt_002", ids)
        self.assertIn("evt_003", ids)

    def test_pagination_calls_list_twice(self):
        """Two API list() calls should be made when pagination is needed."""
        page1 = FIXTURE_EVENTS_PAGE_1.copy()
        page2 = FIXTURE_EVENTS_PAGE_2.copy()

        service = _make_service([page1, page2])
        get_calendar_events("2026-06-10", "2026-06-10", service=service)

        self.assertEqual(service.events.return_value.list.call_count, 2)

    # ------------------------------------------------------------------
    # Exponential backoff + retries
    # ------------------------------------------------------------------

    @patch("time.sleep", return_value=None)
    def test_retries_on_transient_error_and_succeeds(self, mock_sleep):
        """Should retry once on transient error and return events on success."""
        transient_error = Exception("503 Service Unavailable")
        service = _make_service_with_error(transient_error, FIXTURE_SINGLE_EVENT)

        events = get_calendar_events(
            "2026-06-12", "2026-06-12",
            service=service,
            max_retries=2,
            base_backoff=1.0,
        )

        self.assertEqual(len(events), 1)
        # Should have slept once (after first failure) with 1.0s backoff
        mock_sleep.assert_called_once_with(1.0)

    @patch("time.sleep", return_value=None)
    def test_second_retry_uses_doubled_backoff(self, mock_sleep):
        """Second retry backoff should be 2× the first."""
        transient_error = Exception("Timeout")

        service = MagicMock()
        service.events.return_value.list.return_value.execute = MagicMock(
            side_effect=[transient_error, transient_error, FIXTURE_SINGLE_EVENT]
        )

        events = get_calendar_events(
            "2026-06-12", "2026-06-12",
            service=service,
            max_retries=2,
            base_backoff=1.0,
        )

        self.assertEqual(len(events), 1)
        # First retry: sleep(1.0), second retry: sleep(2.0)
        self.assertEqual(mock_sleep.call_count, 2)
        calls = mock_sleep.call_args_list
        self.assertAlmostEqual(calls[0][0][0], 1.0)
        self.assertAlmostEqual(calls[1][0][0], 2.0)

    @patch("time.sleep", return_value=None)
    def test_raises_runtime_error_after_max_retries(self, mock_sleep):
        """Should raise RuntimeError with explicit message after all retries fail."""
        persistent_error = Exception("Connection refused")

        service = MagicMock()
        service.events.return_value.list.return_value.execute = MagicMock(
            side_effect=persistent_error
        )

        with self.assertRaises(RuntimeError) as ctx:
            get_calendar_events(
                "2026-06-12", "2026-06-12",
                service=service,
                max_retries=2,
                base_backoff=0.01,
            )

        error_msg = str(ctx.exception)
        self.assertIn("Unable to retrieve calendar events", error_msg)
        self.assertIn("3 attempts", error_msg)   # initial + 2 retries = 3 attempts
        self.assertIn("Connection refused", error_msg)

    @patch("time.sleep", return_value=None)
    def test_retry_count_is_exactly_max_retries(self, mock_sleep):
        """The API should be called exactly max_retries+1 times before giving up."""
        service = MagicMock()
        service.events.return_value.list.return_value.execute = MagicMock(
            side_effect=Exception("always fails")
        )

        with self.assertRaises(RuntimeError):
            get_calendar_events(
                "2026-06-12", "2026-06-12",
                service=service,
                max_retries=2,
                base_backoff=0.01,
            )

        # 1 initial attempt + 2 retries = 3 total execute() calls
        self.assertEqual(
            service.events.return_value.list.return_value.execute.call_count,
            3,
        )

    # ------------------------------------------------------------------
    # API call parameters
    # ------------------------------------------------------------------

    def test_passes_correct_time_range_to_api(self):
        """API should be called with the normalised RFC 3339 time range."""
        service = _make_service([FIXTURE_EMPTY])

        get_calendar_events("2026-06-10", "2026-06-12", service=service)

        call_kwargs = service.events.return_value.list.call_args.kwargs
        self.assertEqual(call_kwargs["timeMin"], "2026-06-10T00:00:00Z")
        self.assertEqual(call_kwargs["timeMax"], "2026-06-12T23:59:59Z")
        self.assertTrue(call_kwargs["singleEvents"])
        self.assertEqual(call_kwargs["orderBy"], "startTime")

    def test_passes_datetime_strings_unchanged(self):
        """Full datetime strings should be forwarded to the API unchanged."""
        service = _make_service([FIXTURE_EMPTY])

        get_calendar_events(
            "2026-06-10T00:00:00Z", "2026-06-12T23:59:59Z",
            service=service,
        )

        call_kwargs = service.events.return_value.list.call_args.kwargs
        self.assertEqual(call_kwargs["timeMin"], "2026-06-10T00:00:00Z")
        self.assertEqual(call_kwargs["timeMax"], "2026-06-12T23:59:59Z")

    def test_uses_primary_calendar_by_default(self):
        """calendarId should default to 'primary'."""
        service = _make_service([FIXTURE_EMPTY])
        get_calendar_events("2026-06-10", "2026-06-10", service=service)

        call_kwargs = service.events.return_value.list.call_args.kwargs
        self.assertEqual(call_kwargs["calendarId"], "primary")

    def test_custom_calendar_id(self):
        """A custom calendarId should be forwarded to the API."""
        service = _make_service([FIXTURE_EMPTY])
        get_calendar_events(
            "2026-06-10", "2026-06-10",
            service=service,
            calendar_id="custom_cal@group.calendar.google.com",
        )

        call_kwargs = service.events.return_value.list.call_args.kwargs
        self.assertEqual(call_kwargs["calendarId"], "custom_cal@group.calendar.google.com")

    # ------------------------------------------------------------------
    # to_dict serialisation
    # ------------------------------------------------------------------

    def test_to_dict_contains_all_required_keys(self):
        """CalendarEvent.to_dict() must include all ontology-required keys."""
        service = _make_service([FIXTURE_SINGLE_EVENT])
        events = get_calendar_events("2026-06-12", "2026-06-12", service=service)

        d = events[0].to_dict()
        required_keys = {
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
        self.assertEqual(required_keys, set(d.keys()))

    def test_to_dict_values_match_fields(self):
        """to_dict() values must equal the dataclass field values exactly."""
        service = _make_service([FIXTURE_SINGLE_EVENT])
        events = get_calendar_events("2026-06-12", "2026-06-12", service=service)

        evt = events[0]
        d = evt.to_dict()
        self.assertEqual(d["calendar_event_id"], evt.calendar_event_id)
        self.assertEqual(d["calendar_event_title"], evt.calendar_event_title)
        self.assertEqual(d["calendar_event_time"], evt.calendar_event_time)
        self.assertEqual(d["calendar_alert_time"], evt.calendar_alert_time)

    # ------------------------------------------------------------------
    # Missing summary / no title edge case
    # ------------------------------------------------------------------

    def test_event_without_summary_uses_no_title(self):
        """Events lacking 'summary' should use '(No title)' as the title."""
        response = {
            "items": [
                {
                    "id": "evt_notitle",
                    "start": {"dateTime": "2026-06-10T10:00:00Z"},
                    "end":   {"dateTime": "2026-06-10T11:00:00Z"},
                }
            ]
        }
        service = _make_service([response])
        events = get_calendar_events("2026-06-10", "2026-06-10", service=service)

        self.assertEqual(events[0].calendar_event_title, "(No title)")


# ---------------------------------------------------------------------------
# Tests for get_today_events (Sub-AC 2a)
# ---------------------------------------------------------------------------

class TestGetTodayEvents(unittest.TestCase):
    """Unit tests for get_today_events().

    All tests use a mocked Google Calendar API client with fixture data so
    they run fully offline. ``today="2026-06-15"`` is passed explicitly
    throughout to make the date-filtering logic deterministic regardless of
    the real current date.
    """

    # ------------------------------------------------------------------
    # Basic retrieval / return type
    # ------------------------------------------------------------------

    def test_returns_list_of_calendar_events(self):
        """Should return a list of CalendarEvent objects."""
        service = _make_service([FIXTURE_MULTI_DAY_EVENTS])
        events = get_today_events(service=service, today="2026-06-15")

        self.assertIsInstance(events, list)
        self.assertTrue(len(events) > 0)
        for evt in events:
            self.assertIsInstance(evt, CalendarEvent)

    def test_empty_event_list_when_no_events_today(self):
        """Should return an empty list when the API returns no events."""
        service = _make_service([FIXTURE_EMPTY])
        events = get_today_events(service=service, today="2026-06-15")

        self.assertIsInstance(events, list)
        self.assertEqual(events, [])

    # ------------------------------------------------------------------
    # Date-filtering logic
    # ------------------------------------------------------------------

    def test_filters_out_events_not_matching_today(self):
        """Only events whose start date equals 'today' must be returned,
        even if the (mocked) API response also includes events from
        adjacent days."""
        service = _make_service([FIXTURE_MULTI_DAY_EVENTS])
        events = get_today_events(service=service, today="2026-06-15")

        ids = {e.calendar_event_id for e in events}
        self.assertEqual(ids, {"evt_today_1", "evt_today_2"})
        self.assertNotIn("evt_yesterday", ids)
        self.assertNotIn("evt_tomorrow", ids)

    def test_passes_today_as_both_start_and_end_date_to_get_calendar_events(self):
        """The underlying API query must be scoped to a single-day range for
        'today'."""
        service = _make_service([FIXTURE_EMPTY])

        get_today_events(service=service, today="2026-06-15")

        call_kwargs = service.events.return_value.list.call_args.kwargs
        self.assertEqual(call_kwargs["timeMin"], "2026-06-15T00:00:00Z")
        self.assertEqual(call_kwargs["timeMax"], "2026-06-15T23:59:59Z")

    def test_all_day_event_on_today_is_included(self):
        """An all-day event whose date matches 'today' must be included."""
        service = _make_service([FIXTURE_ALL_DAY_EVENT_TODAY])
        events = get_today_events(service=service, today="2026-06-15")

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].calendar_event_id, "evt_allday_today")

    def test_defaults_to_current_server_date_when_today_not_provided(self):
        """When `today` is omitted, the function must use
        datetime.now().date().isoformat() as the query date."""
        service = _make_service([FIXTURE_EMPTY])

        with patch("assistant.tools.calendar.datetime") as mock_datetime:
            mock_datetime.now.return_value.date.return_value.isoformat.return_value = (
                "2026-06-15"
            )
            get_today_events(service=service)

        call_kwargs = service.events.return_value.list.call_args.kwargs
        self.assertEqual(call_kwargs["timeMin"], "2026-06-15T00:00:00Z")
        self.assertEqual(call_kwargs["timeMax"], "2026-06-15T23:59:59Z")

    # ------------------------------------------------------------------
    # Returned schema: event objects expose title, start/end time, etc.
    # ------------------------------------------------------------------

    def test_returned_event_schema_has_title_and_start_end_times(self):
        """Each returned event must expose calendar_event_title,
        calendar_event_time (start), and end_time, plus the full ontology
        field set via to_dict()."""
        service = _make_service([FIXTURE_MULTI_DAY_EVENTS])
        events = get_today_events(service=service, today="2026-06-15")

        # Sorted ascending by start time -> first event is the morning standup
        evt = events[0]
        self.assertEqual(evt.calendar_event_title, "Morning Standup")
        self.assertEqual(evt.calendar_event_time, "2026-06-15T09:00:00+09:00")
        self.assertEqual(evt.end_time, "2026-06-15T09:30:00+09:00")

        d = evt.to_dict()
        required_keys = {
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
        self.assertEqual(required_keys, set(d.keys()))

    # ------------------------------------------------------------------
    # Error propagation
    # ------------------------------------------------------------------

    @patch("time.sleep", return_value=None)
    def test_propagates_runtime_error_from_get_calendar_events(self, mock_sleep):
        """A RuntimeError from the underlying API (after retries) must
        propagate to the caller."""
        service = MagicMock()
        service.events.return_value.list.return_value.execute = MagicMock(
            side_effect=Exception("Connection refused")
        )

        with self.assertRaises(RuntimeError):
            get_today_events(service=service, today="2026-06-15", max_retries=0)


# ---------------------------------------------------------------------------
# Tests for _event_start_date helper (Sub-AC 2a)
# ---------------------------------------------------------------------------

class TestEventStartDate(unittest.TestCase):
    """Tests for _event_start_date()."""

    def test_extracts_date_from_full_datetime(self):
        evt = _parse_event(FIXTURE_MULTI_DAY_EVENTS["items"][0])
        self.assertEqual(_event_start_date(evt), "2026-06-15")

    def test_extracts_date_from_all_day_event(self):
        evt = _parse_event(FIXTURE_ALL_DAY_EVENT_TODAY["items"][0])
        self.assertEqual(_event_start_date(evt), "2026-06-15")


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------

class TestComputeAlertTime(unittest.TestCase):
    """Tests for _compute_alert_time()."""

    def test_utc_datetime(self):
        result = _compute_alert_time("2026-06-10T09:00:00Z")
        # 09:00 - 30min = 08:30 UTC
        self.assertEqual(result, "2026-06-10T08:30:00+00:00")

    def test_offset_aware_datetime(self):
        result = _compute_alert_time("2026-06-10T09:00:00+09:00")
        # 09:00 KST - 30min = 08:30 KST = 2026-06-10T08:30:00+09:00
        self.assertIn("08:30:00", result)

    def test_all_day_date_string(self):
        result = _compute_alert_time("2026-06-15")
        # anchor = 09:00 UTC, minus 30 = 08:30 UTC
        self.assertEqual(result, "2026-06-15T08:30:00+00:00")

    def test_midnight_event(self):
        result = _compute_alert_time("2026-06-10T00:00:00Z")
        # 00:00 - 30min = 2026-06-09T23:30:00+00:00
        self.assertEqual(result, "2026-06-09T23:30:00+00:00")


class TestRfc3339Normalisation(unittest.TestCase):
    """Tests for _to_rfc3339_start() and _to_rfc3339_end()."""

    def test_start_bare_date(self):
        self.assertEqual(_to_rfc3339_start("2026-06-10"), "2026-06-10T00:00:00Z")

    def test_end_bare_date(self):
        self.assertEqual(_to_rfc3339_end("2026-06-10"), "2026-06-10T23:59:59Z")

    def test_start_datetime_with_z_unchanged(self):
        self.assertEqual(
            _to_rfc3339_start("2026-06-10T08:00:00Z"),
            "2026-06-10T08:00:00Z",
        )

    def test_end_datetime_with_z_unchanged(self):
        self.assertEqual(
            _to_rfc3339_end("2026-06-10T20:00:00Z"),
            "2026-06-10T20:00:00Z",
        )

    def test_start_datetime_without_z_appended(self):
        result = _to_rfc3339_start("2026-06-10T00:00:00")
        self.assertTrue(result.endswith("Z"))

    def test_end_datetime_offset_unchanged(self):
        result = _to_rfc3339_end("2026-06-10T23:59:59+09:00")
        self.assertEqual(result, "2026-06-10T23:59:59+09:00")


class TestParseEvent(unittest.TestCase):
    """Tests for _parse_event()."""

    def test_parse_full_event(self):
        raw = FIXTURE_EVENTS_PAGE_1["items"][0]
        evt = _parse_event(raw)

        self.assertEqual(evt.calendar_event_id, "evt_001")
        self.assertEqual(evt.calendar_event_title, "Team Standup")
        self.assertEqual(evt.calendar_event_time, "2026-06-10T09:00:00+09:00")
        self.assertEqual(evt.end_time, "2026-06-10T09:30:00+09:00")
        self.assertEqual(evt.description, "Daily team sync")
        self.assertEqual(evt.location, "Zoom")
        self.assertFalse(evt.is_all_day)
        self.assertEqual(evt.attendees, ["alice@example.com", "bob@example.com"])

    def test_parse_all_day_event(self):
        raw = FIXTURE_ALL_DAY_EVENT["items"][0]
        evt = _parse_event(raw)

        self.assertTrue(evt.is_all_day)
        self.assertEqual(evt.calendar_event_time, "2026-06-15")
        self.assertEqual(evt.end_time, "2026-06-16")

    def test_parse_event_without_attendees(self):
        raw = FIXTURE_EVENTS_PAGE_1["items"][1]
        evt = _parse_event(raw)
        self.assertEqual(evt.attendees, [])

    def test_parse_event_missing_id_defaults_empty(self):
        raw = {"summary": "No ID event", "start": {"dateTime": "2026-06-10T10:00:00Z"}, "end": {"dateTime": "2026-06-10T11:00:00Z"}}
        evt = _parse_event(raw)
        self.assertEqual(evt.calendar_event_id, "")


# ---------------------------------------------------------------------------
# Tests for create_calendar_event
# ---------------------------------------------------------------------------

class TestCreateCalendarEvent(unittest.TestCase):
    """Unit tests for create_calendar_event().

    All tests use a mocked Google Calendar API client (insert endpoint) so
    they run fully offline without real credentials.

    Covers:
    - Returns the created event ID from the API response
    - Correct payload construction: summary, start, end, description
    - calendarId defaults to 'primary', accepts custom value
    - Exponential backoff retry on transient API errors
    - RuntimeError raised after all retries exhausted
    - API called exactly (max_retries + 1) times before giving up
    """

    # ------------------------------------------------------------------
    # Mock helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_insert_service(response: dict) -> MagicMock:
        """Return a mock service whose insert().execute() returns *response*."""
        service = MagicMock()
        service.events.return_value.insert.return_value.execute = MagicMock(
            return_value=response
        )
        return service

    @staticmethod
    def _make_insert_service_with_errors(
        errors: list[Exception],
        success_response: dict | None = None,
    ) -> MagicMock:
        """Return a mock service that raises *errors* in sequence, then succeeds."""
        service = MagicMock()
        side_effects: list = list(errors)
        if success_response is not None:
            side_effects.append(success_response)
        service.events.return_value.insert.return_value.execute = MagicMock(
            side_effect=side_effects
        )
        return service

    # ------------------------------------------------------------------
    # Return value: created event ID
    # ------------------------------------------------------------------

    def test_returns_created_event_id(self):
        """create_calendar_event must return the 'id' from the API response."""
        service = self._make_insert_service({"id": "created_evt_001", "summary": "Team Meeting"})

        event_id = create_calendar_event(
            title="Team Meeting",
            start_time="2026-06-10T09:00:00Z",
            end_time="2026-06-10T10:00:00Z",
            description="Quarterly review",
            service=service,
        )

        self.assertEqual(event_id, "created_evt_001")

    def test_returns_exact_id_from_api_no_modification(self):
        """The returned ID must be the exact string from the API — not altered."""
        raw_id = "abc123XYZ_calendar_event_id"
        service = self._make_insert_service({"id": raw_id})

        event_id = create_calendar_event(
            title="Meeting",
            start_time="2026-06-10T09:00:00Z",
            end_time="2026-06-10T10:00:00Z",
            service=service,
        )

        self.assertEqual(event_id, raw_id)

    def test_returns_empty_string_when_id_missing_from_response(self):
        """If the API response lacks 'id', the function should return empty string."""
        service = self._make_insert_service({"summary": "No ID"})

        event_id = create_calendar_event(
            title="No ID Event",
            start_time="2026-06-10T09:00:00Z",
            end_time="2026-06-10T10:00:00Z",
            service=service,
        )

        self.assertEqual(event_id, "")

    # ------------------------------------------------------------------
    # Payload construction
    # ------------------------------------------------------------------

    def test_payload_summary_matches_title(self):
        """The 'summary' field in the API body must equal the title argument."""
        service = self._make_insert_service({"id": "evt_001"})

        create_calendar_event(
            title="Doctor Appointment",
            start_time="2026-06-12T14:30:00Z",
            end_time="2026-06-12T15:30:00Z",
            service=service,
        )

        body = service.events.return_value.insert.call_args.kwargs["body"]
        self.assertEqual(body["summary"], "Doctor Appointment")

    def test_payload_start_datetime_matches_argument(self):
        """The start dateTime in the API payload must equal start_time argument."""
        service = self._make_insert_service({"id": "evt_001"})

        create_calendar_event(
            title="Test Event",
            start_time="2026-06-10T14:00:00+09:00",
            end_time="2026-06-10T15:00:00+09:00",
            service=service,
        )

        body = service.events.return_value.insert.call_args.kwargs["body"]
        self.assertEqual(body["start"]["dateTime"], "2026-06-10T14:00:00+09:00")

    def test_payload_end_datetime_matches_argument(self):
        """The end dateTime in the API payload must equal end_time argument."""
        service = self._make_insert_service({"id": "evt_001"})

        create_calendar_event(
            title="Test Event",
            start_time="2026-06-10T14:00:00Z",
            end_time="2026-06-10T15:30:00Z",
            service=service,
        )

        body = service.events.return_value.insert.call_args.kwargs["body"]
        self.assertEqual(body["end"]["dateTime"], "2026-06-10T15:30:00Z")

    def test_payload_description_matches_argument(self):
        """The 'description' field in the API payload must equal the description argument."""
        service = self._make_insert_service({"id": "evt_001"})

        create_calendar_event(
            title="Sprint Planning",
            start_time="2026-06-10T10:00:00Z",
            end_time="2026-06-10T12:00:00Z",
            description="Plan Q3 sprint backlog and assign story points",
            service=service,
        )

        body = service.events.return_value.insert.call_args.kwargs["body"]
        self.assertEqual(
            body["description"],
            "Plan Q3 sprint backlog and assign story points",
        )

    def test_payload_description_empty_string_when_omitted(self):
        """When description is not provided, the payload must contain an empty string."""
        service = self._make_insert_service({"id": "evt_001"})

        create_calendar_event(
            title="Quick Sync",
            start_time="2026-06-10T09:00:00Z",
            end_time="2026-06-10T09:15:00Z",
            service=service,
        )

        body = service.events.return_value.insert.call_args.kwargs["body"]
        self.assertEqual(body["description"], "")

    def test_payload_has_all_required_keys(self):
        """The event body must contain 'summary', 'description', 'start', and 'end'."""
        service = self._make_insert_service({"id": "evt_001"})

        create_calendar_event(
            title="Complete Payload Test",
            start_time="2026-06-10T09:00:00Z",
            end_time="2026-06-10T10:00:00Z",
            description="Test description",
            service=service,
        )

        body = service.events.return_value.insert.call_args.kwargs["body"]
        self.assertIn("summary", body)
        self.assertIn("description", body)
        self.assertIn("start", body)
        self.assertIn("end", body)
        self.assertIn("dateTime", body["start"])
        self.assertIn("dateTime", body["end"])

    # ------------------------------------------------------------------
    # calendarId parameter
    # ------------------------------------------------------------------

    def test_uses_primary_calendar_by_default(self):
        """calendarId must default to 'primary' when not specified."""
        service = self._make_insert_service({"id": "evt_001"})

        create_calendar_event(
            title="Test",
            start_time="2026-06-10T09:00:00Z",
            end_time="2026-06-10T10:00:00Z",
            service=service,
        )

        call_kwargs = service.events.return_value.insert.call_args.kwargs
        self.assertEqual(call_kwargs["calendarId"], "primary")

    def test_custom_calendar_id_forwarded_to_api(self):
        """A custom calendar_id must be forwarded to the API insert call."""
        service = self._make_insert_service({"id": "evt_001"})

        create_calendar_event(
            title="Test",
            start_time="2026-06-10T09:00:00Z",
            end_time="2026-06-10T10:00:00Z",
            service=service,
            calendar_id="work@group.calendar.google.com",
        )

        call_kwargs = service.events.return_value.insert.call_args.kwargs
        self.assertEqual(call_kwargs["calendarId"], "work@group.calendar.google.com")

    # ------------------------------------------------------------------
    # Exponential backoff and retry
    # ------------------------------------------------------------------

    @patch("time.sleep", return_value=None)
    def test_retries_once_on_transient_error_and_returns_id(self, mock_sleep):
        """Should retry after a transient error and return the event ID on success."""
        transient_error = Exception("503 Service Unavailable")
        service = self._make_insert_service_with_errors(
            [transient_error],
            success_response={"id": "evt_retry_ok"},
        )

        event_id = create_calendar_event(
            title="Retry Test",
            start_time="2026-06-10T09:00:00Z",
            end_time="2026-06-10T10:00:00Z",
            service=service,
            max_retries=2,
            base_backoff=1.0,
        )

        self.assertEqual(event_id, "evt_retry_ok")
        # One sleep call after the first failure with base_backoff delay
        mock_sleep.assert_called_once_with(1.0)

    @patch("time.sleep", return_value=None)
    def test_second_retry_uses_doubled_backoff(self, mock_sleep):
        """Second retry backoff must be 2× the first (exponential)."""
        service = self._make_insert_service_with_errors(
            [Exception("Timeout"), Exception("Timeout")],
            success_response={"id": "evt_second_retry_ok"},
        )

        event_id = create_calendar_event(
            title="Double Retry Test",
            start_time="2026-06-10T09:00:00Z",
            end_time="2026-06-10T10:00:00Z",
            service=service,
            max_retries=2,
            base_backoff=1.0,
        )

        self.assertEqual(event_id, "evt_second_retry_ok")
        self.assertEqual(mock_sleep.call_count, 2)
        calls = mock_sleep.call_args_list
        self.assertAlmostEqual(calls[0][0][0], 1.0)   # first retry: 1.0s
        self.assertAlmostEqual(calls[1][0][0], 2.0)   # second retry: 2.0s

    @patch("time.sleep", return_value=None)
    def test_raises_runtime_error_after_max_retries_exhausted(self, mock_sleep):
        """Must raise RuntimeError with an explicit message after all attempts fail."""
        service = MagicMock()
        service.events.return_value.insert.return_value.execute = MagicMock(
            side_effect=Exception("Connection refused")
        )

        with self.assertRaises(RuntimeError) as ctx:
            create_calendar_event(
                title="Failing Event",
                start_time="2026-06-10T09:00:00Z",
                end_time="2026-06-10T10:00:00Z",
                service=service,
                max_retries=2,
                base_backoff=0.01,
            )

        error_msg = str(ctx.exception)
        self.assertIn("Unable to create calendar event", error_msg)
        self.assertIn("3 attempts", error_msg)        # initial + 2 retries = 3
        self.assertIn("Connection refused", error_msg)

    @patch("time.sleep", return_value=None)
    def test_api_called_exactly_max_retries_plus_one_times(self, mock_sleep):
        """The insert API must be called exactly (max_retries + 1) times before giving up."""
        service = MagicMock()
        service.events.return_value.insert.return_value.execute = MagicMock(
            side_effect=Exception("always fails")
        )

        with self.assertRaises(RuntimeError):
            create_calendar_event(
                title="Failing Event",
                start_time="2026-06-10T09:00:00Z",
                end_time="2026-06-10T10:00:00Z",
                service=service,
                max_retries=2,
                base_backoff=0.01,
            )

        # 1 initial attempt + 2 retries = 3 total execute() calls
        self.assertEqual(
            service.events.return_value.insert.return_value.execute.call_count,
            3,
        )

    @patch("time.sleep", return_value=None)
    def test_zero_retries_calls_api_once_then_raises(self, mock_sleep):
        """With max_retries=0, the API should be called exactly once before raising."""
        service = MagicMock()
        service.events.return_value.insert.return_value.execute = MagicMock(
            side_effect=Exception("immediate fail")
        )

        with self.assertRaises(RuntimeError):
            create_calendar_event(
                title="No Retry Test",
                start_time="2026-06-10T09:00:00Z",
                end_time="2026-06-10T10:00:00Z",
                service=service,
                max_retries=0,
            )

        self.assertEqual(
            service.events.return_value.insert.return_value.execute.call_count,
            1,
        )
        mock_sleep.assert_not_called()

    # ------------------------------------------------------------------
    # Data fidelity: payload must not be altered
    # ------------------------------------------------------------------

    def test_title_with_special_characters_preserved_in_payload(self):
        """Special characters in the title must be preserved exactly in the payload."""
        special_title = "Møte — Årsgjennomgang (2026) & Q&A"
        service = self._make_insert_service({"id": "evt_special"})

        create_calendar_event(
            title=special_title,
            start_time="2026-06-10T09:00:00Z",
            end_time="2026-06-10T10:00:00Z",
            service=service,
        )

        body = service.events.return_value.insert.call_args.kwargs["body"]
        self.assertEqual(body["summary"], special_title)

    def test_iso8601_timezone_offset_preserved_in_payload(self):
        """Timezone offset in start_time / end_time must be preserved unchanged."""
        service = self._make_insert_service({"id": "evt_tz"})

        create_calendar_event(
            title="TZ Test",
            start_time="2026-06-10T09:00:00+09:00",
            end_time="2026-06-10T10:00:00+09:00",
            service=service,
        )

        body = service.events.return_value.insert.call_args.kwargs["body"]
        self.assertEqual(body["start"]["dateTime"], "2026-06-10T09:00:00+09:00")
        self.assertEqual(body["end"]["dateTime"], "2026-06-10T10:00:00+09:00")


# ---------------------------------------------------------------------------
# Tests for build_calendar_service
# ---------------------------------------------------------------------------

class TestBuildCalendarService(unittest.TestCase):
    """Unit tests for the public build_calendar_service() function.

    All tests mock the Google API client libraries via sys.modules so that
    this test suite runs fully offline without requiring those packages to be
    installed.

    Coverage:
    - Returns the service object produced by googleapiclient.discovery.build()
    - Passes the token file path to Credentials.from_authorized_user_file()
    - Calls build("calendar", "v3", credentials=<creds>)
    - Uses the default Calendar scope when no scopes are provided
    - Accepts a custom scopes list and forwards it to Credentials
    - Refreshes expired credentials and persists the refreshed token
    - Skips Credentials.from_authorized_user_file when token file is absent
    - Raises RuntimeError when credentials_file is missing and no token exists
    - Raises RuntimeError when Google libraries cannot be imported
    - Forwards custom token_file and credentials_file paths correctly
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_google_mocks(
        self,
        *,
        token_file_exists: bool = True,
        creds_valid: bool = True,
        creds_expired: bool = False,
        creds_has_refresh_token: bool = False,
        credentials_file_exists: bool = True,
    ) -> tuple[dict, "MagicMock", "MagicMock", "MagicMock"]:
        """Build a sys.modules patch dict and key mock objects.

        Returns
        -------
        tuple of (sys_modules_patch, mock_credentials_class, mock_creds, mock_build)
        """
        # Mock credentials instance
        mock_creds = MagicMock()
        mock_creds.valid = creds_valid
        mock_creds.expired = creds_expired
        mock_creds.refresh_token = "refresh_tok" if creds_has_refresh_token else None
        mock_creds.to_json.return_value = '{"token": "fake"}'

        # Mock Credentials class
        mock_credentials_class = MagicMock()
        mock_credentials_class.from_authorized_user_file.return_value = mock_creds

        # Mock InstalledAppFlow
        mock_flow_instance = MagicMock()
        mock_flow_instance.run_local_server.return_value = mock_creds
        mock_flow_class = MagicMock()
        mock_flow_class.from_client_secrets_file.return_value = mock_flow_instance

        # Mock Request (used for token refresh)
        mock_request_class = MagicMock()

        # Mock googleapiclient build
        mock_service = MagicMock(name="CalendarService")
        mock_build = MagicMock(return_value=mock_service)

        # Compose the sys.modules patch dict
        sys_modules_patch = {
            "google": MagicMock(),
            "google.oauth2": MagicMock(),
            "google.oauth2.credentials": MagicMock(Credentials=mock_credentials_class),
            "google.auth": MagicMock(),
            "google.auth.transport": MagicMock(),
            "google.auth.transport.requests": MagicMock(Request=mock_request_class),
            "google_auth_oauthlib": MagicMock(),
            "google_auth_oauthlib.flow": MagicMock(InstalledAppFlow=mock_flow_class),
            "googleapiclient": MagicMock(),
            "googleapiclient.discovery": MagicMock(build=mock_build),
        }

        return sys_modules_patch, mock_credentials_class, mock_creds, mock_build

    def _call_with_token_file(
        self,
        tmp_path: str,
        sys_modules_patch: dict,
        *,
        token_file_exists: bool = True,
        credentials_file_exists: bool = True,
        extra_kwargs: dict | None = None,
    ) -> Any:
        """Helper that patches sys.modules and calls build_calendar_service()."""
        import sys
        import tempfile
        import os

        with patch.dict(sys.modules, sys_modules_patch):
            token_path = os.path.join(tmp_path, "calendar_token.json")
            creds_path = os.path.join(tmp_path, "credentials.json")

            if token_file_exists:
                # Create an actual file so Path.exists() returns True
                with open(token_path, "w") as fh:
                    fh.write('{"token": "fake"}')
            if credentials_file_exists:
                with open(creds_path, "w") as fh:
                    fh.write('{"installed": {}}')

            kwargs: dict = {
                "token_file": token_path if token_file_exists else token_path + ".missing",
            }
            if credentials_file_exists:
                kwargs["credentials_file"] = creds_path
            else:
                kwargs["credentials_file"] = creds_path + ".missing"
            if extra_kwargs:
                kwargs.update(extra_kwargs)

            return build_calendar_service(**kwargs)

    # ------------------------------------------------------------------
    # Core: returns correctly initialised service
    # ------------------------------------------------------------------

    def test_returns_service_object(self):
        """build_calendar_service() must return the object produced by build()."""
        import sys
        import tempfile

        sys_modules_patch, _, _, mock_build = self._make_google_mocks()
        mock_service = mock_build.return_value

        with tempfile.TemporaryDirectory() as tmp:
            result = self._call_with_token_file(tmp, sys_modules_patch)

        self.assertIs(result, mock_service)

    def test_build_called_with_calendar_v3(self):
        """googleapiclient.discovery.build must be called with 'calendar' and 'v3'."""
        import sys
        import tempfile

        sys_modules_patch, _, _, mock_build = self._make_google_mocks()

        with tempfile.TemporaryDirectory() as tmp:
            self._call_with_token_file(tmp, sys_modules_patch)

        mock_build.assert_called_once()
        call_args = mock_build.call_args
        self.assertEqual(call_args.args[0], "calendar")
        self.assertEqual(call_args.args[1], "v3")

    def test_build_called_with_credentials_kwarg(self):
        """build() must receive the loaded Credentials object as 'credentials' kwarg."""
        import sys
        import tempfile

        sys_modules_patch, mock_creds_class, mock_creds, mock_build = self._make_google_mocks()

        with tempfile.TemporaryDirectory() as tmp:
            self._call_with_token_file(tmp, sys_modules_patch)

        call_kwargs = mock_build.call_args.kwargs
        self.assertIn("credentials", call_kwargs)
        # The credentials passed to build must be the same object loaded from the token file
        self.assertIs(call_kwargs["credentials"], mock_creds)

    # ------------------------------------------------------------------
    # Token file: path forwarding and Credentials loading
    # ------------------------------------------------------------------

    def test_from_authorized_user_file_called_with_token_path(self):
        """Credentials.from_authorized_user_file must be called with the exact token path."""
        import sys
        import tempfile
        import os

        sys_modules_patch, mock_creds_class, _, _ = self._make_google_mocks()

        with tempfile.TemporaryDirectory() as tmp:
            token_path = os.path.join(tmp, "calendar_token.json")
            with open(token_path, "w") as fh:
                fh.write('{"token": "fake"}')
            creds_path = os.path.join(tmp, "credentials.json")
            with open(creds_path, "w") as fh:
                fh.write("{}")

            with patch.dict(sys.modules, sys_modules_patch):
                build_calendar_service(token_file=token_path, credentials_file=creds_path)

        mock_creds_class.from_authorized_user_file.assert_called_once_with(
            token_path,
            ["https://www.googleapis.com/auth/calendar"],
        )

    def test_from_authorized_user_file_not_called_when_token_absent(self):
        """When the token file does not exist, Credentials.from_authorized_user_file
        must NOT be called (the OAuth flow should be used instead)."""
        import sys
        import tempfile
        import os

        # creds_valid=False so we don't get "valid creds" shortcut
        sys_modules_patch, mock_creds_class, mock_creds, _ = self._make_google_mocks(
            token_file_exists=False,
            creds_valid=True,
        )
        # Simulate the flow producing valid creds
        mock_flow_class = sys_modules_patch["google_auth_oauthlib.flow"].InstalledAppFlow
        mock_flow_class.from_client_secrets_file.return_value.run_local_server.return_value = mock_creds

        with tempfile.TemporaryDirectory() as tmp:
            creds_path = os.path.join(tmp, "credentials.json")
            with open(creds_path, "w") as fh:
                fh.write("{}")
            missing_token = os.path.join(tmp, "no_token.json")  # does NOT exist

            with patch.dict(sys.modules, sys_modules_patch):
                build_calendar_service(
                    token_file=missing_token,
                    credentials_file=creds_path,
                )

        mock_creds_class.from_authorized_user_file.assert_not_called()

    # ------------------------------------------------------------------
    # Scopes
    # ------------------------------------------------------------------

    def test_default_scope_is_full_calendar_access(self):
        """Without explicit scopes, the default full calendar scope must be used."""
        import sys
        import tempfile

        sys_modules_patch, mock_creds_class, _, _ = self._make_google_mocks()

        with tempfile.TemporaryDirectory() as tmp:
            self._call_with_token_file(tmp, sys_modules_patch)

        # Scope is passed as second argument to from_authorized_user_file
        _, scope_arg = mock_creds_class.from_authorized_user_file.call_args.args
        self.assertIn("https://www.googleapis.com/auth/calendar", scope_arg)

    def test_custom_scopes_forwarded_to_credentials(self):
        """When custom scopes are passed, they must be forwarded to Credentials."""
        import sys
        import tempfile
        import os

        sys_modules_patch, mock_creds_class, _, _ = self._make_google_mocks()
        custom_scopes = [
            "https://www.googleapis.com/auth/calendar.readonly",
            "https://www.googleapis.com/auth/calendar.events",
        ]

        with tempfile.TemporaryDirectory() as tmp:
            token_path = os.path.join(tmp, "calendar_token.json")
            with open(token_path, "w") as fh:
                fh.write('{"token": "fake"}')
            creds_path = os.path.join(tmp, "credentials.json")
            with open(creds_path, "w") as fh:
                fh.write("{}")

            with patch.dict(sys.modules, sys_modules_patch):
                build_calendar_service(
                    token_file=token_path,
                    credentials_file=creds_path,
                    scopes=custom_scopes,
                )

        _, scope_arg = mock_creds_class.from_authorized_user_file.call_args.args
        self.assertEqual(scope_arg, custom_scopes)

    # ------------------------------------------------------------------
    # Credential refresh
    # ------------------------------------------------------------------

    def test_expired_credentials_are_refreshed(self):
        """Expired credentials with a refresh token must trigger creds.refresh()."""
        import sys
        import tempfile
        import os

        sys_modules_patch, mock_creds_class, mock_creds, _ = self._make_google_mocks(
            creds_valid=False,
            creds_expired=True,
            creds_has_refresh_token=True,
        )

        with tempfile.TemporaryDirectory() as tmp:
            token_path = os.path.join(tmp, "calendar_token.json")
            with open(token_path, "w") as fh:
                fh.write('{"token": "fake"}')
            creds_path = os.path.join(tmp, "credentials.json")
            with open(creds_path, "w") as fh:
                fh.write("{}")

            with patch.dict(sys.modules, sys_modules_patch):
                build_calendar_service(token_file=token_path, credentials_file=creds_path)

        mock_creds.refresh.assert_called_once()

    def test_refreshed_token_written_to_file(self):
        """After refreshing expired credentials the updated token must be persisted."""
        import sys
        import tempfile
        import os

        sys_modules_patch, mock_creds_class, mock_creds, _ = self._make_google_mocks(
            creds_valid=False,
            creds_expired=True,
            creds_has_refresh_token=True,
        )

        with tempfile.TemporaryDirectory() as tmp:
            token_path = os.path.join(tmp, "calendar_token.json")
            with open(token_path, "w") as fh:
                fh.write('{"token": "fake"}')
            creds_path = os.path.join(tmp, "credentials.json")
            with open(creds_path, "w") as fh:
                fh.write("{}")

            with patch.dict(sys.modules, sys_modules_patch):
                build_calendar_service(token_file=token_path, credentials_file=creds_path)

        # creds.to_json() must have been called and written to disk
        mock_creds.to_json.assert_called_once()

    # ------------------------------------------------------------------
    # Error cases
    # ------------------------------------------------------------------

    def test_raises_runtime_error_when_google_libs_not_installed(self):
        """ImportError for google libraries must be re-raised as RuntimeError."""
        import sys

        # Ensure google modules are absent from sys.modules for this test
        modules_to_remove = [
            "google", "google.oauth2", "google.oauth2.credentials",
            "google.auth", "google.auth.transport", "google.auth.transport.requests",
            "google_auth_oauthlib", "google_auth_oauthlib.flow",
            "googleapiclient", "googleapiclient.discovery",
        ]
        absent_modules = {k: None for k in modules_to_remove}

        with patch.dict(sys.modules, absent_modules):
            with self.assertRaises(RuntimeError) as ctx:
                import tempfile
                import os
                with tempfile.TemporaryDirectory() as tmp:
                    build_calendar_service(
                        token_file=os.path.join(tmp, "token.json"),
                        credentials_file=os.path.join(tmp, "credentials.json"),
                    )

        self.assertIn("not installed", str(ctx.exception))

    def test_raises_runtime_error_when_credentials_file_missing(self):
        """RuntimeError must be raised when no token exists and credentials_file is absent."""
        import sys
        import tempfile
        import os

        sys_modules_patch, mock_creds_class, _, _ = self._make_google_mocks(
            token_file_exists=False,
            credentials_file_exists=False,
            creds_valid=False,
        )

        with tempfile.TemporaryDirectory() as tmp:
            missing_token = os.path.join(tmp, "no_token.json")
            missing_creds = os.path.join(tmp, "no_credentials.json")
            # Neither file is created → both paths are absent

            with patch.dict(sys.modules, sys_modules_patch):
                with self.assertRaises(RuntimeError) as ctx:
                    build_calendar_service(
                        token_file=missing_token,
                        credentials_file=missing_creds,
                    )

        self.assertIn("credentials file not found", str(ctx.exception))

    # ------------------------------------------------------------------
    # Path forwarding: custom token_file and credentials_file
    # ------------------------------------------------------------------

    def test_custom_token_file_path_used(self):
        """The explicit token_file argument must take precedence over env vars."""
        import sys
        import tempfile
        import os

        sys_modules_patch, mock_creds_class, _, _ = self._make_google_mocks()

        with tempfile.TemporaryDirectory() as tmp:
            custom_token = os.path.join(tmp, "my_custom_token.json")
            with open(custom_token, "w") as fh:
                fh.write('{"token": "fake"}')
            creds_path = os.path.join(tmp, "credentials.json")
            with open(creds_path, "w") as fh:
                fh.write("{}")

            with patch.dict(sys.modules, sys_modules_patch):
                build_calendar_service(
                    token_file=custom_token,
                    credentials_file=creds_path,
                )

        # The custom path (not any env default) must be passed to from_authorized_user_file
        path_arg, _ = mock_creds_class.from_authorized_user_file.call_args.args
        self.assertEqual(path_arg, custom_token)

    def test_env_var_token_path_used_when_no_explicit_arg(self):
        """When token_file is None, the GOOGLE_CALENDAR_TOKEN_FILE env var must be used."""
        import sys
        import tempfile
        import os

        sys_modules_patch, mock_creds_class, _, _ = self._make_google_mocks()

        with tempfile.TemporaryDirectory() as tmp:
            env_token = os.path.join(tmp, "env_token.json")
            with open(env_token, "w") as fh:
                fh.write('{"token": "fake"}')
            env_creds = os.path.join(tmp, "env_credentials.json")
            with open(env_creds, "w") as fh:
                fh.write("{}")

            env_patch = {
                "GOOGLE_CALENDAR_TOKEN_FILE": env_token,
                "GOOGLE_CALENDAR_CREDENTIALS_FILE": env_creds,
            }
            with patch.dict(sys.modules, sys_modules_patch), \
                 patch.dict(os.environ, env_patch):
                build_calendar_service()  # no explicit token_file

        path_arg, _ = mock_creds_class.from_authorized_user_file.call_args.args
        self.assertEqual(path_arg, env_token)

    # ------------------------------------------------------------------
    # Service initialisation verification
    # ------------------------------------------------------------------

    def test_returned_service_is_from_build_not_a_new_mock(self):
        """The exact object returned by build() must be the function's return value."""
        import sys
        import tempfile

        sys_modules_patch, _, _, mock_build = self._make_google_mocks()
        # Give the mock service a recognisable identity
        sentinel_service = MagicMock(name="SentinelCalendarService")
        mock_build.return_value = sentinel_service

        with tempfile.TemporaryDirectory() as tmp:
            result = self._call_with_token_file(tmp, sys_modules_patch)

        self.assertIs(result, sentinel_service)

    def test_build_called_exactly_once(self):
        """googleapiclient.discovery.build must be called exactly once per invocation."""
        import sys
        import tempfile

        sys_modules_patch, _, _, mock_build = self._make_google_mocks()

        with tempfile.TemporaryDirectory() as tmp:
            self._call_with_token_file(tmp, sys_modules_patch)

        mock_build.assert_called_once()


# ---------------------------------------------------------------------------
# Tests for update_calendar_event
# ---------------------------------------------------------------------------

class TestUpdateCalendarEvent(unittest.TestCase):
    """Unit tests for update_calendar_event().

    All tests use a mocked Google Calendar API client (patch endpoint) so
    they run fully offline without real credentials.

    Covers:
    - Returns the updated event parsed from the API response
    - Partial-update payload only includes provided fields
    - calendarId / eventId forwarded correctly to the API
    - ValueError when no fields are provided to update
    - Exponential backoff retry on transient API errors
    - RuntimeError raised after all retries exhausted
    """

    @staticmethod
    def _make_patch_service(response: dict) -> MagicMock:
        """Return a mock service whose patch().execute() returns *response*."""
        service = MagicMock()
        service.events.return_value.patch.return_value.execute = MagicMock(
            return_value=response
        )
        return service

    @staticmethod
    def _make_patch_service_with_errors(
        errors: list[Exception],
        success_response: dict | None = None,
    ) -> MagicMock:
        """Return a mock service that raises *errors* in sequence, then succeeds."""
        service = MagicMock()
        side_effects: list = list(errors)
        if success_response is not None:
            side_effects.append(success_response)
        service.events.return_value.patch.return_value.execute = MagicMock(
            side_effect=side_effects
        )
        return service

    # ------------------------------------------------------------------
    # Return value: updated CalendarEvent
    # ------------------------------------------------------------------

    def test_returns_calendar_event_from_response(self):
        """update_calendar_event must return a CalendarEvent parsed from the API response."""
        service = self._make_patch_service(
            {
                "id": "evt_001",
                "summary": "Team Standup (Updated)",
                "start": {"dateTime": "2026-06-10T10:00:00+09:00"},
                "end": {"dateTime": "2026-06-10T10:30:00+09:00"},
            }
        )

        event = update_calendar_event(
            "evt_001", title="Team Standup (Updated)", service=service
        )

        self.assertIsInstance(event, CalendarEvent)
        self.assertEqual(event.calendar_event_id, "evt_001")
        self.assertEqual(event.calendar_event_title, "Team Standup (Updated)")

    def test_returned_time_matches_api_response_exactly(self):
        """Factual fields (time) must be passed through exactly as returned by the API."""
        service = self._make_patch_service(
            {
                "id": "evt_001",
                "summary": "Moved Meeting",
                "start": {"dateTime": "2026-06-12T14:00:00+09:00"},
                "end": {"dateTime": "2026-06-12T15:00:00+09:00"},
            }
        )

        event = update_calendar_event(
            "evt_001", start_time="2026-06-12T14:00:00+09:00", service=service
        )

        self.assertEqual(event.calendar_event_time, "2026-06-12T14:00:00+09:00")
        self.assertEqual(event.end_time, "2026-06-12T15:00:00+09:00")

    # ------------------------------------------------------------------
    # Partial-update payload construction
    # ------------------------------------------------------------------

    def test_payload_contains_only_title_when_only_title_provided(self):
        """When only title is given, the payload must contain only 'summary'."""
        service = self._make_patch_service({"id": "evt_001", "summary": "New Title"})

        update_calendar_event("evt_001", title="New Title", service=service)

        body = service.events.return_value.patch.call_args.kwargs["body"]
        self.assertEqual(body, {"summary": "New Title"})

    def test_payload_contains_only_start_and_end_when_times_provided(self):
        """When only start/end times are given, the payload must contain only those."""
        service = self._make_patch_service(
            {
                "id": "evt_001",
                "start": {"dateTime": "2026-06-10T09:00:00Z"},
                "end": {"dateTime": "2026-06-10T10:00:00Z"},
            }
        )

        update_calendar_event(
            "evt_001",
            start_time="2026-06-10T09:00:00Z",
            end_time="2026-06-10T10:00:00Z",
            service=service,
        )

        body = service.events.return_value.patch.call_args.kwargs["body"]
        self.assertEqual(
            body,
            {
                "start": {"dateTime": "2026-06-10T09:00:00Z"},
                "end": {"dateTime": "2026-06-10T10:00:00Z"},
            },
        )

    def test_payload_contains_description_when_provided(self):
        """When description is given, the payload must contain 'description'."""
        service = self._make_patch_service({"id": "evt_001"})

        update_calendar_event(
            "evt_001", description="Updated description", service=service
        )

        body = service.events.return_value.patch.call_args.kwargs["body"]
        self.assertEqual(body, {"description": "Updated description"})

    def test_payload_contains_all_fields_when_all_provided(self):
        """When every field is given, the payload must contain all of them."""
        service = self._make_patch_service({"id": "evt_001"})

        update_calendar_event(
            "evt_001",
            title="New Title",
            start_time="2026-06-10T09:00:00Z",
            end_time="2026-06-10T10:00:00Z",
            description="New description",
            service=service,
        )

        body = service.events.return_value.patch.call_args.kwargs["body"]
        self.assertEqual(
            body,
            {
                "summary": "New Title",
                "description": "New description",
                "start": {"dateTime": "2026-06-10T09:00:00Z"},
                "end": {"dateTime": "2026-06-10T10:00:00Z"},
            },
        )

    # ------------------------------------------------------------------
    # eventId / calendarId forwarding
    # ------------------------------------------------------------------

    def test_event_id_forwarded_to_api(self):
        """The eventId must be forwarded to the patch() call."""
        service = self._make_patch_service({"id": "evt_xyz"})

        update_calendar_event("evt_xyz", title="X", service=service)

        call_kwargs = service.events.return_value.patch.call_args.kwargs
        self.assertEqual(call_kwargs["eventId"], "evt_xyz")

    def test_uses_primary_calendar_by_default(self):
        """calendarId must default to 'primary' when not specified."""
        service = self._make_patch_service({"id": "evt_001"})

        update_calendar_event("evt_001", title="X", service=service)

        call_kwargs = service.events.return_value.patch.call_args.kwargs
        self.assertEqual(call_kwargs["calendarId"], "primary")

    def test_custom_calendar_id_forwarded_to_api(self):
        """A custom calendar_id must be forwarded to the patch() call."""
        service = self._make_patch_service({"id": "evt_001"})

        update_calendar_event(
            "evt_001",
            title="X",
            service=service,
            calendar_id="work@group.calendar.google.com",
        )

        call_kwargs = service.events.return_value.patch.call_args.kwargs
        self.assertEqual(call_kwargs["calendarId"], "work@group.calendar.google.com")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def test_raises_value_error_when_no_fields_provided(self):
        """update_calendar_event must raise ValueError if nothing is being updated."""
        service = self._make_patch_service({"id": "evt_001"})

        with self.assertRaises(ValueError):
            update_calendar_event("evt_001", service=service)

        service.events.return_value.patch.assert_not_called()

    # ------------------------------------------------------------------
    # Exponential backoff and retry
    # ------------------------------------------------------------------

    @patch("time.sleep", return_value=None)
    def test_retries_once_on_transient_error_and_returns_event(self, mock_sleep):
        """Should retry after a transient error and return the event on success."""
        service = self._make_patch_service_with_errors(
            [Exception("503 Service Unavailable")],
            success_response={"id": "evt_001", "summary": "Retried Title"},
        )

        event = update_calendar_event(
            "evt_001",
            title="Retried Title",
            service=service,
            max_retries=2,
            base_backoff=1.0,
        )

        self.assertEqual(event.calendar_event_title, "Retried Title")
        mock_sleep.assert_called_once_with(1.0)

    @patch("time.sleep", return_value=None)
    def test_second_retry_uses_doubled_backoff(self, mock_sleep):
        """Second retry backoff must be 2x the first (exponential)."""
        service = self._make_patch_service_with_errors(
            [Exception("Timeout"), Exception("Timeout")],
            success_response={"id": "evt_001", "summary": "OK"},
        )

        update_calendar_event(
            "evt_001",
            title="OK",
            service=service,
            max_retries=2,
            base_backoff=1.0,
        )

        self.assertEqual(mock_sleep.call_count, 2)
        calls = mock_sleep.call_args_list
        self.assertAlmostEqual(calls[0][0][0], 1.0)
        self.assertAlmostEqual(calls[1][0][0], 2.0)

    @patch("time.sleep", return_value=None)
    def test_raises_runtime_error_after_max_retries_exhausted(self, mock_sleep):
        """Must raise RuntimeError with an explicit message after all attempts fail."""
        service = MagicMock()
        service.events.return_value.patch.return_value.execute = MagicMock(
            side_effect=Exception("Connection refused")
        )

        with self.assertRaises(RuntimeError) as ctx:
            update_calendar_event(
                "evt_001",
                title="X",
                service=service,
                max_retries=2,
                base_backoff=0.01,
            )

        error_msg = str(ctx.exception)
        self.assertIn("Unable to update calendar event", error_msg)
        self.assertIn("evt_001", error_msg)
        self.assertIn("3 attempts", error_msg)
        self.assertIn("Connection refused", error_msg)

    @patch("time.sleep", return_value=None)
    def test_api_called_exactly_max_retries_plus_one_times(self, mock_sleep):
        """The patch API must be called exactly (max_retries + 1) times before giving up."""
        service = MagicMock()
        service.events.return_value.patch.return_value.execute = MagicMock(
            side_effect=Exception("always fails")
        )

        with self.assertRaises(RuntimeError):
            update_calendar_event(
                "evt_001",
                title="X",
                service=service,
                max_retries=2,
                base_backoff=0.01,
            )

        self.assertEqual(
            service.events.return_value.patch.return_value.execute.call_count,
            3,
        )


# ---------------------------------------------------------------------------
# Tests for delete_calendar_event
# ---------------------------------------------------------------------------

class TestDeleteCalendarEvent(unittest.TestCase):
    """Unit tests for delete_calendar_event().

    All tests use a mocked Google Calendar API client (delete endpoint) so
    they run fully offline without real credentials.

    Covers:
    - Returns True on success
    - eventId / calendarId forwarded correctly to the API
    - Exponential backoff retry on transient API errors
    - RuntimeError raised after all retries exhausted
    """

    @staticmethod
    def _make_delete_service(response: Any = "") -> MagicMock:
        """Return a mock service whose delete().execute() returns *response*."""
        service = MagicMock()
        service.events.return_value.delete.return_value.execute = MagicMock(
            return_value=response
        )
        return service

    @staticmethod
    def _make_delete_service_with_errors(
        errors: list[Exception],
        success_response: Any = "",
    ) -> MagicMock:
        """Return a mock service that raises *errors* in sequence, then succeeds."""
        service = MagicMock()
        side_effects: list = list(errors)
        side_effects.append(success_response)
        service.events.return_value.delete.return_value.execute = MagicMock(
            side_effect=side_effects
        )
        return service

    # ------------------------------------------------------------------
    # Return value
    # ------------------------------------------------------------------

    def test_returns_true_on_success(self):
        """delete_calendar_event must return True once the event is deleted."""
        service = self._make_delete_service()

        result = delete_calendar_event("evt_001", service=service)

        self.assertTrue(result)

    # ------------------------------------------------------------------
    # eventId / calendarId forwarding
    # ------------------------------------------------------------------

    def test_event_id_forwarded_to_api(self):
        """The eventId must be forwarded to the delete() call."""
        service = self._make_delete_service()

        delete_calendar_event("evt_xyz", service=service)

        call_kwargs = service.events.return_value.delete.call_args.kwargs
        self.assertEqual(call_kwargs["eventId"], "evt_xyz")

    def test_uses_primary_calendar_by_default(self):
        """calendarId must default to 'primary' when not specified."""
        service = self._make_delete_service()

        delete_calendar_event("evt_001", service=service)

        call_kwargs = service.events.return_value.delete.call_args.kwargs
        self.assertEqual(call_kwargs["calendarId"], "primary")

    def test_custom_calendar_id_forwarded_to_api(self):
        """A custom calendar_id must be forwarded to the delete() call."""
        service = self._make_delete_service()

        delete_calendar_event(
            "evt_001", service=service, calendar_id="work@group.calendar.google.com"
        )

        call_kwargs = service.events.return_value.delete.call_args.kwargs
        self.assertEqual(call_kwargs["calendarId"], "work@group.calendar.google.com")

    # ------------------------------------------------------------------
    # Exponential backoff and retry
    # ------------------------------------------------------------------

    @patch("time.sleep", return_value=None)
    def test_retries_once_on_transient_error_and_returns_true(self, mock_sleep):
        """Should retry after a transient error and return True on success."""
        service = self._make_delete_service_with_errors(
            [Exception("503 Service Unavailable")]
        )

        result = delete_calendar_event(
            "evt_001", service=service, max_retries=2, base_backoff=1.0
        )

        self.assertTrue(result)
        mock_sleep.assert_called_once_with(1.0)

    @patch("time.sleep", return_value=None)
    def test_second_retry_uses_doubled_backoff(self, mock_sleep):
        """Second retry backoff must be 2x the first (exponential)."""
        service = self._make_delete_service_with_errors(
            [Exception("Timeout"), Exception("Timeout")]
        )

        delete_calendar_event(
            "evt_001", service=service, max_retries=2, base_backoff=1.0
        )

        self.assertEqual(mock_sleep.call_count, 2)
        calls = mock_sleep.call_args_list
        self.assertAlmostEqual(calls[0][0][0], 1.0)
        self.assertAlmostEqual(calls[1][0][0], 2.0)

    @patch("time.sleep", return_value=None)
    def test_raises_runtime_error_after_max_retries_exhausted(self, mock_sleep):
        """Must raise RuntimeError with an explicit message after all attempts fail."""
        service = MagicMock()
        service.events.return_value.delete.return_value.execute = MagicMock(
            side_effect=Exception("Connection refused")
        )

        with self.assertRaises(RuntimeError) as ctx:
            delete_calendar_event(
                "evt_001", service=service, max_retries=2, base_backoff=0.01
            )

        error_msg = str(ctx.exception)
        self.assertIn("Unable to delete calendar event", error_msg)
        self.assertIn("evt_001", error_msg)
        self.assertIn("3 attempts", error_msg)
        self.assertIn("Connection refused", error_msg)

    @patch("time.sleep", return_value=None)
    def test_api_called_exactly_max_retries_plus_one_times(self, mock_sleep):
        """The delete API must be called exactly (max_retries + 1) times before giving up."""
        service = MagicMock()
        service.events.return_value.delete.return_value.execute = MagicMock(
            side_effect=Exception("always fails")
        )

        with self.assertRaises(RuntimeError):
            delete_calendar_event(
                "evt_001", service=service, max_retries=2, base_backoff=0.01
            )

        self.assertEqual(
            service.events.return_value.delete.return_value.execute.call_count,
            3,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
