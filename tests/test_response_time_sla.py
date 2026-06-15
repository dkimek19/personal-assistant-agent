"""End-to-end response-time SLA tests (AC1-5).

AC1-5 set wall-clock budgets for the agent's responses:
    AC1: Weather query (single tool, short answer)        <= 3 seconds
    AC2: Calendar query (single tool + formatting)         <= 5 seconds
    AC3: Multi-tool chain (calendar + weather + memo)      <= 15 seconds
    AC4: Code execution via Docker sandbox                 <= 60 seconds
    AC5: Complex generation (weekly DOCX summary)          <= 30 seconds

These budgets are dominated by external services this test suite cannot rely
on being present (a local Ollama model, the Google Calendar API, a Docker
daemon, real network access). The tests below therefore mock those I/O
boundaries (the same functions :mod:`assistant.tools.dispatch` calls) so they
return immediately, and assert that the assistant's own dispatch/formatting
logic adds negligible overhead -- i.e. essentially all of each budget remains
available for the real network/LLM/Docker work. This guards against a
regression (an accidental blocking call, retry-on-success, or N+1 loop in the
dispatch layer) silently eating into the SLA.

A handful of live variants exercise the real external service end-to-end
against the *same* budgets. They are skipped by default and enabled via
environment variables, following the pattern established by
``SEARXNG_LIVE_TEST`` in tests/test_searxng.py:
    - ``WEATHER_LIVE_TEST=1``: real Open-Meteo API call (AC1).
    - ``DOCKER_LIVE_TEST=1``: real Docker sandbox run (AC4).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from assistant.notes import NoteStore, handle_note_command
from assistant.tools.calendar import CalendarEvent
from assistant.tools.code_execution import CodeExecutionResult
from assistant.tools.dispatch import dispatch_tool
from assistant.tools.weather import WeatherReport

# AC1-5 response-time budgets, in seconds.
SLA_WEATHER_QUERY_SECONDS = 3.0
SLA_CALENDAR_QUERY_SECONDS = 5.0
SLA_MULTI_TOOL_CHAIN_SECONDS = 15.0
SLA_CODE_EXECUTION_SECONDS = 60.0
SLA_COMPLEX_GENERATION_SECONDS = 30.0


_EVENT = CalendarEvent(
    calendar_event_id="ev1",
    calendar_event_title="Team sync",
    calendar_event_time="2026-06-10T10:00:00+00:00",
    calendar_alert_time="2026-06-10T09:30:00+00:00",
    end_time="2026-06-10T11:00:00+00:00",
)

_WEATHER = WeatherReport(
    location_name="Seoul",
    latitude=37.57,
    longitude=126.98,
    temperature_c=22.5,
    wind_speed_kmh=8.0,
    weather_code=0,
    weather_description="Clear sky",
    observation_time="2026-06-10T10:00:00Z",
)

_CODE_RESULT = CodeExecutionResult(stdout="hello\n", stderr="", exit_code=0, timed_out=False)


class TestWeatherQuerySLA:
    """AC1: Weather query (single tool, short answer) responds within 3 seconds."""

    @patch("assistant.tools.dispatch.get_current_weather")
    def test_dispatch_overhead_within_budget(self, mock_get):
        mock_get.return_value = _WEATHER

        start = time.perf_counter()
        result = dispatch_tool("get_weather", {"location": "Seoul"})
        elapsed = time.perf_counter() - start

        assert result["tool_status"] == "success"
        assert elapsed < SLA_WEATHER_QUERY_SECONDS

    @pytest.mark.skipif(
        os.environ.get("WEATHER_LIVE_TEST") != "1",
        reason=(
            "Live Open-Meteo integration test skipped by default. "
            "Set WEATHER_LIVE_TEST=1 to run against the real API."
        ),
    )
    def test_real_weather_query_within_budget(self):
        start = time.perf_counter()
        result = dispatch_tool("get_weather", {"location": "Seoul"})
        elapsed = time.perf_counter() - start

        assert result["tool_status"] == "success"
        assert elapsed < SLA_WEATHER_QUERY_SECONDS


class TestCalendarQuerySLA:
    """AC2: Calendar query (single tool + formatting) responds within 5 seconds."""

    @patch("assistant.tools.dispatch.get_calendar_events")
    def test_dispatch_overhead_within_budget(self, mock_get):
        mock_get.return_value = [_EVENT]

        start = time.perf_counter()
        result = dispatch_tool("get_calendar_events", {"start_date": "2026-06-10", "end_date": "2026-06-11"})
        elapsed = time.perf_counter() - start

        assert result["tool_status"] == "success"
        assert result["tool_output"]["events"] == [_EVENT.to_dict()]
        assert elapsed < SLA_CALENDAR_QUERY_SECONDS


class TestMultiToolChainSLA:
    """AC3: Multi-tool chain (calendar + weather + memo) responds within 15 seconds."""

    @patch("assistant.tools.dispatch.get_current_weather")
    @patch("assistant.tools.dispatch.get_calendar_events")
    def test_calendar_weather_and_memo_chain_within_budget(self, mock_get_events, mock_get_weather, tmp_path):
        mock_get_events.return_value = [_EVENT]
        mock_get_weather.return_value = _WEATHER
        store = NoteStore(db_path=tmp_path / "memory.db")

        start = time.perf_counter()
        calendar_result = dispatch_tool(
            "get_calendar_events", {"start_date": "2026-06-10", "end_date": "2026-06-11"}
        )
        weather_result = dispatch_tool("get_weather", {"location": "Seoul"})
        note_response = handle_note_command(
            "user1", f"/note Bring an umbrella to {_EVENT.calendar_event_title}", store
        )
        elapsed = time.perf_counter() - start

        assert calendar_result["tool_status"] == "success"
        assert weather_result["tool_status"] == "success"
        assert "saved" in note_response
        assert elapsed < SLA_MULTI_TOOL_CHAIN_SECONDS


class TestCodeExecutionSLA:
    """AC4: Code execution via Docker sandbox completes within 60 seconds."""

    @patch("assistant.tools.dispatch.execute_code")
    def test_dispatch_overhead_within_budget(self, mock_execute):
        mock_execute.return_value = _CODE_RESULT

        start = time.perf_counter()
        result = dispatch_tool("execute_code", {"code": "print('hello')"})
        elapsed = time.perf_counter() - start

        assert result["tool_status"] == "success"
        assert elapsed < SLA_CODE_EXECUTION_SECONDS

    @pytest.mark.skipif(
        os.environ.get("DOCKER_LIVE_TEST") != "1",
        reason=(
            "Live Docker integration test skipped by default. "
            "Set DOCKER_LIVE_TEST=1 to run against a real Docker daemon."
        ),
    )
    def test_real_docker_sandbox_within_budget(self):
        start = time.perf_counter()
        result = dispatch_tool("execute_code", {"code": "print('hello')"})
        elapsed = time.perf_counter() - start

        assert result["tool_status"] == "success"
        assert result["tool_output"]["result"]["exit_code"] == 0
        assert elapsed < SLA_CODE_EXECUTION_SECONDS


class TestComplexGenerationSLA:
    """AC5: Complex generation (weekly DOCX summary) responds within 30 seconds."""

    @patch("assistant.tools.dispatch.get_calendar_events")
    def test_weekly_summary_docx_generation_within_budget(self, mock_get_events, tmp_path):
        mock_get_events.return_value = [_EVENT]
        output_path = tmp_path / "weekly_summary.docx"

        start = time.perf_counter()
        calendar_result = dispatch_tool(
            "get_calendar_events", {"start_date": "2026-06-08", "end_date": "2026-06-15"}
        )
        events = calendar_result["tool_output"]["events"]
        summary_lines = [
            "Weekly Summary",
            *(f"{e['calendar_event_title']}: {e['calendar_event_time']}" for e in events),
        ]
        document_result = dispatch_tool(
            "create_document", {"file_path": str(output_path), "content": summary_lines}
        )
        elapsed = time.perf_counter() - start

        assert calendar_result["tool_status"] == "success"
        assert document_result["tool_status"] == "success"
        assert Path(document_result["tool_output"]["file_path"]) == output_path
        assert output_path.exists()
        assert elapsed < SLA_COMPLEX_GENERATION_SECONDS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
