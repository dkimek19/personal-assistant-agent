"""Tests for the agent's tool-calling dispatch pipeline (assistant.tools.dispatch).

Covers:
- dispatch_tool("web_search", ...) selects and invokes the SearXNG pipeline:
  search -> filter_results, in order, with correct data passthrough
  (data-fidelity), and no LLM summarization round-trip
- "searxng" is a recognised alias for the same handler
- num_results / top_n are forwarded correctly, with top_n defaulting to
  num_results
- Result shape: tool_name, tool_input, tool_output, tool_status,
  tool_retry_count, tool_error_message
- Error handling: empty/missing query, ToolError from search() — all
  reported via tool_status="failed" rather than raising
- Unknown tool names are reported via tool_status="failed"
- Calendar, Tasks, Weather, Document, and Code Execution tools (AC23):
  successful dispatch with correct data passthrough, and unified error
  handling -- ValueError (invalid input / non-retryable domain errors),
  FileNotFoundError, RuntimeError after exhausted retries, and unexpected
  exceptions all normalise to tool_status="failed" without dispatch_tool
  raising.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from assistant.tools.calendar import CalendarEvent
from assistant.tools.code_execution import CodeExecutionResult
from assistant.tools.dispatch import _MAX_RETRY_COUNT, dispatch_tool
from assistant.tools.documents import DocumentError
from assistant.tools.searxng import ToolError
from assistant.tools.tasks import TaskItem
from assistant.tools.weather import LocationNotFoundError, WeatherReport

RAW_RESULTS = [
    {"url": "https://example.com/a", "title": "A", "snippet": "Snippet A"},
    {"url": "https://example.com/b", "title": "B", "snippet": "Snippet B"},
]

FILTERED_RESULTS = [
    {**RAW_RESULTS[0], "_relevance_score": 2.0},
]


# ---------------------------------------------------------------------------
# Successful web_search dispatch
# ---------------------------------------------------------------------------


class TestDispatchWebSearchSuccess:
    @patch("assistant.tools.dispatch.filter_results")
    @patch("assistant.tools.dispatch.search")
    def test_pipeline_called_in_order_with_correct_args(self, mock_search, mock_filter):
        mock_search.return_value = RAW_RESULTS
        mock_filter.return_value = FILTERED_RESULTS

        result = dispatch_tool("web_search", {"query": "best result"})

        mock_search.assert_called_once_with("best result", num_results=5)
        mock_filter.assert_called_once_with(RAW_RESULTS, "best result", 5)
        assert result["tool_status"] == "success"

    @patch("assistant.tools.dispatch.filter_results")
    @patch("assistant.tools.dispatch.search")
    def test_result_shape_has_all_ontology_keys(self, mock_search, mock_filter):
        mock_search.return_value = RAW_RESULTS
        mock_filter.return_value = FILTERED_RESULTS

        result = dispatch_tool("web_search", {"query": "q"})

        assert set(result.keys()) == {
            "tool_name",
            "tool_input",
            "tool_output",
            "tool_status",
            "tool_retry_count",
            "tool_error_message",
        }

    @patch("assistant.tools.dispatch.filter_results")
    @patch("assistant.tools.dispatch.search")
    def test_tool_output_contains_only_results(self, mock_search, mock_filter):
        mock_search.return_value = RAW_RESULTS
        mock_filter.return_value = FILTERED_RESULTS

        result = dispatch_tool("web_search", {"query": "q"})

        assert set(result["tool_output"].keys()) == {"results"}
        assert result["tool_output"]["results"] == FILTERED_RESULTS

    @patch("assistant.tools.dispatch.filter_results")
    @patch("assistant.tools.dispatch.search")
    def test_results_passthrough_is_unmodified(self, mock_search, mock_filter):
        mock_search.return_value = RAW_RESULTS
        mock_filter.return_value = FILTERED_RESULTS

        result = dispatch_tool("web_search", {"query": "q"})

        assert result["tool_output"]["results"] is FILTERED_RESULTS

    @patch("assistant.tools.dispatch.filter_results")
    @patch("assistant.tools.dispatch.search")
    def test_echoes_tool_name_and_input(self, mock_search, mock_filter):
        mock_search.return_value = RAW_RESULTS
        mock_filter.return_value = FILTERED_RESULTS

        tool_input = {"query": "q"}
        result = dispatch_tool("web_search", tool_input)

        assert result["tool_name"] == "web_search"
        assert result["tool_input"] == tool_input

    @patch("assistant.tools.dispatch.filter_results")
    @patch("assistant.tools.dispatch.search")
    def test_success_has_zero_retry_count_and_no_error_message(self, mock_search, mock_filter):
        mock_search.return_value = RAW_RESULTS
        mock_filter.return_value = FILTERED_RESULTS

        result = dispatch_tool("web_search", {"query": "q"})

        assert result["tool_retry_count"] == 0
        assert result["tool_error_message"] is None

    @patch("assistant.tools.dispatch.filter_results")
    @patch("assistant.tools.dispatch.search")
    def test_searxng_alias_invokes_same_pipeline(self, mock_search, mock_filter):
        mock_search.return_value = RAW_RESULTS
        mock_filter.return_value = FILTERED_RESULTS

        result = dispatch_tool("searxng", {"query": "q"})

        assert result["tool_status"] == "success"
        mock_search.assert_called_once()

    @patch("assistant.tools.dispatch.filter_results")
    @patch("assistant.tools.dispatch.search")
    def test_num_results_forwarded_to_search(self, mock_search, mock_filter):
        mock_search.return_value = RAW_RESULTS
        mock_filter.return_value = FILTERED_RESULTS

        dispatch_tool("web_search", {"query": "q", "num_results": 10})

        mock_search.assert_called_once_with("q", num_results=10)

    @patch("assistant.tools.dispatch.filter_results")
    @patch("assistant.tools.dispatch.search")
    def test_top_n_defaults_to_num_results(self, mock_search, mock_filter):
        mock_search.return_value = RAW_RESULTS
        mock_filter.return_value = FILTERED_RESULTS

        dispatch_tool("web_search", {"query": "q", "num_results": 8})

        mock_filter.assert_called_once_with(RAW_RESULTS, "q", 8)

    @patch("assistant.tools.dispatch.filter_results")
    @patch("assistant.tools.dispatch.search")
    def test_top_n_overrides_num_results_for_filtering(self, mock_search, mock_filter):
        mock_search.return_value = RAW_RESULTS
        mock_filter.return_value = FILTERED_RESULTS

        dispatch_tool("web_search", {"query": "q", "num_results": 10, "top_n": 3})

        mock_search.assert_called_once_with("q", num_results=10)
        mock_filter.assert_called_once_with(RAW_RESULTS, "q", 3)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestDispatchWebSearchErrors:
    def test_empty_query_returns_failed_status(self):
        result = dispatch_tool("web_search", {"query": ""})

        assert result["tool_status"] == "failed"
        assert result["tool_output"] == {}
        assert result["tool_retry_count"] == 0
        assert "query" in result["tool_error_message"]

    def test_missing_query_key_returns_failed_status(self):
        result = dispatch_tool("web_search", {})

        assert result["tool_status"] == "failed"
        assert "query" in result["tool_error_message"]

    @patch("assistant.tools.dispatch.search")
    def test_search_tool_error_returns_failed_with_max_retry_count(self, mock_search):
        mock_search.side_effect = ToolError("Web search failed after 3 attempts")

        result = dispatch_tool("web_search", {"query": "q"})

        assert result["tool_status"] == "failed"
        assert result["tool_output"] == {}
        assert result["tool_retry_count"] == _MAX_RETRY_COUNT
        assert "Web search failed" in result["tool_error_message"]


# ---------------------------------------------------------------------------
# Unknown tool
# ---------------------------------------------------------------------------


class TestDispatchUnknownTool:
    def test_unknown_tool_returns_failed_status(self):
        result = dispatch_tool("not_a_real_tool", {})

        assert result["tool_status"] == "failed"
        assert result["tool_output"] == {}
        assert result["tool_retry_count"] == 0

    def test_unknown_tool_error_message_mentions_tool_name(self):
        result = dispatch_tool("not_a_real_tool", {})

        assert "not_a_real_tool" in result["tool_error_message"]

    def test_unknown_tool_echoes_name_and_input(self):
        tool_input = {"foo": "bar"}
        result = dispatch_tool("not_a_real_tool", tool_input)

        assert result["tool_name"] == "not_a_real_tool"
        assert result["tool_input"] == tool_input


# ---------------------------------------------------------------------------
# Calendar tools (AC23)
# ---------------------------------------------------------------------------

_EVENT = CalendarEvent(
    calendar_event_id="ev1",
    calendar_event_title="Meeting",
    calendar_event_time="2026-06-10T10:00:00+00:00",
    calendar_alert_time="2026-06-10T09:30:00+00:00",
    end_time="2026-06-10T11:00:00+00:00",
)


class TestDispatchGetCalendarEvents:
    @patch("assistant.tools.dispatch.get_calendar_events")
    def test_returns_events_as_dicts(self, mock_get):
        mock_get.return_value = [_EVENT]

        result = dispatch_tool("get_calendar_events", {"start_date": "2026-06-10", "end_date": "2026-06-11"})

        mock_get.assert_called_once_with("2026-06-10", "2026-06-11")
        assert result["tool_status"] == "success"
        assert result["tool_output"]["events"] == [_EVENT.to_dict()]

    def test_missing_start_date_returns_failed_status(self):
        result = dispatch_tool("get_calendar_events", {"end_date": "2026-06-11"})

        assert result["tool_status"] == "failed"
        assert result["tool_retry_count"] == 0
        assert "start_date" in result["tool_error_message"]

    @patch("assistant.tools.dispatch.get_calendar_events")
    def test_runtime_error_after_retries_returns_max_retry_count(self, mock_get):
        mock_get.side_effect = RuntimeError("Unable to retrieve calendar events after 3 attempts.")

        result = dispatch_tool("get_calendar_events", {"start_date": "2026-06-10", "end_date": "2026-06-11"})

        assert result["tool_status"] == "failed"
        assert result["tool_retry_count"] == _MAX_RETRY_COUNT
        assert "Unable to retrieve calendar events" in result["tool_error_message"]


class TestDispatchCreateCalendarEvent:
    @patch("assistant.tools.dispatch.create_calendar_event")
    def test_creates_event_and_returns_id(self, mock_create):
        mock_create.return_value = "new-event-id"

        result = dispatch_tool(
            "create_calendar_event",
            {
                "title": "Meeting",
                "start_time": "2026-06-10T10:00:00Z",
                "end_time": "2026-06-10T11:00:00Z",
                "description": "Discuss roadmap",
            },
        )

        mock_create.assert_called_once_with(
            "Meeting", "2026-06-10T10:00:00Z", "2026-06-10T11:00:00Z", "Discuss roadmap"
        )
        assert result["tool_status"] == "success"
        assert result["tool_output"] == {"event_id": "new-event-id"}

    def test_missing_required_field_returns_failed_status(self):
        result = dispatch_tool("create_calendar_event", {"title": "Meeting", "start_time": "2026-06-10T10:00:00Z"})

        assert result["tool_status"] == "failed"
        assert result["tool_retry_count"] == 0


class TestDispatchUpdateCalendarEvent:
    @patch("assistant.tools.dispatch.update_calendar_event")
    def test_updates_event_and_returns_event_dict(self, mock_update):
        mock_update.return_value = _EVENT

        result = dispatch_tool("update_calendar_event", {"event_id": "ev1", "title": "Renamed"})

        mock_update.assert_called_once_with(
            "ev1", title="Renamed", start_time=None, end_time=None, description=None
        )
        assert result["tool_status"] == "success"
        assert result["tool_output"] == {"event": _EVENT.to_dict()}

    def test_missing_event_id_returns_failed_status(self):
        result = dispatch_tool("update_calendar_event", {"title": "Renamed"})

        assert result["tool_status"] == "failed"
        assert result["tool_retry_count"] == 0
        assert "event_id" in result["tool_error_message"]

    @patch("assistant.tools.dispatch.update_calendar_event")
    def test_no_update_fields_value_error_returns_failed_status(self, mock_update):
        mock_update.side_effect = ValueError(
            "update_calendar_event requires at least one of: title, start_time, end_time, description"
        )

        result = dispatch_tool("update_calendar_event", {"event_id": "ev1"})

        assert result["tool_status"] == "failed"
        assert result["tool_retry_count"] == 0


class TestDispatchDeleteCalendarEvent:
    @patch("assistant.tools.dispatch.delete_calendar_event")
    def test_deletes_event(self, mock_delete):
        mock_delete.return_value = True

        result = dispatch_tool("delete_calendar_event", {"event_id": "ev1"})

        mock_delete.assert_called_once_with("ev1")
        assert result["tool_status"] == "success"
        assert result["tool_output"] == {"deleted": True}

    def test_missing_event_id_returns_failed_status(self):
        result = dispatch_tool("delete_calendar_event", {})

        assert result["tool_status"] == "failed"
        assert result["tool_retry_count"] == 0


# ---------------------------------------------------------------------------
# Tasks tools (AC23)
# ---------------------------------------------------------------------------

_TASK = TaskItem(task_id="t1", task_title="Buy milk")


class TestDispatchGetTasks:
    @patch("assistant.tools.dispatch.get_tasks")
    def test_returns_tasks_as_dicts(self, mock_get):
        mock_get.return_value = [_TASK]

        result = dispatch_tool("get_tasks", {})

        mock_get.assert_called_once_with(tasklist_id="@default", show_completed=False)
        assert result["tool_status"] == "success"
        assert result["tool_output"]["tasks"] == [_TASK.to_dict()]

    @patch("assistant.tools.dispatch.get_tasks")
    def test_runtime_error_after_retries_returns_max_retry_count(self, mock_get):
        mock_get.side_effect = RuntimeError("Unable to retrieve tasks after 3 attempts.")

        result = dispatch_tool("get_tasks", {})

        assert result["tool_status"] == "failed"
        assert result["tool_retry_count"] == _MAX_RETRY_COUNT


class TestDispatchCreateTask:
    @patch("assistant.tools.dispatch.create_task")
    def test_creates_task_and_returns_id(self, mock_create):
        mock_create.return_value = "new-task-id"

        result = dispatch_tool("create_task", {"title": "Buy milk", "due_date": "2026-06-11", "notes": "2%"})

        mock_create.assert_called_once_with("Buy milk", "2026-06-11", "2%", tasklist_id="@default")
        assert result["tool_status"] == "success"
        assert result["tool_output"] == {"task_id": "new-task-id"}

    def test_empty_title_returns_failed_status(self):
        result = dispatch_tool("create_task", {"title": "   "})

        assert result["tool_status"] == "failed"
        assert result["tool_retry_count"] == 0


class TestDispatchUpdateTask:
    @patch("assistant.tools.dispatch.update_task")
    def test_updates_task_and_returns_task_dict(self, mock_update):
        mock_update.return_value = _TASK

        result = dispatch_tool("update_task", {"task_id": "t1", "title": "Buy oat milk"})

        mock_update.assert_called_once_with(
            "t1", title="Buy oat milk", due_date=None, notes=None, tasklist_id="@default"
        )
        assert result["tool_status"] == "success"
        assert result["tool_output"] == {"task": _TASK.to_dict()}

    def test_missing_task_id_returns_failed_status(self):
        result = dispatch_tool("update_task", {"title": "Buy oat milk"})

        assert result["tool_status"] == "failed"
        assert result["tool_retry_count"] == 0


class TestDispatchCompleteTask:
    @patch("assistant.tools.dispatch.complete_task")
    def test_completes_task_and_returns_task_dict(self, mock_complete):
        mock_complete.return_value = _TASK

        result = dispatch_tool("complete_task", {"task_id": "t1"})

        mock_complete.assert_called_once_with("t1", tasklist_id="@default")
        assert result["tool_status"] == "success"
        assert result["tool_output"] == {"task": _TASK.to_dict()}

    def test_missing_task_id_returns_failed_status(self):
        result = dispatch_tool("complete_task", {})

        assert result["tool_status"] == "failed"
        assert result["tool_retry_count"] == 0


# ---------------------------------------------------------------------------
# Weather tool (AC23)
# ---------------------------------------------------------------------------

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


class TestDispatchGetWeather:
    @patch("assistant.tools.dispatch.get_current_weather")
    def test_returns_weather_dict(self, mock_get):
        mock_get.return_value = _WEATHER

        result = dispatch_tool("get_weather", {"location": "Seoul"})

        mock_get.assert_called_once_with("Seoul")
        assert result["tool_status"] == "success"
        assert result["tool_output"] == {"weather": _WEATHER.to_dict()}

    def test_empty_location_returns_failed_status(self):
        result = dispatch_tool("get_weather", {"location": ""})

        assert result["tool_status"] == "failed"
        assert result["tool_retry_count"] == 0

    @patch("assistant.tools.dispatch.get_current_weather")
    def test_location_not_found_returns_failed_with_zero_retry_count(self, mock_get):
        mock_get.side_effect = LocationNotFoundError("No location found matching 'Nowhereville'")

        result = dispatch_tool("get_weather", {"location": "Nowhereville"})

        assert result["tool_status"] == "failed"
        assert result["tool_retry_count"] == 0
        assert "Nowhereville" in result["tool_error_message"]

    @patch("assistant.tools.dispatch.get_current_weather")
    def test_runtime_error_after_retries_returns_max_retry_count(self, mock_get):
        mock_get.side_effect = RuntimeError("Unable to retrieve weather for 'Seoul' after 3 attempts.")

        result = dispatch_tool("get_weather", {"location": "Seoul"})

        assert result["tool_status"] == "failed"
        assert result["tool_retry_count"] == _MAX_RETRY_COUNT


# ---------------------------------------------------------------------------
# Document tools (AC23)
# ---------------------------------------------------------------------------


class TestDispatchReadDocument:
    def test_reads_pdf_by_extension(self):
        mock_reader = MagicMock(return_value="extracted pdf text")

        with patch.dict("assistant.tools.dispatch._SUPPORTED_DOCUMENT_READERS", {".pdf": mock_reader}):
            result = dispatch_tool("read_document", {"file_path": "/tmp/report.pdf"})

        mock_reader.assert_called_once()
        assert result["tool_status"] == "success"
        assert result["tool_output"] == {"text": "extracted pdf text"}

    def test_reads_docx_by_extension(self):
        mock_reader = MagicMock(return_value="extracted docx text")

        with patch.dict("assistant.tools.dispatch._SUPPORTED_DOCUMENT_READERS", {".docx": mock_reader}):
            result = dispatch_tool("read_document", {"file_path": "/tmp/report.docx"})

        mock_reader.assert_called_once()
        assert result["tool_status"] == "success"
        assert result["tool_output"] == {"text": "extracted docx text"}

    def test_unsupported_extension_returns_failed_status(self):
        result = dispatch_tool("read_document", {"file_path": "/tmp/report.txt"})

        assert result["tool_status"] == "failed"
        assert result["tool_retry_count"] == 0

    def test_missing_file_path_returns_failed_status(self):
        result = dispatch_tool("read_document", {})

        assert result["tool_status"] == "failed"
        assert result["tool_retry_count"] == 0

    def test_file_not_found_returns_failed_with_zero_retry_count(self):
        mock_reader = MagicMock(side_effect=FileNotFoundError("PDF file not found: /tmp/missing.pdf"))

        with patch.dict("assistant.tools.dispatch._SUPPORTED_DOCUMENT_READERS", {".pdf": mock_reader}):
            result = dispatch_tool("read_document", {"file_path": "/tmp/missing.pdf"})

        assert result["tool_status"] == "failed"
        assert result["tool_retry_count"] == 0
        assert "missing.pdf" in result["tool_error_message"]

    def test_document_error_returns_failed_with_max_retry_count(self):
        mock_reader = MagicMock(side_effect=DocumentError("Unable to read PDF '/tmp/corrupt.pdf': bad header"))

        with patch.dict("assistant.tools.dispatch._SUPPORTED_DOCUMENT_READERS", {".pdf": mock_reader}):
            result = dispatch_tool("read_document", {"file_path": "/tmp/corrupt.pdf"})

        assert result["tool_status"] == "failed"
        assert result["tool_retry_count"] == _MAX_RETRY_COUNT


class TestDispatchCreateDocument:
    @patch("assistant.tools.dispatch.create_docx")
    def test_creates_docx_and_returns_path(self, mock_create_docx):
        mock_create_docx.return_value = Path("/tmp/summary.docx")

        result = dispatch_tool("create_document", {"file_path": "/tmp/summary.docx", "content": "Hello world"})

        mock_create_docx.assert_called_once_with("/tmp/summary.docx", "Hello world")
        assert result["tool_status"] == "success"
        assert result["tool_output"] == {"file_path": "/tmp/summary.docx"}

    def test_missing_content_returns_failed_status(self):
        result = dispatch_tool("create_document", {"file_path": "/tmp/summary.docx"})

        assert result["tool_status"] == "failed"
        assert result["tool_retry_count"] == 0


# ---------------------------------------------------------------------------
# Code execution tool (AC23)
# ---------------------------------------------------------------------------

_CODE_RESULT = CodeExecutionResult(stdout="hello\n", stderr="", exit_code=0, timed_out=False)


class TestDispatchExecuteCode:
    @patch("assistant.tools.dispatch.execute_code")
    def test_executes_code_and_returns_result_dict(self, mock_execute):
        mock_execute.return_value = _CODE_RESULT

        result = dispatch_tool("execute_code", {"code": "print('hello')"})

        mock_execute.assert_called_once_with("print('hello')", language="python")
        assert result["tool_status"] == "success"
        assert result["tool_output"] == {"result": _CODE_RESULT.to_dict()}

    def test_empty_code_returns_failed_status(self):
        result = dispatch_tool("execute_code", {"code": "   "})

        assert result["tool_status"] == "failed"
        assert result["tool_retry_count"] == 0

    @patch("assistant.tools.dispatch.execute_code")
    def test_runtime_error_after_retries_returns_max_retry_count(self, mock_execute):
        mock_execute.side_effect = RuntimeError("Unable to execute code in the Docker sandbox after 3 attempts.")

        result = dispatch_tool("execute_code", {"code": "print('hello')"})

        assert result["tool_status"] == "failed"
        assert result["tool_retry_count"] == _MAX_RETRY_COUNT


# ---------------------------------------------------------------------------
# Unified catch-all for unexpected errors (AC23)
# ---------------------------------------------------------------------------


class TestDispatchUnexpectedError:
    @patch("assistant.tools.dispatch.get_current_weather")
    def test_unexpected_exception_returns_failed_with_zero_retry_count(self, mock_get):
        mock_get.side_effect = TypeError("boom")

        result = dispatch_tool("get_weather", {"location": "Seoul"})

        assert result["tool_status"] == "failed"
        assert result["tool_output"] == {}
        assert result["tool_retry_count"] == 0
        assert "get_weather" in result["tool_error_message"]
        assert "boom" in result["tool_error_message"]
