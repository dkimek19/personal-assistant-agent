"""Tool-calling dispatch pipeline for the Personal Assistant Agent.

Provides :func:`dispatch_tool`, the single entry point the agent's
tool-calling loop uses to select and invoke a registered tool by name and
return a result shaped per the ontology's ``tool_*`` fields (``tool_name``,
``tool_input``, ``tool_output``, ``tool_status``, ``tool_retry_count``,
``tool_error_message``).

Registered tools
-----------------
- ``"web_search"`` / ``"searxng"``: SearXNG web search, ranked/filtered by
  relevance. The filtered results are returned as-is (titles, URLs,
  snippets) for the main chat LLM to read directly as tool-call context --
  no separate Ollama summarization round-trip. Pipeline:
  :func:`assistant.tools.searxng.search` ->
  :func:`assistant.tools.searxng.filter_results`.
- ``"get_calendar_events"`` / ``"create_calendar_event"`` /
  ``"update_calendar_event"`` / ``"delete_calendar_event"``: Google Calendar
  read/write (:mod:`assistant.tools.calendar`).
- ``"get_tasks"`` / ``"create_task"`` / ``"update_task"`` /
  ``"complete_task"``: Google Tasks read/write (:mod:`assistant.tools.tasks`).
- ``"get_weather"``: current weather conditions
  (:mod:`assistant.tools.weather`).
- ``"read_document"`` / ``"create_document"``: PDF/DOCX read and DOCX
  generation (:mod:`assistant.tools.documents`).
- ``"execute_code"``: Docker sandbox code execution
  (:mod:`assistant.tools.code_execution`).

Data-fidelity principle
------------------------
``tool_output`` is the raw, unmodified return value of the underlying tool
function(s) — it is never altered by this dispatcher.

Unified error handling (AC23)
------------------------------
Every tool above performs its own retries with exponential backoff (2
retries, 3 attempts total) before raising. :func:`dispatch_tool` never lets
a handler exception propagate: it normalises *any* failure -- whether from
exhausted retries, invalid input, a missing file, or an unexpected bug in a
handler -- into the same ``tool_status="failed"`` / ``tool_error_message``
shape, so the agent can always surface a clear message to the user.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from assistant.tools.calendar import (
    create_calendar_event,
    delete_calendar_event,
    get_calendar_events,
    update_calendar_event,
)
from assistant.tools.code_execution import execute_code
from assistant.tools.documents import create_docx, read_docx, read_pdf
from assistant.tools.searxng import _DEFAULT_NUM_RESULTS, filter_results, search
from assistant.tools.tasks import complete_task, create_task, get_tasks, update_task
from assistant.tools.weather import get_current_weather

logger = logging.getLogger(__name__)

# Retry count reported when a tool fails after exhausting its internal
# retries. Mirrors `max_retries=2` (3 attempts total) used throughout
# `assistant.tools.*` and `assistant.llm`.
_MAX_RETRY_COUNT: int = 2


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


def _handle_web_search(tool_input: dict[str, Any]) -> dict[str, Any]:
    """Run the web-search pipeline: search -> filter.

    Args:
        tool_input: Must contain a non-empty ``"query"`` string. Optional
            keys: ``"num_results"`` (results requested from SearXNG,
            default :data:`assistant.tools.searxng._DEFAULT_NUM_RESULTS`) and
            ``"top_n"`` (results kept after relevance filtering, defaults to
            ``num_results``).

    Returns:
        A dict with key ``"results"`` -- the filtered, ranked result dicts
        (titles, URLs, snippets), passed through verbatim for the main chat
        LLM to read directly.

    Raises:
        ValueError: If ``"query"`` is missing or empty.
        ToolError: If the SearXNG search exhausts its retries.
    """
    query = tool_input.get("query", "")
    if not query or not str(query).strip():
        raise ValueError("web_search requires a non-empty 'query' field")

    num_results = tool_input.get("num_results", _DEFAULT_NUM_RESULTS)
    top_n = tool_input.get("top_n", num_results)

    raw_results = search(query, num_results=num_results)
    filtered = filter_results(raw_results, query, top_n)

    return {"results": filtered}


def _handle_get_calendar_events(tool_input: dict[str, Any]) -> dict[str, Any]:
    """List Google Calendar events in a date range.

    Args:
        tool_input: Must contain ``"start_date"`` and ``"end_date"`` (ISO
            8601 date or datetime strings).

    Returns:
        ``{"events": [...]}`` -- each event as a plain dict
        (:meth:`~assistant.tools.calendar.CalendarEvent.to_dict`).

    Raises:
        ValueError: If ``"start_date"`` or ``"end_date"`` is missing.
        RuntimeError: If the Google Calendar API exhausts its retries.
    """
    start_date = tool_input.get("start_date")
    end_date = tool_input.get("end_date")
    if not start_date or not end_date:
        raise ValueError("get_calendar_events requires 'start_date' and 'end_date' fields")

    events = get_calendar_events(start_date, end_date)
    return {"events": [event.to_dict() for event in events]}


def _handle_create_calendar_event(tool_input: dict[str, Any]) -> dict[str, Any]:
    """Create a new Google Calendar event.

    Args:
        tool_input: Must contain ``"title"``, ``"start_time"``, and
            ``"end_time"``. Optional: ``"description"``.

    Returns:
        ``{"event_id": ...}``.

    Raises:
        ValueError: If a required field is missing.
        RuntimeError: If the Google Calendar API exhausts its retries.
    """
    title = tool_input.get("title")
    start_time = tool_input.get("start_time")
    end_time = tool_input.get("end_time")
    if not title or not start_time or not end_time:
        raise ValueError("create_calendar_event requires 'title', 'start_time', and 'end_time' fields")

    event_id = create_calendar_event(title, start_time, end_time, tool_input.get("description", ""))
    return {"event_id": event_id}


def _handle_update_calendar_event(tool_input: dict[str, Any]) -> dict[str, Any]:
    """Update fields of an existing Google Calendar event.

    Args:
        tool_input: Must contain ``"event_id"``. At least one of
            ``"title"``, ``"start_time"``, ``"end_time"``, ``"description"``
            must also be present.

    Returns:
        ``{"event": {...}}`` -- the updated event as a plain dict.

    Raises:
        ValueError: If ``"event_id"`` is missing, or no fields to update are
            provided.
        RuntimeError: If the Google Calendar API exhausts its retries.
    """
    event_id = tool_input.get("event_id")
    if not event_id:
        raise ValueError("update_calendar_event requires an 'event_id' field")

    event = update_calendar_event(
        event_id,
        title=tool_input.get("title"),
        start_time=tool_input.get("start_time"),
        end_time=tool_input.get("end_time"),
        description=tool_input.get("description"),
    )
    return {"event": event.to_dict()}


def _handle_delete_calendar_event(tool_input: dict[str, Any]) -> dict[str, Any]:
    """Delete a Google Calendar event by ID.

    Args:
        tool_input: Must contain ``"event_id"``.

    Returns:
        ``{"deleted": True}``.

    Raises:
        ValueError: If ``"event_id"`` is missing.
        RuntimeError: If the Google Calendar API exhausts its retries.
    """
    event_id = tool_input.get("event_id")
    if not event_id:
        raise ValueError("delete_calendar_event requires an 'event_id' field")

    deleted = delete_calendar_event(event_id)
    return {"deleted": deleted}


def _handle_get_tasks(tool_input: dict[str, Any]) -> dict[str, Any]:
    """List Google Tasks.

    Args:
        tool_input: Optional keys: ``"tasklist_id"`` (default
            ``"@default"``), ``"show_completed"`` (default ``False``).

    Returns:
        ``{"tasks": [...]}`` -- each task as a plain dict
        (:meth:`~assistant.tools.tasks.TaskItem.to_dict`).

    Raises:
        RuntimeError: If the Google Tasks API exhausts its retries.
    """
    tasks = get_tasks(
        tasklist_id=tool_input.get("tasklist_id", "@default"),
        show_completed=tool_input.get("show_completed", False),
    )
    return {"tasks": [task.to_dict() for task in tasks]}


def _handle_create_task(tool_input: dict[str, Any]) -> dict[str, Any]:
    """Create a new Google Tasks item.

    Args:
        tool_input: Must contain a non-empty ``"title"``. Optional:
            ``"due_date"``, ``"notes"``, ``"tasklist_id"`` (default
            ``"@default"``).

    Returns:
        ``{"task_id": ...}``.

    Raises:
        ValueError: If ``"title"`` is missing or empty.
        RuntimeError: If the Google Tasks API exhausts its retries.
    """
    title = tool_input.get("title")
    if not title or not str(title).strip():
        raise ValueError("create_task requires a non-empty 'title' field")

    task_id = create_task(
        title,
        tool_input.get("due_date"),
        tool_input.get("notes", ""),
        tasklist_id=tool_input.get("tasklist_id", "@default"),
    )
    return {"task_id": task_id}


def _handle_update_task(tool_input: dict[str, Any]) -> dict[str, Any]:
    """Update fields of an existing Google Tasks item.

    Args:
        tool_input: Must contain ``"task_id"``. At least one of
            ``"title"``, ``"due_date"``, ``"notes"`` must also be present.
            Optional: ``"tasklist_id"`` (default ``"@default"``).

    Returns:
        ``{"task": {...}}`` -- the updated task as a plain dict.

    Raises:
        ValueError: If ``"task_id"`` is missing, or no fields to update are
            provided.
        RuntimeError: If the Google Tasks API exhausts its retries.
    """
    task_id = tool_input.get("task_id")
    if not task_id:
        raise ValueError("update_task requires a 'task_id' field")

    task = update_task(
        task_id,
        title=tool_input.get("title"),
        due_date=tool_input.get("due_date"),
        notes=tool_input.get("notes"),
        tasklist_id=tool_input.get("tasklist_id", "@default"),
    )
    return {"task": task.to_dict()}


def _handle_complete_task(tool_input: dict[str, Any]) -> dict[str, Any]:
    """Mark a Google Tasks item as completed.

    Args:
        tool_input: Must contain ``"task_id"``. Optional: ``"tasklist_id"``
            (default ``"@default"``).

    Returns:
        ``{"task": {...}}`` -- the completed task as a plain dict.

    Raises:
        ValueError: If ``"task_id"`` is missing.
        RuntimeError: If the Google Tasks API exhausts its retries.
    """
    task_id = tool_input.get("task_id")
    if not task_id:
        raise ValueError("complete_task requires a 'task_id' field")

    task = complete_task(task_id, tasklist_id=tool_input.get("tasklist_id", "@default"))
    return {"task": task.to_dict()}


def _handle_get_weather(tool_input: dict[str, Any]) -> dict[str, Any]:
    """Get current weather conditions for a named location.

    Args:
        tool_input: Must contain a non-empty ``"location"`` string.

    Returns:
        ``{"weather": {...}}``
        (:meth:`~assistant.tools.weather.WeatherReport.to_dict`).

    Raises:
        ValueError: If ``"location"`` is missing or empty, or if no matching
            location is found (:class:`~assistant.tools.weather.LocationNotFoundError`).
        RuntimeError: If the weather API exhausts its retries.
    """
    location = tool_input.get("location")
    if not location or not str(location).strip():
        raise ValueError("get_weather requires a non-empty 'location' field")

    report = get_current_weather(location)
    return {"weather": report.to_dict()}


_SUPPORTED_DOCUMENT_READERS: dict[str, Callable[[Path], str]] = {
    ".pdf": read_pdf,
    ".docx": read_docx,
}


def _handle_read_document(tool_input: dict[str, Any]) -> dict[str, Any]:
    """Extract text from a PDF or DOCX file.

    Args:
        tool_input: Must contain ``"file_path"`` (a ``.pdf`` or ``.docx``
            path).

    Returns:
        ``{"text": ...}`` -- the extracted text, unmodified.

    Raises:
        ValueError: If ``"file_path"`` is missing or has an unsupported
            extension.
        FileNotFoundError: If the file does not exist.
        DocumentError: If the file cannot be parsed.
    """
    file_path = tool_input.get("file_path")
    if not file_path:
        raise ValueError("read_document requires a 'file_path' field")

    path = Path(file_path)
    reader = _SUPPORTED_DOCUMENT_READERS.get(path.suffix.lower())
    if reader is None:
        raise ValueError(f"read_document: unsupported file type {path.suffix!r} (expected .pdf or .docx)")

    return {"text": reader(path)}


def _handle_create_document(tool_input: dict[str, Any]) -> dict[str, Any]:
    """Generate a new DOCX file from text content.

    Args:
        tool_input: Must contain ``"file_path"`` and non-empty ``"content"``
            (a string, or a list of paragraph strings).

    Returns:
        ``{"file_path": ...}`` -- the path the file was written to.

    Raises:
        ValueError: If ``"file_path"`` or ``"content"`` is missing/empty.
        DocumentError: If the file cannot be written.
    """
    file_path = tool_input.get("file_path")
    content = tool_input.get("content")
    if not file_path:
        raise ValueError("create_document requires a 'file_path' field")
    if not content:
        raise ValueError("create_document requires non-empty 'content'")

    path = create_docx(file_path, content)
    return {"file_path": str(path)}


def _handle_execute_code(tool_input: dict[str, Any]) -> dict[str, Any]:
    """Run code inside the Docker sandbox.

    Args:
        tool_input: Must contain a non-empty ``"code"`` string. Optional:
            ``"language"`` (default ``"python"``).

    Returns:
        ``{"result": {...}}``
        (:meth:`~assistant.tools.code_execution.CodeExecutionResult.to_dict`).

    Raises:
        ValueError: If ``"code"`` is missing or empty.
        RuntimeError: If the Docker sandbox exhausts its retries.
    """
    code = tool_input.get("code")
    if not code or not str(code).strip():
        raise ValueError("execute_code requires a non-empty 'code' field")

    result = execute_code(code, language=tool_input.get("language", "python"))
    return {"result": result.to_dict()}


_TOOL_HANDLERS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "web_search": _handle_web_search,
    "searxng": _handle_web_search,
    "get_calendar_events": _handle_get_calendar_events,
    "create_calendar_event": _handle_create_calendar_event,
    "update_calendar_event": _handle_update_calendar_event,
    "delete_calendar_event": _handle_delete_calendar_event,
    "get_tasks": _handle_get_tasks,
    "create_task": _handle_create_task,
    "update_task": _handle_update_task,
    "complete_task": _handle_complete_task,
    "get_weather": _handle_get_weather,
    "read_document": _handle_read_document,
    "create_document": _handle_create_document,
    "execute_code": _handle_execute_code,
}


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def _failed_result(
    tool_name: str, tool_input: dict[str, Any], *, retry_count: int, error_message: str
) -> dict[str, Any]:
    """Build the normalised ``tool_status="failed"`` result shape."""
    return {
        "tool_name": tool_name,
        "tool_input": tool_input,
        "tool_output": {},
        "tool_status": "failed",
        "tool_retry_count": retry_count,
        "tool_error_message": error_message,
    }


def dispatch_tool(tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    """Select and invoke the tool named *tool_name* with *tool_input*.

    This is the agent's tool-calling pipeline entry point: given a tool name
    (as the LLM would select it) and its input parameters, it looks up the
    matching handler, invokes it, and normalises the outcome into the
    ontology's ``tool_*`` shape so callers never need to handle handler
    exceptions directly.

    Args:
        tool_name: Name of the tool to invoke (e.g. ``"web_search"``).
        tool_input: Parameters passed to the tool.

    Returns:
        A dict with keys:
            - ``tool_name`` (str): echoed back from *tool_name*.
            - ``tool_input`` (dict): echoed back from *tool_input*.
            - ``tool_output`` (dict): the handler's raw return value on
              success, or ``{}`` on failure.
            - ``tool_status`` (str): ``"success"`` or ``"failed"``.
            - ``tool_retry_count`` (int): number of retries the underlying
              tool performed before giving up (``0`` on success or for
              non-retryable errors, :data:`_MAX_RETRY_COUNT` when retries
              were exhausted).
            - ``tool_error_message`` (str | None): human-readable error
              description on failure, otherwise ``None``.

    This function never raises -- for unknown tools, invalid input, missing
    files, exhausted retries, or any other handler failure, the outcome is
    reported via ``tool_status`` / ``tool_error_message`` so the agent can
    surface a clear message to the user (per the ``error_transparency``
    evaluation principle and AC23).
    """
    handler = _TOOL_HANDLERS.get(tool_name)

    if handler is None:
        logger.error("dispatch_tool: unknown tool %r", tool_name)
        return _failed_result(
            tool_name, tool_input, retry_count=0, error_message=f"Unknown tool: {tool_name!r}"
        )

    try:
        output = handler(tool_input)
    except ValueError as exc:
        # Non-retryable: invalid/missing input, or a domain error where
        # retrying would not change the outcome (e.g. LocationNotFoundError,
        # an "update with no fields" error).
        logger.error("dispatch_tool: %r received invalid input: %s", tool_name, exc)
        return _failed_result(tool_name, tool_input, retry_count=0, error_message=str(exc))
    except FileNotFoundError as exc:
        # Non-retryable: the referenced file does not exist.
        logger.error("dispatch_tool: %r could not find file: %s", tool_name, exc)
        return _failed_result(tool_name, tool_input, retry_count=0, error_message=str(exc))
    except RuntimeError as exc:
        # The underlying tool already retried internally (2 retries with
        # exponential backoff -- ToolError, OllamaError, DocumentError, and
        # the plain RuntimeErrors raised by calendar/tasks/weather/
        # code_execution all derive from RuntimeError) and gave up.
        logger.error("dispatch_tool: %r failed after retries: %s", tool_name, exc)
        return _failed_result(tool_name, tool_input, retry_count=_MAX_RETRY_COUNT, error_message=str(exc))
    except Exception as exc:  # noqa: BLE001 - last resort so dispatch_tool never raises
        logger.exception("dispatch_tool: %r raised an unexpected error", tool_name)
        return _failed_result(
            tool_name,
            tool_input,
            retry_count=0,
            error_message=f"An unexpected error occurred while running '{tool_name}': {exc}",
        )

    return {
        "tool_name": tool_name,
        "tool_input": tool_input,
        "tool_output": output,
        "tool_status": "success",
        "tool_retry_count": 0,
        "tool_error_message": None,
    }
