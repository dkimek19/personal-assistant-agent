"""Tool definitions for the LLM tool-calling loop (AC1-3, AC23).

:data:`TOOL_DEFINITIONS` describes every tool registered in
:mod:`assistant.tools.dispatch` using the JSON-schema ``{"type": "function",
"function": {...}}`` shape expected by Ollama's (and OpenAI-compatible)
``/api/chat`` ``tools`` parameter. :func:`assistant.agent_core.default_responder`
sends this list with every chat request so the model can choose to call a
tool; :func:`assistant.tools.dispatch.dispatch_tool` is then invoked with the
tool name and arguments the model selects.

Each ``name`` here matches a key in
:data:`assistant.tools.dispatch._TOOL_HANDLERS` exactly (the alias
``"searxng"`` is omitted -- ``"web_search"`` is the canonical name offered to
the model).
"""

from __future__ import annotations

from typing import Any

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web via SearXNG and return the most relevant "
                "results (title, URL, snippet for each). Use this for "
                "current events, facts, or anything not already known, then "
                "answer the user using these results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to fetch from SearXNG (optional).",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of top results to keep after relevance filtering (optional).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_calendar_events",
            "description": "List Google Calendar events between two dates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "Start of the range, as an ISO 8601 date or datetime.",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End of the range, as an ISO 8601 date or datetime.",
                    },
                },
                "required": ["start_date", "end_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_calendar_event",
            "description": "Create a new Google Calendar event.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Event title."},
                    "start_time": {"type": "string", "description": "Start time, ISO 8601."},
                    "end_time": {"type": "string", "description": "End time, ISO 8601."},
                    "description": {"type": "string", "description": "Event description (optional)."},
                },
                "required": ["title", "start_time", "end_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_calendar_event",
            "description": "Update one or more fields of an existing Google Calendar event.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "string", "description": "ID of the event to update."},
                    "title": {"type": "string", "description": "New title (optional)."},
                    "start_time": {"type": "string", "description": "New start time, ISO 8601 (optional)."},
                    "end_time": {"type": "string", "description": "New end time, ISO 8601 (optional)."},
                    "description": {"type": "string", "description": "New description (optional)."},
                },
                "required": ["event_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_calendar_event",
            "description": "Delete a Google Calendar event by ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "string", "description": "ID of the event to delete."},
                },
                "required": ["event_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_tasks",
            "description": "List items from a Google Tasks list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tasklist_id": {
                        "type": "string",
                        "description": "Task list ID (optional, defaults to the primary list).",
                    },
                    "show_completed": {
                        "type": "boolean",
                        "description": "Include completed tasks (optional, defaults to false).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_task",
            "description": "Create a new Google Tasks item.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Task title."},
                    "due_date": {"type": "string", "description": "Due date, ISO 8601 (optional)."},
                    "notes": {"type": "string", "description": "Additional notes (optional)."},
                    "tasklist_id": {
                        "type": "string",
                        "description": "Task list ID (optional, defaults to the primary list).",
                    },
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_task",
            "description": "Update one or more fields of an existing Google Tasks item.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "ID of the task to update."},
                    "title": {"type": "string", "description": "New title (optional)."},
                    "due_date": {"type": "string", "description": "New due date, ISO 8601 (optional)."},
                    "notes": {"type": "string", "description": "New notes (optional)."},
                    "tasklist_id": {
                        "type": "string",
                        "description": "Task list ID (optional, defaults to the primary list).",
                    },
                },
                "required": ["task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "complete_task",
            "description": "Mark a Google Tasks item as completed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "ID of the task to complete."},
                    "tasklist_id": {
                        "type": "string",
                        "description": "Task list ID (optional, defaults to the primary list).",
                    },
                },
                "required": ["task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather conditions for a named location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City or place name, e.g. 'Seoul'."},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_document",
            "description": "Extract text from a PDF or DOCX file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to a .pdf or .docx file."},
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_document",
            "description": "Generate a new DOCX file from text content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to write the .docx file to."},
                    "content": {
                        "type": ["string", "array"],
                        "items": {"type": "string"},
                        "description": "Document text, either a single string or a list of paragraphs.",
                    },
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": "Run code in an isolated Docker sandbox and return its stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Source code to execute."},
                    "language": {
                        "type": "string",
                        "description": "Programming language (optional, defaults to 'python').",
                    },
                },
                "required": ["code"],
            },
        },
    },
]
