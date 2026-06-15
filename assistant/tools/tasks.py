"""Google Tasks tool for the Personal Assistant Agent.

Provides four public functions:
- get_tasks: Retrieves tasks from a Google Tasks list.
- create_task: Creates a new task and returns its ID.
- update_task: Updates fields of an existing task by task ID.
- complete_task: Marks an existing task as completed.

Tool-returned factual data (titles, due dates, status) is never altered by
the LLM — formatting only, no content modification.

Constraint: exponential backoff with 2 retries on API failures, then explicit
error message to user.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class TaskItem:
    """Structured representation of a single Google Tasks item.

    All factual fields (task_id, task_title, task_due_date, task_status,
    task_completed) are stored exactly as returned by the API — no LLM
    modification allowed.
    """

    task_id: str
    task_title: str
    task_due_date: str = ""
    task_status: str = "needsAction"
    task_completed: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict representation (safe to pass to LLM context)."""
        return {
            "task_id": self.task_id,
            "task_title": self.task_title,
            "task_due_date": self.task_due_date,
            "task_status": self.task_status,
            "task_completed": self.task_completed,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_task(raw: dict[str, Any]) -> TaskItem:
    """Convert a raw Google Tasks API task dict to a :class:`TaskItem`.

    Factual data is passed through without modification.
    """
    status: str = raw.get("status", "needsAction")
    return TaskItem(
        task_id=raw.get("id", ""),
        task_title=raw.get("title", "(No title)"),
        task_due_date=raw.get("due", ""),
        task_status=status,
        task_completed=status == "completed",
        notes=raw.get("notes", ""),
    )


# ---------------------------------------------------------------------------
# Main public functions
# ---------------------------------------------------------------------------


def get_tasks(
    *,
    service: Any | None = None,
    tasklist_id: str = "@default",
    show_completed: bool = False,
    max_results: int = 100,
    max_retries: int = 2,
    base_backoff: float = 1.0,
) -> list[TaskItem]:
    """Retrieve tasks from a Google Tasks list.

    Parameters
    ----------
    service:
        An already-constructed Google Tasks API resource object
        (``googleapiclient.discovery.Resource``).  When *None* the function
        builds one automatically using :func:`_build_tasks_service`.
    tasklist_id:
        The Tasks list ID to query.  Defaults to ``"@default"`` (the user's
        default task list).
    show_completed:
        Whether to include tasks that have already been marked completed.
        Defaults to ``False``.
    max_results:
        Maximum number of tasks to retrieve across all pages.
    max_retries:
        Number of retry attempts on transient failures (default 2).
    base_backoff:
        Base delay in seconds for exponential backoff.

    Returns
    -------
    list[TaskItem]
        List of :class:`TaskItem` objects. Returns an empty list when no
        tasks are found.

    Raises
    ------
    RuntimeError
        After *max_retries* failures, a ``RuntimeError`` is raised with an
        explicit human-readable message for the user.
    """
    if service is None:
        service = _build_tasks_service()

    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        if attempt > 0:
            delay = base_backoff * (2 ** (attempt - 1))
            logger.warning(
                "Google Tasks API retry %d/%d after %.1fs backoff (error: %s)",
                attempt,
                max_retries,
                delay,
                last_error,
            )
            time.sleep(delay)

        try:
            tasks = _fetch_all_tasks(
                service=service,
                tasklist_id=tasklist_id,
                show_completed=show_completed,
                max_results=max_results,
            )
            logger.info("Retrieved %d tasks from list '%s'", len(tasks), tasklist_id)
            return tasks

        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.error("Google Tasks API error on attempt %d: %s", attempt + 1, exc)

    raise RuntimeError(
        f"Unable to retrieve tasks after {max_retries + 1} attempts. "
        f"Last error: {last_error}. "
        "Please check your Google Tasks credentials and network connection."
    )


def create_task(
    title: str,
    due_date: str | None = None,
    notes: str = "",
    *,
    service: Any | None = None,
    tasklist_id: str = "@default",
    max_retries: int = 2,
    base_backoff: float = 1.0,
) -> str:
    """Create a new task and return the created task ID.

    Parameters
    ----------
    title:
        Task title.
    due_date:
        Optional RFC 3339 timestamp for the task's due date
        (e.g. ``"2026-06-15T00:00:00Z"``). Omitted from the payload if *None*.
    notes:
        Optional free-text notes for the task.
    service:
        An already-constructed Google Tasks API resource object.  When
        *None* the function builds one automatically using
        :func:`_build_tasks_service`.
    tasklist_id:
        The Tasks list ID to create the task in.  Defaults to ``"@default"``.
    max_retries:
        Number of retry attempts on transient failures (default 2).
    base_backoff:
        Base delay in seconds for exponential backoff.

    Returns
    -------
    str
        The ``id`` of the newly created task, exactly as returned by the
        Google Tasks API — no modification.

    Raises
    ------
    RuntimeError
        After *max_retries* failures, a ``RuntimeError`` is raised with an
        explicit human-readable message for the user.
    """
    if service is None:
        service = _build_tasks_service()

    body: dict[str, Any] = {"title": title}
    if due_date is not None:
        body["due"] = due_date
    if notes:
        body["notes"] = notes

    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        if attempt > 0:
            delay = base_backoff * (2 ** (attempt - 1))
            logger.warning(
                "Google Tasks API create retry %d/%d after %.1fs backoff (error: %s)",
                attempt,
                max_retries,
                delay,
                last_error,
            )
            time.sleep(delay)

        try:
            response: dict[str, Any] = (
                service.tasks().insert(tasklist=tasklist_id, body=body).execute()
            )
            task_id: str = response.get("id", "")
            logger.info("Created task '%s' with ID '%s'", title, task_id)
            return task_id

        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.error(
                "Google Tasks API create error on attempt %d: %s", attempt + 1, exc
            )

    raise RuntimeError(
        f"Unable to create task after {max_retries + 1} attempts. "
        f"Last error: {last_error}. "
        "Please check your Google Tasks credentials and network connection."
    )


def update_task(
    task_id: str,
    *,
    title: str | None = None,
    due_date: str | None = None,
    notes: str | None = None,
    service: Any | None = None,
    tasklist_id: str = "@default",
    max_retries: int = 2,
    base_backoff: float = 1.0,
) -> TaskItem:
    """Update fields of an existing Google Tasks item by task ID.

    Only the fields explicitly provided are changed (a partial update via the
    Tasks API's ``patch`` method) — omitted fields retain their existing
    values on the task.

    Parameters
    ----------
    task_id:
        The ``id`` of the task to update, as returned by :func:`get_tasks` or
        :func:`create_task`.
    title:
        New task title.  Unchanged if *None*.
    due_date:
        New RFC 3339 due-date timestamp.  Unchanged if *None*.
    notes:
        New free-text notes.  Unchanged if *None*.
    service:
        An already-constructed Google Tasks API resource object.  When
        *None* the function builds one automatically using
        :func:`_build_tasks_service`.
    tasklist_id:
        The Tasks list ID containing the task.  Defaults to ``"@default"``.
    max_retries:
        Number of retry attempts on transient failures (default 2).
    base_backoff:
        Base delay in seconds for exponential backoff.

    Returns
    -------
    TaskItem
        The updated task, parsed from the Google Tasks API response —
        factual fields are passed through without modification.

    Raises
    ------
    ValueError
        If none of *title*, *due_date*, or *notes* is provided (nothing to
        update).
    RuntimeError
        After *max_retries* failures, a ``RuntimeError`` is raised with an
        explicit human-readable message for the user.
    """
    if title is None and due_date is None and notes is None:
        raise ValueError(
            "update_task requires at least one of: title, due_date, notes"
        )

    if service is None:
        service = _build_tasks_service()

    body: dict[str, Any] = {}
    if title is not None:
        body["title"] = title
    if due_date is not None:
        body["due"] = due_date
    if notes is not None:
        body["notes"] = notes

    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        if attempt > 0:
            delay = base_backoff * (2 ** (attempt - 1))
            logger.warning(
                "Google Tasks API update retry %d/%d after %.1fs backoff (error: %s)",
                attempt,
                max_retries,
                delay,
                last_error,
            )
            time.sleep(delay)

        try:
            response: dict[str, Any] = (
                service.tasks()
                .patch(tasklist=tasklist_id, task=task_id, body=body)
                .execute()
            )
            logger.info("Updated task '%s'", task_id)
            return _parse_task(response)

        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.error(
                "Google Tasks API update error on attempt %d: %s", attempt + 1, exc
            )

    raise RuntimeError(
        f"Unable to update task '{task_id}' after {max_retries + 1} attempts. "
        f"Last error: {last_error}. "
        "Please check your Google Tasks credentials and network connection."
    )


def complete_task(
    task_id: str,
    *,
    service: Any | None = None,
    tasklist_id: str = "@default",
    max_retries: int = 2,
    base_backoff: float = 1.0,
) -> TaskItem:
    """Mark an existing Google Tasks item as completed.

    Parameters
    ----------
    task_id:
        The ``id`` of the task to complete, as returned by :func:`get_tasks`
        or :func:`create_task`.
    service:
        An already-constructed Google Tasks API resource object.  When
        *None* the function builds one automatically using
        :func:`_build_tasks_service`.
    tasklist_id:
        The Tasks list ID containing the task.  Defaults to ``"@default"``.
    max_retries:
        Number of retry attempts on transient failures (default 2).
    base_backoff:
        Base delay in seconds for exponential backoff.

    Returns
    -------
    TaskItem
        The completed task, parsed from the Google Tasks API response.
        ``task_status`` will be ``"completed"`` and ``task_completed`` will
        be ``True``.

    Raises
    ------
    RuntimeError
        After *max_retries* failures, a ``RuntimeError`` is raised with an
        explicit human-readable message for the user.
    """
    if service is None:
        service = _build_tasks_service()

    body: dict[str, Any] = {"status": "completed"}

    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        if attempt > 0:
            delay = base_backoff * (2 ** (attempt - 1))
            logger.warning(
                "Google Tasks API complete retry %d/%d after %.1fs backoff (error: %s)",
                attempt,
                max_retries,
                delay,
                last_error,
            )
            time.sleep(delay)

        try:
            response: dict[str, Any] = (
                service.tasks()
                .patch(tasklist=tasklist_id, task=task_id, body=body)
                .execute()
            )
            logger.info("Completed task '%s'", task_id)
            return _parse_task(response)

        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.error(
                "Google Tasks API complete error on attempt %d: %s", attempt + 1, exc
            )

    raise RuntimeError(
        f"Unable to complete task '{task_id}' after {max_retries + 1} attempts. "
        f"Last error: {last_error}. "
        "Please check your Google Tasks credentials and network connection."
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fetch_all_tasks(
    service: Any,
    tasklist_id: str,
    show_completed: bool,
    max_results: int,
) -> list[TaskItem]:
    """Fetch all pages of tasks from the Google Tasks API.

    Handles pagination via nextPageToken automatically.
    """
    all_tasks: list[TaskItem] = []
    page_token: str | None = None

    while True:
        kwargs: dict[str, Any] = {
            "tasklist": tasklist_id,
            "showCompleted": show_completed,
            "maxResults": min(max_results, 100),  # API default page size cap
        }
        if page_token:
            kwargs["pageToken"] = page_token

        response: dict[str, Any] = service.tasks().list(**kwargs).execute()

        raw_items: list[dict] = response.get("items", [])
        for raw in raw_items:
            all_tasks.append(_parse_task(raw))

        page_token = response.get("nextPageToken")
        if not page_token:
            break

        if len(all_tasks) >= max_results:
            all_tasks = all_tasks[:max_results]
            break

    return all_tasks


def build_tasks_service(
    token_file: "str | Path | None" = None,
    credentials_file: "str | Path | None" = None,
    scopes: "list[str] | None" = None,
) -> Any:
    """Build and return an authenticated Google Tasks API service object.

    Constructs a ``googleapiclient.discovery.Resource`` for the Google Tasks
    v1 API using OAuth2 credentials loaded from a local token file.  When the
    token does not yet exist (first run) the function initiates the standard
    OAuth2 authorisation-code flow using *credentials_file*.

    Parameters
    ----------
    token_file:
        Path to the OAuth2 token JSON file produced by the authorisation flow.
        Defaults to ``~/assistant/credentials/tasks_token.json`` (or the
        value of the ``GOOGLE_TASKS_TOKEN_FILE`` environment variable).
    credentials_file:
        Path to the OAuth2 client-secrets JSON file downloaded from the Google
        Cloud Console.  Only required when no valid token exists yet.
        Defaults to ``~/assistant/credentials/credentials.json`` (or the
        value of the ``GOOGLE_TASKS_CREDENTIALS_FILE`` environment variable).
    scopes:
        OAuth2 scopes to request.  Defaults to
        ``["https://www.googleapis.com/auth/tasks"]`` which grants full
        read/write access to Google Tasks.

    Returns
    -------
    googleapiclient.discovery.Resource
        An authenticated Google Tasks API v1 service object ready for use.

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

    _scopes: list[str] = scopes or ["https://www.googleapis.com/auth/tasks"]

    credentials_dir = _Path(
        os.environ.get(
            "ASSISTANT_CREDENTIALS_DIR",
            _Path.home() / "assistant" / "credentials",
        )
    )
    _token_path = _Path(token_file) if token_file is not None else _Path(
        os.environ.get(
            "GOOGLE_TASKS_TOKEN_FILE",
            credentials_dir / "tasks_token.json",
        )
    )
    _credentials_path = _Path(credentials_file) if credentials_file is not None else _Path(
        os.environ.get(
            "GOOGLE_TASKS_CREDENTIALS_FILE",
            credentials_dir / "credentials.json",
        )
    )

    creds: Any = None

    if _token_path.exists():
        creds = Credentials.from_authorized_user_file(str(_token_path), _scopes)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not _credentials_path.exists():
                raise RuntimeError(
                    f"Google Tasks credentials file not found at {_credentials_path}. "
                    "Download credentials.json from Google Cloud Console and place it there."
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                str(_credentials_path), _scopes
            )
            creds = flow.run_local_server(port=0)

        _token_path.parent.mkdir(parents=True, exist_ok=True)
        _token_path.write_text(creds.to_json())

    return build("tasks", "v1", credentials=creds)


def _build_tasks_service() -> Any:
    """Build and return an authenticated Google Tasks API service object.

    This is an internal convenience wrapper that forwards to the public
    :func:`build_tasks_service` using environment-variable / default-path
    resolution.  Prefer calling :func:`build_tasks_service` directly when
    explicit path control is needed (e.g., in tests).

    Raises
    ------
    RuntimeError
        If credentials cannot be loaded.
    """
    return build_tasks_service()
