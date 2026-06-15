"""Tests for assistant.tools.tasks (Google Tasks read/write module).

All tests use a mocked Google Tasks API client with fixture data so they run
fully offline with no real credentials required.

Covers:
- get_tasks: parsing, pagination, show_completed forwarding, retries
- create_task: payload construction, returned task ID, retries
- update_task: partial-update payload, ValueError on no fields, retries
- complete_task: status="completed" payload, returned TaskItem, retries
- _parse_task: field mapping and defaults
- build_tasks_service: OAuth2 flow (mocked Google API client libraries)
- Data fidelity: factual fields (id, title, due date, status) are not altered
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import MagicMock, patch

from assistant.tools.tasks import (
    TaskItem,
    build_tasks_service,
    complete_task,
    create_task,
    get_tasks,
    update_task,
    _parse_task,
)


# ---------------------------------------------------------------------------
# Fixture data
# ---------------------------------------------------------------------------

FIXTURE_TASKS_PAGE_1 = {
    "kind": "tasks#tasks",
    "nextPageToken": "TOKEN_PAGE_2",
    "items": [
        {
            "id": "task_001",
            "title": "Buy groceries",
            "status": "needsAction",
            "due": "2026-06-12T00:00:00.000Z",
            "notes": "Milk, eggs, bread",
        },
        {
            "id": "task_002",
            "title": "Finish report",
            "status": "needsAction",
        },
    ],
}

FIXTURE_TASKS_PAGE_2 = {
    "kind": "tasks#tasks",
    "items": [
        {
            "id": "task_003",
            "title": "Call dentist",
            "status": "completed",
            "due": "2026-06-10T00:00:00.000Z",
        },
    ],
}

FIXTURE_TASKS_SINGLE = {
    "kind": "tasks#tasks",
    "items": [
        {
            "id": "task_solo",
            "title": "Read book",
            "status": "needsAction",
            "due": "2026-06-20T00:00:00.000Z",
            "notes": "Chapter 5 onward",
        },
    ],
}

FIXTURE_TASKS_EMPTY = {"kind": "tasks#tasks", "items": []}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_list_service(responses: list[dict]) -> MagicMock:
    """Return a mock service whose tasks().list(...).execute() yields *responses*."""
    service = MagicMock()
    service.tasks.return_value.list.return_value.execute = MagicMock(
        side_effect=responses
    )
    return service


def _make_list_service_with_error(
    error: Exception, success_response: dict | None = None
) -> MagicMock:
    """Return a mock service that raises *error* once, then succeeds (if provided)."""
    service = MagicMock()
    if success_response is not None:
        execute_mock = MagicMock(side_effect=[error, success_response])
    else:
        execute_mock = MagicMock(side_effect=error)
    service.tasks.return_value.list.return_value.execute = execute_mock
    return service


# ---------------------------------------------------------------------------
# Tests for _parse_task
# ---------------------------------------------------------------------------


class TestParseTask(unittest.TestCase):
    def test_parses_full_task(self):
        raw = {
            "id": "task_001",
            "title": "Buy groceries",
            "status": "needsAction",
            "due": "2026-06-12T00:00:00.000Z",
            "notes": "Milk, eggs, bread",
        }
        task = _parse_task(raw)

        self.assertEqual(task.task_id, "task_001")
        self.assertEqual(task.task_title, "Buy groceries")
        self.assertEqual(task.task_status, "needsAction")
        self.assertEqual(task.task_due_date, "2026-06-12T00:00:00.000Z")
        self.assertEqual(task.notes, "Milk, eggs, bread")
        self.assertFalse(task.task_completed)

    def test_completed_status_sets_task_completed_true(self):
        task = _parse_task({"id": "t1", "title": "Done thing", "status": "completed"})
        self.assertTrue(task.task_completed)
        self.assertEqual(task.task_status, "completed")

    def test_missing_id_defaults_to_empty_string(self):
        task = _parse_task({"title": "No ID"})
        self.assertEqual(task.task_id, "")

    def test_missing_title_defaults_to_no_title(self):
        task = _parse_task({"id": "t1"})
        self.assertEqual(task.task_title, "(No title)")

    def test_missing_status_defaults_to_needs_action(self):
        task = _parse_task({"id": "t1", "title": "X"})
        self.assertEqual(task.task_status, "needsAction")
        self.assertFalse(task.task_completed)

    def test_missing_due_and_notes_default_to_empty_string(self):
        task = _parse_task({"id": "t1", "title": "X"})
        self.assertEqual(task.task_due_date, "")
        self.assertEqual(task.notes, "")

    def test_to_dict_contains_all_ontology_keys(self):
        task = _parse_task(
            {
                "id": "t1",
                "title": "X",
                "status": "needsAction",
                "due": "2026-06-12T00:00:00.000Z",
                "notes": "n",
            }
        )
        d = task.to_dict()
        self.assertEqual(
            set(d.keys()),
            {
                "task_id",
                "task_title",
                "task_due_date",
                "task_status",
                "task_completed",
                "notes",
            },
        )


# ---------------------------------------------------------------------------
# Tests for get_tasks
# ---------------------------------------------------------------------------


class TestGetTasks(unittest.TestCase):
    def test_returns_list_of_task_items(self):
        service = _make_list_service([FIXTURE_TASKS_SINGLE])
        tasks = get_tasks(service=service)

        self.assertEqual(len(tasks), 1)
        self.assertIsInstance(tasks[0], TaskItem)

    def test_task_fields_match_fixture_data(self):
        service = _make_list_service([FIXTURE_TASKS_SINGLE])
        tasks = get_tasks(service=service)

        task = tasks[0]
        self.assertEqual(task.task_id, "task_solo")
        self.assertEqual(task.task_title, "Read book")
        self.assertEqual(task.task_due_date, "2026-06-20T00:00:00.000Z")
        self.assertEqual(task.task_status, "needsAction")
        self.assertFalse(task.task_completed)
        self.assertEqual(task.notes, "Chapter 5 onward")

    def test_empty_task_list(self):
        service = _make_list_service([FIXTURE_TASKS_EMPTY])
        tasks = get_tasks(service=service)
        self.assertEqual(tasks, [])

    def test_pagination_fetches_all_pages(self):
        service = _make_list_service([FIXTURE_TASKS_PAGE_1, FIXTURE_TASKS_PAGE_2])
        tasks = get_tasks(service=service)

        self.assertEqual(len(tasks), 3)
        self.assertEqual(
            [t.task_id for t in tasks], ["task_001", "task_002", "task_003"]
        )

    def test_pagination_calls_list_twice(self):
        service = _make_list_service([FIXTURE_TASKS_PAGE_1, FIXTURE_TASKS_PAGE_2])
        get_tasks(service=service)
        self.assertEqual(service.tasks.return_value.list.call_count, 2)

    def test_default_tasklist_id_is_default(self):
        service = _make_list_service([FIXTURE_TASKS_EMPTY])
        get_tasks(service=service)

        call_kwargs = service.tasks.return_value.list.call_args.kwargs
        self.assertEqual(call_kwargs["tasklist"], "@default")

    def test_custom_tasklist_id_forwarded(self):
        service = _make_list_service([FIXTURE_TASKS_EMPTY])
        get_tasks(service=service, tasklist_id="my-list-id")

        call_kwargs = service.tasks.return_value.list.call_args.kwargs
        self.assertEqual(call_kwargs["tasklist"], "my-list-id")

    def test_show_completed_defaults_to_false(self):
        service = _make_list_service([FIXTURE_TASKS_EMPTY])
        get_tasks(service=service)

        call_kwargs = service.tasks.return_value.list.call_args.kwargs
        self.assertEqual(call_kwargs["showCompleted"], False)

    def test_show_completed_true_forwarded(self):
        service = _make_list_service([FIXTURE_TASKS_EMPTY])
        get_tasks(service=service, show_completed=True)

        call_kwargs = service.tasks.return_value.list.call_args.kwargs
        self.assertEqual(call_kwargs["showCompleted"], True)

    # ------------------------------------------------------------------
    # Exponential backoff and retry
    # ------------------------------------------------------------------

    @patch("time.sleep", return_value=None)
    def test_retries_on_transient_error_and_succeeds(self, mock_sleep):
        service = _make_list_service_with_error(
            Exception("503 Service Unavailable"), FIXTURE_TASKS_SINGLE
        )

        tasks = get_tasks(service=service, max_retries=2, base_backoff=1.0)

        self.assertEqual(len(tasks), 1)
        mock_sleep.assert_called_once_with(1.0)

    @patch("time.sleep", return_value=None)
    def test_second_retry_uses_doubled_backoff(self, mock_sleep):
        service = MagicMock()
        service.tasks.return_value.list.return_value.execute = MagicMock(
            side_effect=[Exception("Timeout"), Exception("Timeout"), FIXTURE_TASKS_SINGLE]
        )

        tasks = get_tasks(service=service, max_retries=2, base_backoff=1.0)

        self.assertEqual(len(tasks), 1)
        self.assertEqual(mock_sleep.call_count, 2)
        calls = mock_sleep.call_args_list
        self.assertAlmostEqual(calls[0][0][0], 1.0)
        self.assertAlmostEqual(calls[1][0][0], 2.0)

    @patch("time.sleep", return_value=None)
    def test_raises_runtime_error_after_max_retries(self, mock_sleep):
        service = MagicMock()
        service.tasks.return_value.list.return_value.execute = MagicMock(
            side_effect=Exception("Connection refused")
        )

        with self.assertRaises(RuntimeError) as ctx:
            get_tasks(service=service, max_retries=2, base_backoff=0.01)

        error_msg = str(ctx.exception)
        self.assertIn("Unable to retrieve tasks", error_msg)
        self.assertIn("3 attempts", error_msg)
        self.assertIn("Connection refused", error_msg)

    @patch("time.sleep", return_value=None)
    def test_retry_count_is_exactly_max_retries_plus_one(self, mock_sleep):
        service = MagicMock()
        service.tasks.return_value.list.return_value.execute = MagicMock(
            side_effect=Exception("always fails")
        )

        with self.assertRaises(RuntimeError):
            get_tasks(service=service, max_retries=2, base_backoff=0.01)

        self.assertEqual(
            service.tasks.return_value.list.return_value.execute.call_count, 3
        )


# ---------------------------------------------------------------------------
# Tests for create_task
# ---------------------------------------------------------------------------


class TestCreateTask(unittest.TestCase):
    @staticmethod
    def _make_insert_service(response: dict) -> MagicMock:
        service = MagicMock()
        service.tasks.return_value.insert.return_value.execute = MagicMock(
            return_value=response
        )
        return service

    @staticmethod
    def _make_insert_service_with_errors(
        errors: list[Exception], success_response: dict | None = None
    ) -> MagicMock:
        service = MagicMock()
        side_effects: list = list(errors)
        if success_response is not None:
            side_effects.append(success_response)
        service.tasks.return_value.insert.return_value.execute = MagicMock(
            side_effect=side_effects
        )
        return service

    def test_returns_created_task_id(self):
        service = self._make_insert_service({"id": "task_new_001", "title": "New Task"})

        task_id = create_task("New Task", service=service)

        self.assertEqual(task_id, "task_new_001")

    def test_returns_empty_string_when_id_missing(self):
        service = self._make_insert_service({"title": "No ID"})
        task_id = create_task("No ID", service=service)
        self.assertEqual(task_id, "")

    def test_payload_contains_title(self):
        service = self._make_insert_service({"id": "t1"})
        create_task("Buy milk", service=service)

        body = service.tasks.return_value.insert.call_args.kwargs["body"]
        self.assertEqual(body["title"], "Buy milk")

    def test_payload_omits_due_and_notes_when_not_provided(self):
        service = self._make_insert_service({"id": "t1"})
        create_task("Buy milk", service=service)

        body = service.tasks.return_value.insert.call_args.kwargs["body"]
        self.assertNotIn("due", body)
        self.assertNotIn("notes", body)

    def test_payload_includes_due_date_when_provided(self):
        service = self._make_insert_service({"id": "t1"})
        create_task("Buy milk", due_date="2026-06-15T00:00:00.000Z", service=service)

        body = service.tasks.return_value.insert.call_args.kwargs["body"]
        self.assertEqual(body["due"], "2026-06-15T00:00:00.000Z")

    def test_payload_includes_notes_when_provided(self):
        service = self._make_insert_service({"id": "t1"})
        create_task("Buy milk", notes="2% milk", service=service)

        body = service.tasks.return_value.insert.call_args.kwargs["body"]
        self.assertEqual(body["notes"], "2% milk")

    def test_default_tasklist_id_is_default(self):
        service = self._make_insert_service({"id": "t1"})
        create_task("Buy milk", service=service)

        call_kwargs = service.tasks.return_value.insert.call_args.kwargs
        self.assertEqual(call_kwargs["tasklist"], "@default")

    def test_custom_tasklist_id_forwarded(self):
        service = self._make_insert_service({"id": "t1"})
        create_task("Buy milk", service=service, tasklist_id="my-list-id")

        call_kwargs = service.tasks.return_value.insert.call_args.kwargs
        self.assertEqual(call_kwargs["tasklist"], "my-list-id")

    @patch("time.sleep", return_value=None)
    def test_retries_once_on_transient_error_and_returns_id(self, mock_sleep):
        service = self._make_insert_service_with_errors(
            [Exception("503 Service Unavailable")],
            success_response={"id": "task_retry_ok"},
        )

        task_id = create_task(
            "Retry Test", service=service, max_retries=2, base_backoff=1.0
        )

        self.assertEqual(task_id, "task_retry_ok")
        mock_sleep.assert_called_once_with(1.0)

    @patch("time.sleep", return_value=None)
    def test_raises_runtime_error_after_max_retries(self, mock_sleep):
        service = MagicMock()
        service.tasks.return_value.insert.return_value.execute = MagicMock(
            side_effect=Exception("Connection refused")
        )

        with self.assertRaises(RuntimeError) as ctx:
            create_task("Failing Task", service=service, max_retries=2, base_backoff=0.01)

        error_msg = str(ctx.exception)
        self.assertIn("Unable to create task", error_msg)
        self.assertIn("3 attempts", error_msg)
        self.assertIn("Connection refused", error_msg)


# ---------------------------------------------------------------------------
# Tests for update_task
# ---------------------------------------------------------------------------


class TestUpdateTask(unittest.TestCase):
    @staticmethod
    def _make_patch_service(response: dict) -> MagicMock:
        service = MagicMock()
        service.tasks.return_value.patch.return_value.execute = MagicMock(
            return_value=response
        )
        return service

    @staticmethod
    def _make_patch_service_with_errors(
        errors: list[Exception], success_response: dict | None = None
    ) -> MagicMock:
        service = MagicMock()
        side_effects: list = list(errors)
        if success_response is not None:
            side_effects.append(success_response)
        service.tasks.return_value.patch.return_value.execute = MagicMock(
            side_effect=side_effects
        )
        return service

    def test_returns_task_item_from_response(self):
        service = self._make_patch_service(
            {"id": "task_001", "title": "Updated Title", "status": "needsAction"}
        )

        task = update_task("task_001", title="Updated Title", service=service)

        self.assertIsInstance(task, TaskItem)
        self.assertEqual(task.task_id, "task_001")
        self.assertEqual(task.task_title, "Updated Title")

    def test_payload_contains_only_title_when_only_title_provided(self):
        service = self._make_patch_service({"id": "task_001", "title": "New Title"})

        update_task("task_001", title="New Title", service=service)

        body = service.tasks.return_value.patch.call_args.kwargs["body"]
        self.assertEqual(body, {"title": "New Title"})

    def test_payload_contains_only_due_when_only_due_provided(self):
        service = self._make_patch_service({"id": "task_001"})

        update_task("task_001", due_date="2026-06-20T00:00:00.000Z", service=service)

        body = service.tasks.return_value.patch.call_args.kwargs["body"]
        self.assertEqual(body, {"due": "2026-06-20T00:00:00.000Z"})

    def test_payload_contains_only_notes_when_only_notes_provided(self):
        service = self._make_patch_service({"id": "task_001"})

        update_task("task_001", notes="Updated notes", service=service)

        body = service.tasks.return_value.patch.call_args.kwargs["body"]
        self.assertEqual(body, {"notes": "Updated notes"})

    def test_payload_contains_all_fields_when_all_provided(self):
        service = self._make_patch_service({"id": "task_001"})

        update_task(
            "task_001",
            title="New Title",
            due_date="2026-06-20T00:00:00.000Z",
            notes="New notes",
            service=service,
        )

        body = service.tasks.return_value.patch.call_args.kwargs["body"]
        self.assertEqual(
            body,
            {
                "title": "New Title",
                "due": "2026-06-20T00:00:00.000Z",
                "notes": "New notes",
            },
        )

    def test_task_id_forwarded_to_api(self):
        service = self._make_patch_service({"id": "task_xyz"})

        update_task("task_xyz", title="X", service=service)

        call_kwargs = service.tasks.return_value.patch.call_args.kwargs
        self.assertEqual(call_kwargs["task"], "task_xyz")

    def test_default_tasklist_id_is_default(self):
        service = self._make_patch_service({"id": "task_001"})

        update_task("task_001", title="X", service=service)

        call_kwargs = service.tasks.return_value.patch.call_args.kwargs
        self.assertEqual(call_kwargs["tasklist"], "@default")

    def test_custom_tasklist_id_forwarded(self):
        service = self._make_patch_service({"id": "task_001"})

        update_task("task_001", title="X", service=service, tasklist_id="my-list-id")

        call_kwargs = service.tasks.return_value.patch.call_args.kwargs
        self.assertEqual(call_kwargs["tasklist"], "my-list-id")

    def test_raises_value_error_when_no_fields_provided(self):
        service = self._make_patch_service({"id": "task_001"})

        with self.assertRaises(ValueError):
            update_task("task_001", service=service)

        service.tasks.return_value.patch.assert_not_called()

    @patch("time.sleep", return_value=None)
    def test_retries_once_on_transient_error_and_returns_task(self, mock_sleep):
        service = self._make_patch_service_with_errors(
            [Exception("503 Service Unavailable")],
            success_response={"id": "task_001", "title": "Retried"},
        )

        task = update_task(
            "task_001", title="Retried", service=service, max_retries=2, base_backoff=1.0
        )

        self.assertEqual(task.task_title, "Retried")
        mock_sleep.assert_called_once_with(1.0)

    @patch("time.sleep", return_value=None)
    def test_raises_runtime_error_after_max_retries(self, mock_sleep):
        service = MagicMock()
        service.tasks.return_value.patch.return_value.execute = MagicMock(
            side_effect=Exception("Connection refused")
        )

        with self.assertRaises(RuntimeError) as ctx:
            update_task("task_001", title="X", service=service, max_retries=2, base_backoff=0.01)

        error_msg = str(ctx.exception)
        self.assertIn("Unable to update task", error_msg)
        self.assertIn("task_001", error_msg)
        self.assertIn("3 attempts", error_msg)
        self.assertIn("Connection refused", error_msg)


# ---------------------------------------------------------------------------
# Tests for complete_task
# ---------------------------------------------------------------------------


class TestCompleteTask(unittest.TestCase):
    @staticmethod
    def _make_patch_service(response: dict) -> MagicMock:
        service = MagicMock()
        service.tasks.return_value.patch.return_value.execute = MagicMock(
            return_value=response
        )
        return service

    @staticmethod
    def _make_patch_service_with_errors(
        errors: list[Exception], success_response: dict | None = None
    ) -> MagicMock:
        service = MagicMock()
        side_effects: list = list(errors)
        if success_response is not None:
            side_effects.append(success_response)
        service.tasks.return_value.patch.return_value.execute = MagicMock(
            side_effect=side_effects
        )
        return service

    def test_returns_task_item_with_completed_status(self):
        service = self._make_patch_service(
            {"id": "task_001", "title": "Buy milk", "status": "completed"}
        )

        task = complete_task("task_001", service=service)

        self.assertEqual(task.task_status, "completed")
        self.assertTrue(task.task_completed)

    def test_payload_sets_status_completed(self):
        service = self._make_patch_service({"id": "task_001", "status": "completed"})

        complete_task("task_001", service=service)

        body = service.tasks.return_value.patch.call_args.kwargs["body"]
        self.assertEqual(body, {"status": "completed"})

    def test_task_id_and_tasklist_forwarded(self):
        service = self._make_patch_service({"id": "task_xyz", "status": "completed"})

        complete_task("task_xyz", service=service, tasklist_id="my-list-id")

        call_kwargs = service.tasks.return_value.patch.call_args.kwargs
        self.assertEqual(call_kwargs["task"], "task_xyz")
        self.assertEqual(call_kwargs["tasklist"], "my-list-id")

    @patch("time.sleep", return_value=None)
    def test_retries_once_on_transient_error_and_returns_task(self, mock_sleep):
        service = self._make_patch_service_with_errors(
            [Exception("503 Service Unavailable")],
            success_response={"id": "task_001", "status": "completed"},
        )

        task = complete_task("task_001", service=service, max_retries=2, base_backoff=1.0)

        self.assertTrue(task.task_completed)
        mock_sleep.assert_called_once_with(1.0)

    @patch("time.sleep", return_value=None)
    def test_raises_runtime_error_after_max_retries(self, mock_sleep):
        service = MagicMock()
        service.tasks.return_value.patch.return_value.execute = MagicMock(
            side_effect=Exception("Connection refused")
        )

        with self.assertRaises(RuntimeError) as ctx:
            complete_task("task_001", service=service, max_retries=2, base_backoff=0.01)

        error_msg = str(ctx.exception)
        self.assertIn("Unable to complete task", error_msg)
        self.assertIn("task_001", error_msg)
        self.assertIn("3 attempts", error_msg)
        self.assertIn("Connection refused", error_msg)


# ---------------------------------------------------------------------------
# Tests for build_tasks_service
# ---------------------------------------------------------------------------


class TestBuildTasksService(unittest.TestCase):
    """Unit tests for the public build_tasks_service() function.

    All tests mock the Google API client libraries via sys.modules so that
    this test suite runs fully offline without requiring those packages to
    be installed.
    """

    def _make_google_mocks(
        self,
        *,
        creds_valid: bool = True,
        creds_expired: bool = False,
        creds_has_refresh_token: bool = False,
    ) -> tuple[dict, "MagicMock", "MagicMock", "MagicMock"]:
        mock_creds = MagicMock()
        mock_creds.valid = creds_valid
        mock_creds.expired = creds_expired
        mock_creds.refresh_token = "refresh_tok" if creds_has_refresh_token else None
        mock_creds.to_json.return_value = '{"token": "fake"}'

        mock_credentials_class = MagicMock()
        mock_credentials_class.from_authorized_user_file.return_value = mock_creds

        mock_flow_instance = MagicMock()
        mock_flow_instance.run_local_server.return_value = mock_creds
        mock_flow_class = MagicMock()
        mock_flow_class.from_client_secrets_file.return_value = mock_flow_instance

        mock_request_class = MagicMock()

        mock_service = MagicMock(name="TasksService")
        mock_build = MagicMock(return_value=mock_service)

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
        import sys
        import os

        with patch.dict(sys.modules, sys_modules_patch):
            token_path = os.path.join(tmp_path, "tasks_token.json")
            creds_path = os.path.join(tmp_path, "credentials.json")

            if token_file_exists:
                with open(token_path, "w") as fh:
                    fh.write('{"token": "fake"}')
            if credentials_file_exists:
                with open(creds_path, "w") as fh:
                    fh.write('{"installed": {}}')

            kwargs: dict = {
                "token_file": token_path if token_file_exists else token_path + ".missing",
            }
            kwargs["credentials_file"] = (
                creds_path if credentials_file_exists else creds_path + ".missing"
            )
            if extra_kwargs:
                kwargs.update(extra_kwargs)

            return build_tasks_service(**kwargs)

    def test_returns_service_object(self):
        import tempfile

        sys_modules_patch, _, _, mock_build = self._make_google_mocks()
        mock_service = mock_build.return_value

        with tempfile.TemporaryDirectory() as tmp:
            result = self._call_with_token_file(tmp, sys_modules_patch)

        self.assertIs(result, mock_service)

    def test_build_called_with_tasks_v1(self):
        import tempfile

        sys_modules_patch, _, _, mock_build = self._make_google_mocks()

        with tempfile.TemporaryDirectory() as tmp:
            self._call_with_token_file(tmp, sys_modules_patch)

        mock_build.assert_called_once()
        call_args = mock_build.call_args
        self.assertEqual(call_args.args[0], "tasks")
        self.assertEqual(call_args.args[1], "v1")

    def test_build_called_with_credentials_kwarg(self):
        import tempfile

        sys_modules_patch, _, mock_creds, mock_build = self._make_google_mocks()

        with tempfile.TemporaryDirectory() as tmp:
            self._call_with_token_file(tmp, sys_modules_patch)

        call_kwargs = mock_build.call_args.kwargs
        self.assertIs(call_kwargs["credentials"], mock_creds)

    def test_default_scope_is_tasks_access(self):
        import tempfile

        sys_modules_patch, mock_creds_class, _, _ = self._make_google_mocks()

        with tempfile.TemporaryDirectory() as tmp:
            self._call_with_token_file(tmp, sys_modules_patch)

        _, scopes = mock_creds_class.from_authorized_user_file.call_args.args
        self.assertEqual(scopes, ["https://www.googleapis.com/auth/tasks"])

    def test_custom_scopes_forwarded_to_credentials(self):
        import tempfile

        sys_modules_patch, mock_creds_class, _, _ = self._make_google_mocks()
        custom_scopes = ["https://www.googleapis.com/auth/tasks.readonly"]

        with tempfile.TemporaryDirectory() as tmp:
            self._call_with_token_file(
                tmp, sys_modules_patch, extra_kwargs={"scopes": custom_scopes}
            )

        _, scopes = mock_creds_class.from_authorized_user_file.call_args.args
        self.assertEqual(scopes, custom_scopes)

    def test_expired_credentials_are_refreshed(self):
        import tempfile

        sys_modules_patch, mock_creds_class, mock_creds, _ = self._make_google_mocks(
            creds_valid=False, creds_expired=True, creds_has_refresh_token=True
        )

        with tempfile.TemporaryDirectory() as tmp:
            self._call_with_token_file(tmp, sys_modules_patch)

        mock_creds.refresh.assert_called_once()

    def test_raises_runtime_error_when_credentials_file_missing(self):
        import tempfile

        sys_modules_patch, mock_creds_class, _, _ = self._make_google_mocks(
            creds_valid=False, creds_expired=False, creds_has_refresh_token=False
        )

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError) as ctx:
                self._call_with_token_file(
                    tmp,
                    sys_modules_patch,
                    token_file_exists=False,
                    credentials_file_exists=False,
                )

        self.assertIn("credentials file not found", str(ctx.exception))

    def test_raises_runtime_error_when_google_libs_not_installed(self):
        import sys

        # Simulate the google client libraries being unimportable.
        modules_to_block = [
            "google.oauth2.credentials",
            "google.auth.transport.requests",
            "google_auth_oauthlib.flow",
            "googleapiclient.discovery",
        ]
        patch_dict = {name: None for name in modules_to_block}

        with patch.dict(sys.modules, patch_dict):
            with self.assertRaises(RuntimeError) as ctx:
                build_tasks_service(token_file="/nonexistent/tasks_token.json")

        self.assertIn("Google API client libraries not installed", str(ctx.exception))

    def test_custom_token_file_path_used(self):
        import tempfile
        import os

        sys_modules_patch, mock_creds_class, _, _ = self._make_google_mocks()

        with tempfile.TemporaryDirectory() as tmp:
            custom_token = os.path.join(tmp, "custom_tasks_token.json")
            with open(custom_token, "w") as fh:
                fh.write('{"token": "fake"}')
            creds_path = os.path.join(tmp, "credentials.json")
            with open(creds_path, "w") as fh:
                fh.write("{}")

            with patch.dict(__import__("sys").modules, sys_modules_patch):
                build_tasks_service(token_file=custom_token, credentials_file=creds_path)

        path_arg, _ = mock_creds_class.from_authorized_user_file.call_args.args
        self.assertEqual(path_arg, custom_token)

    def test_build_called_exactly_once(self):
        import tempfile

        sys_modules_patch, _, _, mock_build = self._make_google_mocks()

        with tempfile.TemporaryDirectory() as tmp:
            self._call_with_token_file(tmp, sys_modules_patch)

        mock_build.assert_called_once()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
