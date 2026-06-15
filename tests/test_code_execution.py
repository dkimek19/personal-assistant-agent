"""Tests for the Docker sandbox code execution tool (assistant.tools.code_execution).

Covers:
- CodeExecutionResult.to_dict()
- _build_docker_client: SDK not installed, daemon unreachable, success
- _run_container: success (exit 0), non-zero exit code, execution timeout
  (kill + partial output captured), container always removed even if
  kill()/remove() themselves raise
- execute_code:
  - input validation (empty code, unsupported language)
  - default client built via _build_docker_client when client=None
  - success path returns a CodeExecutionResult (python and bash languages)
  - execution timeouts are returned (not retried)
  - transient containers.run errors are retried with exponential backoff
    (sleep 1.0, then 2.0)
  - RuntimeError raised after max_retries exhausted, message mentions the
    attempt count and underlying error
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from assistant.tools.code_execution import (
    CodeExecutionResult,
    _build_docker_client,
    _run_container,
    execute_code,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_logs_side_effect(stdout_bytes: bytes = b"", stderr_bytes: bytes = b""):
    """Return a side_effect function for ``container.logs(stdout=, stderr=)``."""

    def _logs(*, stdout: bool = True, stderr: bool = True, **kwargs):
        if stdout and not stderr:
            return stdout_bytes
        if stderr and not stdout:
            return stderr_bytes
        return stdout_bytes + stderr_bytes

    return _logs


def _make_client(container: MagicMock) -> MagicMock:
    client = MagicMock()
    client.containers.run.return_value = container
    return client


# ---------------------------------------------------------------------------
# Tests for CodeExecutionResult
# ---------------------------------------------------------------------------


class TestCodeExecutionResult:
    def test_to_dict_contains_all_keys(self):
        result = CodeExecutionResult(stdout="out", stderr="err", exit_code=0, timed_out=False)

        assert result.to_dict() == {
            "stdout": "out",
            "stderr": "err",
            "exit_code": 0,
            "timed_out": False,
        }


# ---------------------------------------------------------------------------
# Tests for _build_docker_client
# ---------------------------------------------------------------------------


class TestBuildDockerClient:
    def test_raises_runtime_error_if_docker_sdk_not_installed(self):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "docker":
                raise ImportError("No module named 'docker'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with pytest.raises(RuntimeError) as exc_info:
                _build_docker_client()

        assert "not installed" in str(exc_info.value)

    @patch("docker.from_env")
    def test_raises_runtime_error_if_daemon_unreachable(self, mock_from_env):
        mock_from_env.side_effect = Exception("Cannot connect to the Docker daemon")

        with pytest.raises(RuntimeError) as exc_info:
            _build_docker_client()

        assert "Unable to connect to the Docker daemon" in str(exc_info.value)

    @patch("docker.from_env")
    def test_returns_client_from_docker_from_env(self, mock_from_env):
        mock_client = MagicMock()
        mock_from_env.return_value = mock_client

        client = _build_docker_client()

        assert client is mock_client


# ---------------------------------------------------------------------------
# Tests for _run_container
# ---------------------------------------------------------------------------


class TestRunContainer:
    def test_success_returns_captured_output(self):
        container = MagicMock()
        container.wait.return_value = {"StatusCode": 0}
        container.logs.side_effect = _make_logs_side_effect(b"hello\n", b"")
        client = _make_client(container)

        result = _run_container(
            client, "python:3.11-slim", ["python3", "-c", "print('hello')"],
            timeout=30, mem_limit="256m",
        )

        assert result.stdout == "hello\n"
        assert result.stderr == ""
        assert result.exit_code == 0
        assert result.timed_out is False
        container.remove.assert_called_once_with(force=True)

    def test_nonzero_exit_code_captured_without_error(self):
        container = MagicMock()
        container.wait.return_value = {"StatusCode": 1}
        container.logs.side_effect = _make_logs_side_effect(b"", b"Traceback...\n")
        client = _make_client(container)

        result = _run_container(
            client, "python:3.11-slim", ["python3", "-c", "raise ValueError()"],
            timeout=30, mem_limit="256m",
        )

        assert result.exit_code == 1
        assert result.timed_out is False
        assert "Traceback" in result.stderr

    def test_timeout_kills_container_and_returns_partial_output(self):
        container = MagicMock()
        container.wait.side_effect = Exception("client read timeout")
        container.logs.side_effect = _make_logs_side_effect(b"partial output\n", b"")
        client = _make_client(container)

        result = _run_container(
            client, "python:3.11-slim", ["python3", "-c", "while True: pass"],
            timeout=1, mem_limit="256m",
        )

        assert result.timed_out is True
        assert result.exit_code == -1
        assert result.stdout == "partial output\n"
        container.kill.assert_called_once()
        container.remove.assert_called_once_with(force=True)

    def test_container_removed_even_if_kill_raises(self):
        container = MagicMock()
        container.wait.side_effect = Exception("client read timeout")
        container.kill.side_effect = Exception("already exited")
        container.logs.side_effect = _make_logs_side_effect(b"", b"")
        client = _make_client(container)

        result = _run_container(
            client, "python:3.11-slim", ["python3", "-c", "..."],
            timeout=1, mem_limit="256m",
        )

        assert result.timed_out is True
        container.remove.assert_called_once_with(force=True)

    def test_result_returned_even_if_remove_raises(self):
        container = MagicMock()
        container.wait.return_value = {"StatusCode": 0}
        container.logs.side_effect = _make_logs_side_effect(b"ok\n", b"")
        container.remove.side_effect = Exception("remove failed")
        client = _make_client(container)

        result = _run_container(
            client, "python:3.11-slim", ["python3", "-c", "print('ok')"],
            timeout=30, mem_limit="256m",
        )

        assert result.stdout == "ok\n"
        assert result.exit_code == 0

    def test_run_called_with_network_disabled_and_mem_limit(self):
        container = MagicMock()
        container.wait.return_value = {"StatusCode": 0}
        container.logs.side_effect = _make_logs_side_effect(b"", b"")
        client = _make_client(container)

        _run_container(
            client, "python:3.11-slim", ["python3", "-c", "pass"],
            timeout=30, mem_limit="128m",
        )

        call_kwargs = client.containers.run.call_args.kwargs
        assert call_kwargs["network_disabled"] is True
        assert call_kwargs["mem_limit"] == "128m"
        assert call_kwargs["detach"] is True


# ---------------------------------------------------------------------------
# Tests for execute_code
# ---------------------------------------------------------------------------


class TestExecuteCode:
    def test_empty_code_raises_value_error(self):
        with pytest.raises(ValueError):
            execute_code("", client=MagicMock())

    def test_whitespace_only_code_raises_value_error(self):
        with pytest.raises(ValueError):
            execute_code("   ", client=MagicMock())

    def test_unsupported_language_raises_value_error(self):
        with pytest.raises(ValueError) as exc_info:
            execute_code("print(1)", language="ruby", client=MagicMock())

        assert "ruby" in str(exc_info.value)

    def test_success_returns_code_execution_result(self):
        container = MagicMock()
        container.wait.return_value = {"StatusCode": 0}
        container.logs.side_effect = _make_logs_side_effect(b"42\n", b"")
        client = _make_client(container)

        result = execute_code("print(40 + 2)", client=client)

        assert isinstance(result, CodeExecutionResult)
        assert result.stdout == "42\n"
        assert result.exit_code == 0

    def test_python_image_and_command_used_by_default(self):
        container = MagicMock()
        container.wait.return_value = {"StatusCode": 0}
        container.logs.side_effect = _make_logs_side_effect(b"", b"")
        client = _make_client(container)

        execute_code("print('hi')", client=client)

        args = client.containers.run.call_args.args
        assert args[0] == "python:3.11-slim"
        assert args[1] == ["python3", "-c", "print('hi')"]

    def test_bash_language_uses_bash_image_and_command(self):
        container = MagicMock()
        container.wait.return_value = {"StatusCode": 0}
        container.logs.side_effect = _make_logs_side_effect(b"", b"")
        client = _make_client(container)

        execute_code("echo hi", language="bash", client=client)

        args = client.containers.run.call_args.args
        assert args[0] == "bash:5"
        assert args[1] == ["bash", "-c", "echo hi"]

    @patch("assistant.tools.code_execution._build_docker_client")
    def test_default_client_built_when_not_provided(self, mock_build_client):
        container = MagicMock()
        container.wait.return_value = {"StatusCode": 0}
        container.logs.side_effect = _make_logs_side_effect(b"", b"")
        client = _make_client(container)
        mock_build_client.return_value = client

        execute_code("print(1)")

        mock_build_client.assert_called_once()

    def test_execution_timeout_is_returned_not_retried(self):
        container = MagicMock()
        container.wait.side_effect = Exception("client read timeout")
        container.logs.side_effect = _make_logs_side_effect(b"partial\n", b"")
        client = _make_client(container)

        result = execute_code("while True: pass", client=client, timeout=1)

        assert result.timed_out is True
        assert result.stdout == "partial\n"
        assert client.containers.run.call_count == 1

    @patch("time.sleep", return_value=None)
    def test_retries_on_transient_run_error_and_succeeds(self, mock_sleep):
        container = MagicMock()
        container.wait.return_value = {"StatusCode": 0}
        container.logs.side_effect = _make_logs_side_effect(b"ok\n", b"")
        client = MagicMock()
        client.containers.run.side_effect = [
            Exception("daemon temporarily unavailable"),
            container,
        ]

        result = execute_code("print('ok')", client=client, max_retries=2, base_backoff=1.0)

        assert result.stdout == "ok\n"
        mock_sleep.assert_called_once_with(1.0)

    @patch("time.sleep", return_value=None)
    def test_second_retry_uses_doubled_backoff(self, mock_sleep):
        container = MagicMock()
        container.wait.return_value = {"StatusCode": 0}
        container.logs.side_effect = _make_logs_side_effect(b"ok\n", b"")
        client = MagicMock()
        client.containers.run.side_effect = [
            Exception("err1"),
            Exception("err2"),
            container,
        ]

        result = execute_code("print('ok')", client=client, max_retries=2, base_backoff=1.0)

        assert result.stdout == "ok\n"
        assert mock_sleep.call_count == 2
        calls = mock_sleep.call_args_list
        assert calls[0][0][0] == 1.0
        assert calls[1][0][0] == 2.0

    @patch("time.sleep", return_value=None)
    def test_raises_runtime_error_after_max_retries(self, mock_sleep):
        client = MagicMock()
        client.containers.run.side_effect = Exception("daemon unreachable")

        with pytest.raises(RuntimeError) as exc_info:
            execute_code("print('x')", client=client, max_retries=2, base_backoff=0.01)

        error_msg = str(exc_info.value)
        assert "Unable to execute code in the Docker sandbox" in error_msg
        assert "3 attempts" in error_msg
        assert "daemon unreachable" in error_msg

    @patch("time.sleep", return_value=None)
    def test_run_called_exactly_max_retries_plus_one_times(self, mock_sleep):
        client = MagicMock()
        client.containers.run.side_effect = Exception("daemon unreachable")

        with pytest.raises(RuntimeError):
            execute_code("print('x')", client=client, max_retries=2, base_backoff=0.01)

        assert client.containers.run.call_count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
