"""Docker sandbox code execution tool (assistant.tools.code_execution).

Provides one public function, :func:`execute_code`, which runs
user-submitted code inside an ephemeral, network-disabled Docker container
and returns the captured stdout/stderr/exit code.

Retry policy
------------
Two distinct failure modes are handled differently:

- **Docker daemon / container-creation failures** (e.g. the daemon is
  briefly unreachable, an image pull races) occur in
  ``client.containers.run(...)``. These are transient infrastructure
  problems, so they use the project's standard retry-with-backoff policy
  (``max_retries=2``, ``base_backoff=1.0``, doubling). After exhaustion a
  :class:`RuntimeError` is raised.
- **Execution timeouts** (the submitted code itself runs too long) occur in
  ``container.wait(timeout=...)``. Re-running code that hangs would just
  hang again, so this is *not* retried: the container is killed, its
  partial output is captured, and a :class:`CodeExecutionResult` with
  ``timed_out=True`` is returned normally (this is a successful *tool*
  invocation that reports the program did not finish in time).

Security
--------
Containers run with ``network_disabled=True``, a memory limit
(``mem_limit``, default ``"256m"``), and are always removed
(``remove(force=True)``) after execution.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_MAX_RETRIES: int = 2
_BACKOFF_BASE_SECONDS: float = 1.0
_DEFAULT_TIMEOUT_SECONDS: int = 30
_DEFAULT_MEM_LIMIT: str = "256m"

# Supported execution languages: image to run and the command prefix used to
# execute the submitted code as the final argument.
_SUPPORTED_LANGUAGES: dict[str, dict[str, Any]] = {
    "python": {"image": "python:3.11-slim", "command": ["python3", "-c"]},
    "bash": {"image": "bash:5", "command": ["bash", "-c"]},
}


@dataclass
class CodeExecutionResult:
    """Result of running code inside the Docker sandbox."""

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool

    def to_dict(self) -> dict[str, Any]:
        """Return this result as a plain dict."""
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
        }


def _build_docker_client() -> Any:
    """Return a connected Docker SDK client.

    Raises:
        RuntimeError: If the ``docker`` package is not installed, or the
            Docker daemon cannot be reached.
    """
    try:
        import docker
    except ImportError as exc:
        raise RuntimeError(
            "Docker SDK not installed. Install it with: pip install docker"
        ) from exc

    try:
        return docker.from_env()
    except Exception as exc:
        raise RuntimeError(
            f"Unable to connect to the Docker daemon: {exc}. "
            "Please ensure Docker is installed and running."
        ) from exc


def _run_container(
    client: Any,
    image: str,
    command: list[str],
    *,
    timeout: int,
    mem_limit: str,
) -> CodeExecutionResult:
    """Run *command* in a fresh, network-disabled container and capture output."""
    container = client.containers.run(
        image,
        command,
        detach=True,
        network_disabled=True,
        mem_limit=mem_limit,
    )
    try:
        try:
            wait_result = container.wait(timeout=timeout)
            exit_code = int(wait_result.get("StatusCode", -1))
            timed_out = False
        except Exception:
            try:
                container.kill()
            except Exception:
                pass
            exit_code = -1
            timed_out = True

        stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
        stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")
    finally:
        try:
            container.remove(force=True)
        except Exception:
            logger.warning("execute_code: failed to remove container %r", container)

    return CodeExecutionResult(
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        timed_out=timed_out,
    )


def execute_code(
    code: str,
    *,
    language: str = "python",
    timeout: int = _DEFAULT_TIMEOUT_SECONDS,
    mem_limit: str = _DEFAULT_MEM_LIMIT,
    client: Any | None = None,
    max_retries: int = _MAX_RETRIES,
    base_backoff: float = _BACKOFF_BASE_SECONDS,
) -> CodeExecutionResult:
    """Run *code* inside a Docker sandbox and return its output.

    Args:
        code: Source code to execute. Must be non-empty.
        language: One of the supported languages (currently ``"python"``
            or ``"bash"``). Defaults to ``"python"``.
        timeout: Maximum seconds to wait for the code to finish before it
            is killed and ``timed_out=True`` is reported. Defaults to 30.
        mem_limit: Container memory limit, e.g. ``"256m"``.
        client: An optional pre-built Docker SDK client (primarily for
            tests). If ``None``, one is built via :func:`_build_docker_client`.
        max_retries: Number of retries for transient Docker
            daemon/container-creation failures.
        base_backoff: Base delay (seconds) for exponential backoff between
            retries; doubles on each subsequent attempt.

    Returns:
        A :class:`CodeExecutionResult` with the captured stdout, stderr,
        exit code, and whether execution was killed for exceeding *timeout*.

    Raises:
        ValueError: If *code* is empty/whitespace-only, or *language* is
            not supported.
        RuntimeError: If the Docker daemon cannot be reached after
            exhausting retries (or is unavailable/not installed).
    """
    if not code or not code.strip():
        raise ValueError("execute_code() requires a non-empty 'code' string")
    if language not in _SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language: {language!r}. Supported languages: "
            f"{sorted(_SUPPORTED_LANGUAGES)}"
        )

    if client is None:
        client = _build_docker_client()

    spec = _SUPPORTED_LANGUAGES[language]
    command = [*spec["command"], code]
    image = spec["image"]

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        if attempt > 0:
            delay = base_backoff * (2 ** (attempt - 1))
            logger.warning(
                "execute_code: attempt %d failed (%s), retrying in %.1fs",
                attempt,
                last_error,
                delay,
            )
            time.sleep(delay)
        try:
            return _run_container(client, image, command, timeout=timeout, mem_limit=mem_limit)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Unable to execute code in the Docker sandbox after {max_retries + 1} attempts. "
        f"Last error: {last_error}. Please ensure the Docker daemon is running."
    )
