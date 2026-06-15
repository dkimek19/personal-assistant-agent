"""Disk usage warning via Telegram (AC22).

Checks the total size of the assistant's data directory (``~/assistant``,
which holds ``memory.db``/``user.db`` and the daily backups created by
:mod:`assistant.backup` -- AC18/19) and sends a Telegram warning via
:func:`assistant.interfaces.telegram_bot.send_message` once its size reaches
a configurable threshold (default 20GB).

The warning is sent only once per threshold breach: once sent, it is not
repeated on subsequent checks until the directory size drops back below the
threshold (e.g. after old backups are purged).

:func:`run_disk_usage_check` is the entry point intended to be invoked
periodically (e.g. daily by a launchd job, see AC20).
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Awaitable, Callable

from assistant.interfaces.telegram_bot import send_message

logger = logging.getLogger(__name__)

#: Directory whose total size is monitored.
_DEFAULT_MONITORED_DIR = Path.home() / "assistant"

#: AC22 -- warn once total usage reaches this many bytes (20GB).
_DEFAULT_THRESHOLD_BYTES = 20 * 1024 ** 3

#: Marker file recording that the warning has already been sent for the
#: current threshold breach. Removed once usage drops back below threshold.
_DEFAULT_STATE_PATH = Path.home() / "assistant" / "data" / "disk_warning_sent.flag"

_BYTES_PER_GB = 1024 ** 3


def get_directory_size(path: str | Path) -> int:
    """Return the total size in bytes of all files under *path* (recursive).

    Args:
        path: Directory to measure. If it does not exist, ``0`` is returned.

    Returns:
        The sum of ``st_size`` for every regular file found by recursively
        walking *path*. Entries that disappear or cannot be stat'd while
        walking are skipped.
    """
    path = Path(path)
    if not path.exists():
        return 0

    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            try:
                total += entry.stat().st_size
            except OSError:
                continue
    return total


def format_disk_warning_message(size_bytes: int, threshold_bytes: int, path: str | Path) -> str:
    """Return the Telegram warning text for a disk usage threshold breach."""
    size_gb = size_bytes / _BYTES_PER_GB
    threshold_gb = threshold_bytes / _BYTES_PER_GB
    return (
        f"Disk usage warning: {path} is using {size_gb:.2f} GB, "
        f"which has reached the {threshold_gb:.0f} GB threshold."
    )


class DiskWarningState:
    """Tracks whether the disk usage warning has already been sent for the
    current threshold breach.

    Args:
        state_path: Path to the marker file. Defaults to
            ``~/assistant/data/disk_warning_sent.flag``. The parent
            directory is created automatically if it does not exist.
    """

    def __init__(self, state_path: str | Path | None = None) -> None:
        if state_path is None:
            state_path = _DEFAULT_STATE_PATH
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    def was_sent(self) -> bool:
        """Return ``True`` if the warning has already been sent and not yet cleared."""
        return self.state_path.exists()

    def mark_sent(self) -> None:
        """Record that the warning has been sent."""
        self.state_path.touch()

    def clear(self) -> None:
        """Clear the sent marker (e.g. once usage drops back below threshold)."""
        self.state_path.unlink(missing_ok=True)


async def run_disk_usage_check(
    *,
    path: str | Path = _DEFAULT_MONITORED_DIR,
    threshold_bytes: int = _DEFAULT_THRESHOLD_BYTES,
    chat_id: int | str | None = None,
    state: DiskWarningState | None = None,
    token: str | None = None,
    send: Callable[..., Awaitable[None]] = send_message,
) -> dict[str, Any]:
    """Check *path*'s total size and send a Telegram warning if over threshold (AC22).

    Args:
        path: Directory to measure. Defaults to ``~/assistant``.
        threshold_bytes: Size in bytes that triggers the warning. Defaults
            to 20GB.
        chat_id: The Telegram chat ID to send the warning to. Defaults to
            the ``TELEGRAM_CHAT_ID`` environment variable.
        state: Tracks whether the warning has already been sent for the
            current breach. Defaults to a new :class:`DiskWarningState`.
        token: Telegram bot token, forwarded to *send*.
        send: The function used to deliver the warning message. Defaults to
            :func:`assistant.interfaces.telegram_bot.send_message`.

    Returns:
        A summary dict with keys ``"path"``, ``"size_bytes"``,
        ``"threshold_bytes"``, ``"exceeded"`` (bool), and ``"sent"`` (bool,
        whether a warning was sent during this call).

    Raises:
        RuntimeError: If the threshold is exceeded, no warning has been sent
            yet, and no *chat_id* is given and ``TELEGRAM_CHAT_ID`` is not
            set.
    """
    path = Path(path)
    state = state or DiskWarningState()

    size_bytes = get_directory_size(path)
    exceeded = size_bytes >= threshold_bytes

    result: dict[str, Any] = {
        "path": path,
        "size_bytes": size_bytes,
        "threshold_bytes": threshold_bytes,
        "exceeded": exceeded,
        "sent": False,
    }

    if not exceeded:
        state.clear()
        return result

    if state.was_sent():
        return result

    resolved_chat_id = chat_id if chat_id is not None else os.environ.get("TELEGRAM_CHAT_ID")
    if not resolved_chat_id:
        raise RuntimeError("TELEGRAM_CHAT_ID environment variable is not set")

    await send(resolved_chat_id, format_disk_warning_message(size_bytes, threshold_bytes, path), token=token)
    state.mark_sent()
    result["sent"] = True
    logger.info("run_disk_usage_check: sent disk usage warning for %s (%d bytes)", path, size_bytes)

    return result


def main() -> None:
    """CLI entry point: run a single disk usage check and log a summary."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = asyncio.run(run_disk_usage_check())
    logger.info(
        "Disk usage check complete: %s is using %d bytes (threshold %d bytes), warning sent: %s",
        result["path"],
        result["size_bytes"],
        result["threshold_bytes"],
        result["sent"],
    )


if __name__ == "__main__":
    main()
