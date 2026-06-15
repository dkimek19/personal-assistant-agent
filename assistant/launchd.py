"""launchd jobs for the personal assistant agent (AC18-22).

Generates and installs macOS launchd ``LaunchAgent`` plists for:

- The main Web UI service (:mod:`assistant.main`), which automatically
  restarts if the process exits or crashes (``KeepAlive``), and starts on
  login as well (``RunAtLoad``) -- AC20.
- Three periodic, run-and-exit jobs, scheduled via ``StartCalendarInterval``
  or ``StartInterval`` rather than ``KeepAlive``:

  - :mod:`assistant.backup` -- daily database backup and 30-day purge
    (AC18/19).
  - :mod:`assistant.calendar_alerts` -- periodic check for upcoming
    calendar events to alert on via Telegram (AC21).
  - :mod:`assistant.disk_monitor` -- periodic disk-usage check, sending a
    Telegram warning when the threshold is reached (AC22).

Usage::

    >>> from assistant.launchd import install, install_all
    >>> install()  # just the main agent (AC20)
    PosixPath('/Users/you/Library/LaunchAgents/com.personalassistant.agent.plist')
    >>> install_all()  # main agent + all periodic jobs
    {'com.personalassistant.agent': PosixPath('...'), ...}

Then load a plist with launchd::

    launchctl load -w ~/Library/LaunchAgents/com.personalassistant.agent.plist
"""

from __future__ import annotations

import plistlib
import subprocess
import sys
from pathlib import Path
from typing import Any

#: launchd label identifying the main, always-running agent (AC20).
LABEL: str = "com.personalassistant.agent"

#: launchd label for the daily backup-and-purge job (AC18/19).
LABEL_BACKUP: str = "com.personalassistant.backup"

#: launchd label for the periodic calendar-alerts job (AC21).
LABEL_CALENDAR_ALERTS: str = "com.personalassistant.calendar-alerts"

#: launchd label for the periodic disk-usage monitor job (AC22).
LABEL_DISK_MONITOR: str = "com.personalassistant.disk-monitor"

#: Root directory of the project (used as the agent's working directory).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Default directory for the agent's stdout/stderr log files.
_DEFAULT_LOG_DIR = Path.home() / "assistant" / "logs"

#: AC21 -- how often (seconds) to check for due calendar alerts.
_CALENDAR_ALERTS_INTERVAL_SECONDS: int = 5 * 60

#: AC22 -- how often (seconds) to check disk usage.
_DISK_MONITOR_INTERVAL_SECONDS: int = 60 * 60

#: AC18/19 -- time of day (local time) the daily backup runs.
_BACKUP_START_CALENDAR_INTERVAL: dict[str, int] = {"Hour": 3, "Minute": 0}


def build_plist(
    *,
    label: str = LABEL,
    program_arguments: list[str | Path],
    working_directory: str | Path,
    stdout_path: str | Path,
    stderr_path: str | Path,
    run_at_load: bool = True,
    keep_alive: bool = True,
) -> dict[str, Any]:
    """Build a launchd plist dict for a long-running, auto-restarting agent.

    Args:
        label: The launchd job label (reverse-DNS style identifier).
        program_arguments: The command and arguments to run, e.g.
            ``[sys.executable, "-m", "assistant.main"]``.
        working_directory: Directory the process is launched from.
        stdout_path: File that the process's stdout is appended to.
        stderr_path: File that the process's stderr is appended to.
        run_at_load: If ``True``, launchd starts the job as soon as it is
            loaded (e.g. on login).
        keep_alive: If ``True``, launchd restarts the job whenever it exits,
            for any reason -- this is the auto-restart behaviour required by
            AC20.

    Returns:
        A dict suitable for serialisation with :mod:`plistlib`.
    """
    return {
        "Label": label,
        "ProgramArguments": [str(arg) for arg in program_arguments],
        "WorkingDirectory": str(working_directory),
        "RunAtLoad": run_at_load,
        "KeepAlive": keep_alive,
        "StandardOutPath": str(stdout_path),
        "StandardErrorPath": str(stderr_path),
    }


def generate_default_plist(
    *,
    project_dir: str | Path | None = None,
    log_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build the default plist for running :mod:`assistant.main` under launchd.

    Args:
        project_dir: Working directory for the agent. Defaults to this
            project's root directory.
        log_dir: Directory for stdout/stderr log files. Defaults to
            ``~/assistant/logs``.

    Returns:
        A plist dict, as returned by :func:`build_plist`.
    """
    project_dir = Path(project_dir) if project_dir is not None else _PROJECT_ROOT
    log_dir = Path(log_dir) if log_dir is not None else _DEFAULT_LOG_DIR

    return build_plist(
        program_arguments=[sys.executable, "-m", "assistant.main"],
        working_directory=project_dir,
        stdout_path=log_dir / "assistant.out.log",
        stderr_path=log_dir / "assistant.err.log",
    )


def build_periodic_plist(
    *,
    label: str,
    program_arguments: list[str | Path],
    working_directory: str | Path,
    stdout_path: str | Path,
    stderr_path: str | Path,
    start_interval: int | None = None,
    start_calendar_interval: dict[str, int] | list[dict[str, int]] | None = None,
    run_at_load: bool = False,
) -> dict[str, Any]:
    """Build a launchd plist dict for a periodic, run-and-exit job.

    Unlike :func:`build_plist`, periodic jobs are not kept alive -- launchd
    runs the command to completion on the configured schedule rather than
    restarting it whenever it exits.

    Args:
        label: The launchd job label (reverse-DNS style identifier).
        program_arguments: The command and arguments to run, e.g.
            ``[sys.executable, "-m", "assistant.backup"]``.
        working_directory: Directory the process is launched from.
        stdout_path: File that the process's stdout is appended to.
        stderr_path: File that the process's stderr is appended to.
        start_interval: If given, launchd runs the job every
            *start_interval* seconds (``StartInterval``).
        start_calendar_interval: If given, launchd runs the job at the
            specified calendar time(s) (``StartCalendarInterval``), e.g.
            ``{"Hour": 3, "Minute": 0}`` for daily at 03:00.
        run_at_load: If ``True``, also run the job immediately when loaded.

    Returns:
        A dict suitable for serialisation with :mod:`plistlib`.

    Raises:
        ValueError: Unless exactly one of *start_interval* or
            *start_calendar_interval* is provided.
    """
    if (start_interval is None) == (start_calendar_interval is None):
        raise ValueError(
            "exactly one of start_interval or start_calendar_interval must be provided"
        )

    plist: dict[str, Any] = {
        "Label": label,
        "ProgramArguments": [str(arg) for arg in program_arguments],
        "WorkingDirectory": str(working_directory),
        "RunAtLoad": run_at_load,
        "StandardOutPath": str(stdout_path),
        "StandardErrorPath": str(stderr_path),
    }

    if start_interval is not None:
        plist["StartInterval"] = start_interval
    else:
        plist["StartCalendarInterval"] = start_calendar_interval

    return plist


def generate_backup_plist(
    *,
    project_dir: str | Path | None = None,
    log_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build the plist for the daily backup-and-purge job (AC18/19).

    Runs ``python -m assistant.backup`` once a day (see
    :data:`_BACKUP_START_CALENDAR_INTERVAL`).

    Args:
        project_dir: Working directory for the job. Defaults to this
            project's root directory.
        log_dir: Directory for stdout/stderr log files. Defaults to
            ``~/assistant/logs``.

    Returns:
        A plist dict, as returned by :func:`build_periodic_plist`.
    """
    project_dir = Path(project_dir) if project_dir is not None else _PROJECT_ROOT
    log_dir = Path(log_dir) if log_dir is not None else _DEFAULT_LOG_DIR

    return build_periodic_plist(
        label=LABEL_BACKUP,
        program_arguments=[sys.executable, "-m", "assistant.backup"],
        working_directory=project_dir,
        stdout_path=log_dir / "backup.out.log",
        stderr_path=log_dir / "backup.err.log",
        start_calendar_interval=dict(_BACKUP_START_CALENDAR_INTERVAL),
    )


def generate_calendar_alerts_plist(
    *,
    project_dir: str | Path | None = None,
    log_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build the plist for the periodic calendar-alerts job (AC21).

    Runs ``python -m assistant.calendar_alerts`` every
    :data:`_CALENDAR_ALERTS_INTERVAL_SECONDS` seconds.

    Args:
        project_dir: Working directory for the job. Defaults to this
            project's root directory.
        log_dir: Directory for stdout/stderr log files. Defaults to
            ``~/assistant/logs``.

    Returns:
        A plist dict, as returned by :func:`build_periodic_plist`.
    """
    project_dir = Path(project_dir) if project_dir is not None else _PROJECT_ROOT
    log_dir = Path(log_dir) if log_dir is not None else _DEFAULT_LOG_DIR

    return build_periodic_plist(
        label=LABEL_CALENDAR_ALERTS,
        program_arguments=[sys.executable, "-m", "assistant.calendar_alerts"],
        working_directory=project_dir,
        stdout_path=log_dir / "calendar_alerts.out.log",
        stderr_path=log_dir / "calendar_alerts.err.log",
        start_interval=_CALENDAR_ALERTS_INTERVAL_SECONDS,
    )


def generate_disk_monitor_plist(
    *,
    project_dir: str | Path | None = None,
    log_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build the plist for the periodic disk-usage monitor job (AC22).

    Runs ``python -m assistant.disk_monitor`` every
    :data:`_DISK_MONITOR_INTERVAL_SECONDS` seconds.

    Args:
        project_dir: Working directory for the job. Defaults to this
            project's root directory.
        log_dir: Directory for stdout/stderr log files. Defaults to
            ``~/assistant/logs``.

    Returns:
        A plist dict, as returned by :func:`build_periodic_plist`.
    """
    project_dir = Path(project_dir) if project_dir is not None else _PROJECT_ROOT
    log_dir = Path(log_dir) if log_dir is not None else _DEFAULT_LOG_DIR

    return build_periodic_plist(
        label=LABEL_DISK_MONITOR,
        program_arguments=[sys.executable, "-m", "assistant.disk_monitor"],
        working_directory=project_dir,
        stdout_path=log_dir / "disk_monitor.out.log",
        stderr_path=log_dir / "disk_monitor.err.log",
        start_interval=_DISK_MONITOR_INTERVAL_SECONDS,
    )


def default_plist_path(label: str = LABEL) -> Path:
    """Return the standard install location for a per-user launchd agent plist."""
    return Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"


def write_plist(plist: dict[str, Any], dest_path: str | Path) -> Path:
    """Serialise *plist* to *dest_path* as an XML plist.

    Args:
        plist: The plist contents, as returned by :func:`build_plist`.
        dest_path: File to write. Parent directories are created if needed.

    Returns:
        *dest_path*, as a :class:`~pathlib.Path`.
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with dest_path.open("wb") as f:
        plistlib.dump(plist, f)
    return dest_path


def install(
    *,
    dest_path: str | Path | None = None,
    project_dir: str | Path | None = None,
    log_dir: str | Path | None = None,
) -> Path:
    """Generate the default plist and write it to *dest_path*.

    Also creates the log directory referenced by the plist's
    ``StandardOutPath``/``StandardErrorPath``, since launchd does not create
    missing directories for these files.

    Args:
        dest_path: Where to write the plist. Defaults to
            :func:`default_plist_path`.
        project_dir: Forwarded to :func:`generate_default_plist`.
        log_dir: Forwarded to :func:`generate_default_plist`.

    Returns:
        The path the plist was written to.
    """
    plist = generate_default_plist(project_dir=project_dir, log_dir=log_dir)
    dest = Path(dest_path) if dest_path is not None else default_plist_path()

    Path(plist["StandardOutPath"]).parent.mkdir(parents=True, exist_ok=True)
    return write_plist(plist, dest)


def install_all(
    *,
    dest_dir: str | Path | None = None,
    project_dir: str | Path | None = None,
    log_dir: str | Path | None = None,
    load: bool = True,
) -> dict[str, Path]:
    """Generate, write, and (optionally) load every launchd job.

    Installs the main auto-restart agent (AC20, see :func:`install`) plus
    the three periodic jobs: daily backup (AC18/19), calendar alerts
    (AC21), and disk-usage monitoring (AC22).

    Also creates the log directory referenced by each plist's
    ``StandardOutPath``/``StandardErrorPath``, since launchd does not create
    missing directories for these files.

    Args:
        dest_dir: Directory to write the plists into. Defaults to the
            standard per-user ``~/Library/LaunchAgents`` directory.
        project_dir: Forwarded to each plist generator.
        log_dir: Forwarded to each plist generator.
        load: If ``True``, also call :func:`load_agent` on each plist after
            writing it.

    Returns:
        A dict mapping each job's launchd label to the path its plist was
        written to.
    """
    dest_dir = Path(dest_dir) if dest_dir is not None else default_plist_path().parent

    generators = {
        LABEL: generate_default_plist,
        LABEL_BACKUP: generate_backup_plist,
        LABEL_CALENDAR_ALERTS: generate_calendar_alerts_plist,
        LABEL_DISK_MONITOR: generate_disk_monitor_plist,
    }

    paths: dict[str, Path] = {}
    for label, generate in generators.items():
        plist = generate(project_dir=project_dir, log_dir=log_dir)
        Path(plist["StandardOutPath"]).parent.mkdir(parents=True, exist_ok=True)
        path = write_plist(plist, dest_dir / f"{label}.plist")
        paths[label] = path
        if load:
            load_agent(path)

    return paths


def load_agent(plist_path: str | Path) -> None:
    """Load (and enable) the agent at *plist_path* via ``launchctl load -w``."""
    subprocess.run(["launchctl", "load", "-w", str(plist_path)], check=True)


def unload_agent(plist_path: str | Path) -> None:
    """Unload (and disable) the agent at *plist_path* via ``launchctl unload -w``."""
    subprocess.run(["launchctl", "unload", "-w", str(plist_path)], check=True)
