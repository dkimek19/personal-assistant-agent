"""Daily database backup and 30-day purge (assistant.backup).

Implements:

- **AC18** -- :func:`backup_databases`: copy every SQLite database under
  ``~/assistant/data/`` into a timestamped subdirectory of
  ``~/assistant/backups/``, using the SQLite online backup API so the
  snapshot is consistent even while the source databases are open in WAL
  mode.
- **AC19** -- :func:`purge_old_backups`: delete timestamped backup
  directories older than 30 days.

:func:`run_daily_backup` runs both steps and is the entry point intended to
be invoked once per day (e.g. by a launchd job, see AC20).
"""

from __future__ import annotations

import logging
import shutil
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Directory containing the assistant's SQLite databases.
_DEFAULT_DATA_DIR = Path.home() / "assistant" / "data"

#: Directory where timestamped backup snapshots are written.
_DEFAULT_BACKUP_DIR = Path.home() / "assistant" / "backups"

#: AC19 -- backups older than this many days are purged.
_RETENTION_DAYS: int = 30

#: Format used for timestamped backup directory names.
_TIMESTAMP_FORMAT: str = "%Y%m%dT%H%M%SZ"


def _backup_one_db(db_path: Path, dest_path: Path) -> None:
    """Copy *db_path* to *dest_path* using the SQLite online backup API.

    Unlike a plain file copy, this produces a consistent snapshot even if
    *db_path* is open in WAL mode with uncheckpointed changes.

    Raises:
        sqlite3.Error: If *db_path* is not a valid SQLite database.
    """
    src_conn = sqlite3.connect(str(db_path))
    try:
        try:
            dest_conn = sqlite3.connect(str(dest_path))
            try:
                src_conn.backup(dest_conn)
            finally:
                dest_conn.close()
        except Exception:
            # sqlite3.connect() creates an empty file even if backup() then
            # fails (e.g. db_path is not a valid database) -- clean it up so
            # callers don't see a zero-byte "backup" of a failed source.
            dest_path.unlink(missing_ok=True)
            raise
    finally:
        src_conn.close()


def backup_databases(
    data_dir: str | Path = _DEFAULT_DATA_DIR,
    backup_dir: str | Path = _DEFAULT_BACKUP_DIR,
    *,
    now: datetime | None = None,
) -> Path:
    """Back up every ``*.db`` file in *data_dir* into a timestamped directory.

    Args:
        data_dir: Directory containing the SQLite databases to back up.
        backup_dir: Directory under which the timestamped backup directory
            is created.
        now: Timestamp to use for naming the backup directory. Defaults to
            the current UTC time.

    Returns:
        The path to the created (timestamped) backup directory. This is
        always created, even if *data_dir* contains no ``*.db`` files.

    Note:
        If an individual database cannot be backed up (e.g. it is corrupt),
        a warning is logged and the remaining databases are still backed up.
    """
    data_dir = Path(data_dir)
    backup_dir = Path(backup_dir)
    now = now or datetime.now(timezone.utc)

    target_dir = backup_dir / now.strftime(_TIMESTAMP_FORMAT)
    target_dir.mkdir(parents=True, exist_ok=True)

    db_files = sorted(data_dir.glob("*.db"))
    backed_up = 0
    for db_file in db_files:
        try:
            _backup_one_db(db_file, target_dir / db_file.name)
            backed_up += 1
        except Exception:
            logger.exception("backup_databases: failed to back up %s", db_file)

    logger.info(
        "backup_databases: backed up %d/%d database(s) from %s to %s",
        backed_up,
        len(db_files),
        data_dir,
        target_dir,
    )
    return target_dir


def purge_old_backups(
    backup_dir: str | Path = _DEFAULT_BACKUP_DIR,
    *,
    retention_days: int = _RETENTION_DAYS,
    now: datetime | None = None,
) -> list[Path]:
    """Delete timestamped backup directories older than *retention_days*.

    Args:
        backup_dir: Directory containing timestamped backup directories
            (as created by :func:`backup_databases`).
        retention_days: Backups older than this many days are deleted.
        now: Reference time for computing age. Defaults to the current UTC
            time.

    Returns:
        A list of paths that were deleted. Entries in *backup_dir* whose
        names do not match the expected timestamp format, or that are not
        directories, are left untouched and not included in the result.
    """
    backup_dir = Path(backup_dir)
    if not backup_dir.is_dir():
        return []

    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=retention_days)

    deleted: list[Path] = []
    for entry in sorted(backup_dir.iterdir()):
        if not entry.is_dir():
            continue
        try:
            entry_time = datetime.strptime(entry.name, _TIMESTAMP_FORMAT).replace(tzinfo=timezone.utc)
        except ValueError:
            continue

        if entry_time < cutoff:
            shutil.rmtree(entry)
            deleted.append(entry)
            logger.info("purge_old_backups: deleted %s (older than %d days)", entry, retention_days)

    return deleted


def run_daily_backup(
    data_dir: str | Path = _DEFAULT_DATA_DIR,
    backup_dir: str | Path = _DEFAULT_BACKUP_DIR,
    *,
    retention_days: int = _RETENTION_DAYS,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Run the daily backup-and-purge cycle (AC18 + AC19).

    Args:
        data_dir: Directory containing the SQLite databases to back up.
        backup_dir: Directory under which timestamped backups are stored.
        retention_days: Backups older than this many days are purged.
        now: Reference time for both the new backup's timestamp and the
            purge cutoff. Defaults to the current UTC time.

    Returns:
        A summary dict with keys ``"backup_path"`` (the directory just
        created) and ``"purged"`` (list of deleted backup directory paths).
    """
    now = now or datetime.now(timezone.utc)
    backup_path = backup_databases(data_dir, backup_dir, now=now)
    purged = purge_old_backups(backup_dir, retention_days=retention_days, now=now)
    return {"backup_path": backup_path, "purged": purged}


def main() -> None:
    """CLI entry point: run the daily backup-and-purge cycle and log a summary."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run_daily_backup()
    logger.info(
        "Daily backup complete: backed up to %s, purged %d old backup(s)",
        result["backup_path"],
        len(result["purged"]),
    )


if __name__ == "__main__":
    main()
