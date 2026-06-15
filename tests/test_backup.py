"""Tests for daily database backup + 30-day purge (assistant.backup).

Covers:
- backup_databases: creates a timestamped directory, copies all *.db files
  via the SQLite backup API (verified by reading data back from the copy,
  including a consistent snapshot under WAL mode), ignores non-.db files,
  handles an empty data_dir, and does not abort on a corrupt source DB.
- purge_old_backups: deletes timestamped directories older than
  retention_days, keeps recent/boundary ones, ignores non-matching names
  and non-directory entries, handles a missing backup_dir.
- run_daily_backup: orchestrates backup + purge and returns a summary dict.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from assistant.backup import backup_databases, purge_old_backups, run_daily_backup


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_db(path, values=()):
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, value TEXT)")
    for value in values:
        conn.execute("INSERT INTO items (value) VALUES (?)", (value,))
    conn.commit()
    conn.close()


def _read_values(path):
    conn = sqlite3.connect(str(path))
    rows = conn.execute("SELECT value FROM items ORDER BY id").fetchall()
    conn.close()
    return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# Tests for backup_databases (AC18)
# ---------------------------------------------------------------------------


class TestBackupDatabases:
    def test_creates_timestamped_directory(self, tmp_path):
        data_dir = tmp_path / "data"
        backup_dir = tmp_path / "backups"
        data_dir.mkdir()
        _make_db(data_dir / "memory.db", ["a"])

        now = datetime(2026, 6, 10, 3, 0, 0, tzinfo=timezone.utc)
        target = backup_databases(data_dir, backup_dir, now=now)

        assert target == backup_dir / "20260610T030000Z"
        assert target.is_dir()

    def test_backs_up_all_db_files(self, tmp_path):
        data_dir = tmp_path / "data"
        backup_dir = tmp_path / "backups"
        data_dir.mkdir()
        _make_db(data_dir / "memory.db", ["a"])
        _make_db(data_dir / "user.db", ["x", "y"])

        target = backup_databases(data_dir, backup_dir)

        assert (target / "memory.db").is_file()
        assert (target / "user.db").is_file()

    def test_backed_up_db_contains_same_data(self, tmp_path):
        data_dir = tmp_path / "data"
        backup_dir = tmp_path / "backups"
        data_dir.mkdir()
        _make_db(data_dir / "memory.db", ["alpha", "beta"])

        target = backup_databases(data_dir, backup_dir)

        assert _read_values(target / "memory.db") == ["alpha", "beta"]

    def test_consistent_snapshot_under_wal_mode(self, tmp_path):
        data_dir = tmp_path / "data"
        backup_dir = tmp_path / "backups"
        data_dir.mkdir()
        db_path = data_dir / "memory.db"

        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO items (value) VALUES ('wal-data')")
        conn.commit()

        try:
            target = backup_databases(data_dir, backup_dir)
        finally:
            conn.close()

        assert _read_values(target / "memory.db") == ["wal-data"]

    def test_ignores_non_db_files(self, tmp_path):
        data_dir = tmp_path / "data"
        backup_dir = tmp_path / "backups"
        data_dir.mkdir()
        _make_db(data_dir / "memory.db", ["a"])
        (data_dir / "notes.txt").write_text("not a database")

        target = backup_databases(data_dir, backup_dir)

        assert sorted(p.name for p in target.iterdir()) == ["memory.db"]

    def test_empty_data_dir_creates_empty_backup_directory(self, tmp_path):
        data_dir = tmp_path / "data"
        backup_dir = tmp_path / "backups"
        data_dir.mkdir()

        target = backup_databases(data_dir, backup_dir)

        assert target.is_dir()
        assert list(target.iterdir()) == []

    def test_corrupt_db_does_not_abort_other_backups(self, tmp_path):
        data_dir = tmp_path / "data"
        backup_dir = tmp_path / "backups"
        data_dir.mkdir()
        _make_db(data_dir / "memory.db", ["a"])
        (data_dir / "broken.db").write_bytes(b"not a sqlite database")

        target = backup_databases(data_dir, backup_dir)

        assert (target / "memory.db").is_file()
        assert _read_values(target / "memory.db") == ["a"]
        assert not (target / "broken.db").exists()


# ---------------------------------------------------------------------------
# Tests for purge_old_backups (AC19)
# ---------------------------------------------------------------------------


class TestPurgeOldBackups:
    def test_deletes_directories_older_than_retention(self, tmp_path):
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()
        old_dir = backup_dir / "20260101T000000Z"
        old_dir.mkdir()
        (old_dir / "memory.db").write_text("old")

        now = datetime(2026, 6, 10, tzinfo=timezone.utc)
        deleted = purge_old_backups(backup_dir, retention_days=30, now=now)

        assert deleted == [old_dir]
        assert not old_dir.exists()

    def test_keeps_directories_within_retention(self, tmp_path):
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()
        recent_dir = backup_dir / "20260605T000000Z"
        recent_dir.mkdir()

        now = datetime(2026, 6, 10, tzinfo=timezone.utc)
        deleted = purge_old_backups(backup_dir, retention_days=30, now=now)

        assert deleted == []
        assert recent_dir.exists()

    def test_boundary_exactly_retention_days_old_is_kept(self, tmp_path):
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()
        now = datetime(2026, 6, 10, tzinfo=timezone.utc)
        boundary_dir = backup_dir / (now - timedelta(days=30)).strftime("%Y%m%dT%H%M%SZ")
        boundary_dir.mkdir()

        deleted = purge_old_backups(backup_dir, retention_days=30, now=now)

        assert deleted == []
        assert boundary_dir.exists()

    def test_one_day_past_retention_is_deleted(self, tmp_path):
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()
        now = datetime(2026, 6, 10, tzinfo=timezone.utc)
        past_dir = backup_dir / (now - timedelta(days=31)).strftime("%Y%m%dT%H%M%SZ")
        past_dir.mkdir()

        deleted = purge_old_backups(backup_dir, retention_days=30, now=now)

        assert deleted == [past_dir]
        assert not past_dir.exists()

    def test_ignores_non_matching_directory_names(self, tmp_path):
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()
        other_dir = backup_dir / "not-a-timestamp"
        other_dir.mkdir()

        now = datetime(2026, 6, 10, tzinfo=timezone.utc)
        deleted = purge_old_backups(backup_dir, retention_days=30, now=now)

        assert deleted == []
        assert other_dir.exists()

    def test_ignores_files_in_backup_dir(self, tmp_path):
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()
        stray_file = backup_dir / "20200101T000000Z"
        stray_file.write_text("not a directory")

        now = datetime(2026, 6, 10, tzinfo=timezone.utc)
        deleted = purge_old_backups(backup_dir, retention_days=30, now=now)

        assert deleted == []
        assert stray_file.exists()

    def test_missing_backup_dir_returns_empty_list(self, tmp_path):
        deleted = purge_old_backups(tmp_path / "does-not-exist")

        assert deleted == []


# ---------------------------------------------------------------------------
# Tests for run_daily_backup
# ---------------------------------------------------------------------------


class TestRunDailyBackup:
    def test_returns_backup_path_and_purged_list(self, tmp_path):
        data_dir = tmp_path / "data"
        backup_dir = tmp_path / "backups"
        data_dir.mkdir()
        _make_db(data_dir / "memory.db", ["a"])

        old_dir = backup_dir / "20200101T000000Z"
        old_dir.mkdir(parents=True)

        now = datetime(2026, 6, 10, 3, 0, 0, tzinfo=timezone.utc)
        result = run_daily_backup(data_dir, backup_dir, now=now)

        assert result["backup_path"] == backup_dir / "20260610T030000Z"
        assert result["backup_path"].is_dir()
        assert (result["backup_path"] / "memory.db").is_file()
        assert result["purged"] == [old_dir]
        assert not old_dir.exists()

    def test_respects_custom_retention_days(self, tmp_path):
        data_dir = tmp_path / "data"
        backup_dir = tmp_path / "backups"
        data_dir.mkdir()
        _make_db(data_dir / "memory.db", ["a"])

        now = datetime(2026, 6, 10, tzinfo=timezone.utc)
        recent_dir = backup_dir / (now - timedelta(days=10)).strftime("%Y%m%dT%H%M%SZ")
        recent_dir.mkdir(parents=True)

        result = run_daily_backup(data_dir, backup_dir, retention_days=5, now=now)

        assert result["purged"] == [recent_dir]
        assert not recent_dir.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
