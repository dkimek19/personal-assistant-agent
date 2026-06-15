"""
Runnable test for SessionManager (Sub-AC 6.1.1 — schema initialisation).

Verifies that constructing a SessionManager:
  1. Creates the backing SQLite database file on first run (if missing).
  2. Creates the `sessions` table.
  3. The `sessions` table has the expected columns.
  4. Initialisation is idempotent (safe to construct repeatedly).
  5. Basic session lifecycle (create / get / update) works against the
     freshly initialised schema.

Run with:
    python -m pytest tests/test_session_manager.py -v
    # — or — without pytest:
    python tests/test_session_manager.py
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure the package root is on sys.path when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from assistant.session_manager import SessionManager


class TestSessionManagerSchemaInit(unittest.TestCase):
    """Schema-initialisation tests for SessionManager — Sub-AC 6.1.1 (1)."""

    def setUp(self) -> None:
        """Each test gets its own isolated temp directory for the db file."""
        self._tmpdir = tempfile.TemporaryDirectory()
        # Use a path that does NOT yet exist to verify "first run" creation.
        self.db_path = Path(self._tmpdir.name) / "memory.db"

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    # ------------------------------------------------------------------
    # Core: db file + table created on first run
    # ------------------------------------------------------------------

    def test_db_file_does_not_exist_before_init(self) -> None:
        """Sanity check: the db file must not exist prior to construction."""
        self.assertFalse(self.db_path.exists())

    def test_db_file_created_on_first_run(self) -> None:
        """Constructing SessionManager creates the database file on disk."""
        SessionManager(db_path=self.db_path)
        self.assertTrue(self.db_path.exists(), "memory.db should be created on first run")

    def test_sessions_table_created(self) -> None:
        """The `sessions` table must exist after initialisation."""
        manager = SessionManager(db_path=self.db_path)
        self.assertTrue(manager.table_exists("sessions"))

    def test_sessions_table_has_expected_columns(self) -> None:
        """The `sessions` table must contain all expected columns."""
        manager = SessionManager(db_path=self.db_path)
        columns = manager.get_columns("sessions")

        for expected_col in SessionManager.SESSIONS_COLUMNS:
            with self.subTest(column=expected_col):
                self.assertIn(expected_col, columns)

        self.assertEqual(set(columns), set(SessionManager.SESSIONS_COLUMNS))

    def test_is_initialized_true_after_construction(self) -> None:
        """is_initialized() reports True once the schema has been created."""
        manager = SessionManager(db_path=self.db_path)
        self.assertTrue(manager.is_initialized())

    def test_table_exists_false_for_unknown_table(self) -> None:
        """table_exists() returns False for a table that was never created."""
        manager = SessionManager(db_path=self.db_path)
        self.assertFalse(manager.table_exists("not_a_real_table"))

    def test_get_columns_empty_for_unknown_table(self) -> None:
        """get_columns() returns [] for a table that does not exist."""
        manager = SessionManager(db_path=self.db_path)
        self.assertEqual(manager.get_columns("not_a_real_table"), [])

    # ------------------------------------------------------------------
    # Verification via a raw, independent sqlite3 connection
    # ------------------------------------------------------------------

    def test_raw_sqlite_connection_sees_sessions_table_and_columns(self) -> None:
        """
        Independently (without using SessionManager helpers), open the db
        file with sqlite3 and confirm the `sessions` table and its columns
        exist exactly as expected.
        """
        SessionManager(db_path=self.db_path)

        conn = sqlite3.connect(str(self.db_path))
        try:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            self.assertIn("sessions", tables)

            columns = {row[1] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()}
            self.assertEqual(
                columns,
                {"user_id", "session_id", "context", "created_at", "updated_at"},
            )
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Idempotency
    # ------------------------------------------------------------------

    def test_initialization_idempotent(self) -> None:
        """Constructing SessionManager multiple times must not raise or reset data."""
        manager1 = SessionManager(db_path=self.db_path)
        manager1.create_session("default", {"working_memory": [{"role": "user", "content": "hi"}]})

        # Second construction against the same db file must succeed and
        # must not wipe the previously-stored session.
        manager2 = SessionManager(db_path=self.db_path)
        self.assertTrue(manager2.is_initialized())

        ctx = manager2.get_session("default")
        self.assertIsNotNone(ctx)
        self.assertEqual(ctx["working_memory"][0]["content"], "hi")

    # ------------------------------------------------------------------
    # Parent directory creation
    # ------------------------------------------------------------------

    def test_parent_directory_created_if_missing(self) -> None:
        """The parent directory of db_path is created if it does not exist."""
        nested_path = Path(self._tmpdir.name) / "nested" / "subdir" / "memory.db"
        self.assertFalse(nested_path.parent.exists())

        SessionManager(db_path=nested_path)

        self.assertTrue(nested_path.parent.exists())
        self.assertTrue(nested_path.exists())

    # ------------------------------------------------------------------
    # Default db path constant
    # ------------------------------------------------------------------

    def test_default_db_path_points_to_memory_db_in_assistant_data(self) -> None:
        """The default database path must be ~/assistant/data/memory.db."""
        from assistant.session_manager import _DEFAULT_DB_PATH

        self.assertEqual(_DEFAULT_DB_PATH, Path.home() / "assistant" / "data" / "memory.db")


class TestSessionManagerLifecycle(unittest.TestCase):
    """Basic session lifecycle operations against the initialised schema."""

    def setUp(self) -> None:
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self.manager = SessionManager(db_path=self._tmp.name)

    def tearDown(self) -> None:
        Path(self._tmp.name).unlink(missing_ok=True)
        for ext in ("-wal", "-shm"):
            Path(self._tmp.name + ext).unlink(missing_ok=True)

    def test_create_session_returns_session_id(self) -> None:
        session_id = self.manager.create_session("default")
        self.assertIsInstance(session_id, str)
        self.assertEqual(len(session_id), 36)

    def test_create_session_default_context_has_hierarchical_memory_keys(self) -> None:
        self.manager.create_session("default")
        ctx = self.manager.get_session("default")

        self.assertIsNotNone(ctx)
        self.assertIn("working_memory", ctx)
        self.assertIn("session_memory", ctx)
        self.assertIn("long_term_memory", ctx)

    def test_update_session_preserves_session_id(self) -> None:
        sid_1 = self.manager.create_session("default")
        sid_2 = self.manager.update_session(
            "default",
            {"working_memory": [{"role": "user", "content": "Hello"}]},
        )
        self.assertEqual(sid_1, sid_2)

    def test_get_session_unknown_user_returns_none(self) -> None:
        self.assertIsNone(self.manager.get_session("nobody"))


# ---------------------------------------------------------------------------
# Allow running the test file directly (without pytest)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestSessionManagerSchemaInit))
    suite.addTests(loader.loadTestsFromTestCase(TestSessionManagerLifecycle))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
