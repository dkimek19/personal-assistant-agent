"""
Runnable test for SessionStore (Sub-AC 6.1).

Tests:
  1. Create a session for a user  — verify it round-trips correctly.
  2. Update the session context    — verify the new context is persisted.
  3. Retrieve the session          — verify session_id is stable across updates.
  4. Delete a session              — verify get_session returns None afterwards.
  5. Unknown user                  — verify get_session returns None gracefully.
  6. Input validation              — verify ValueError on bad user_id / context.

Run with:
    python -m pytest tests/test_session_store.py -v
    # — or — without pytest:
    python tests/test_session_store.py
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

# Make sure the package is importable when running directly from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from assistant.session_store import SessionStore


class TestSessionStore(unittest.TestCase):
    """Full lifecycle tests for SessionStore."""

    def setUp(self) -> None:
        """Each test gets its own isolated in-memory/temp database."""
        # Use a real temp file so WAL mode works correctly (in-memory SQLite
        # doesn't support WAL and may give different locking behaviour).
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self.store = SessionStore(db_path=self._tmp.name)

    def tearDown(self) -> None:
        Path(self._tmp.name).unlink(missing_ok=True)
        # Remove WAL / SHM sidecar files if present
        for ext in ("-wal", "-shm"):
            Path(self._tmp.name + ext).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # AC 6.1 — core scenario: create → update → retrieve
    # ------------------------------------------------------------------

    def test_create_session(self) -> None:
        """Creating a session stores the context and returns a session_id."""
        initial_ctx = {
            "working_memory": [],
            "source_interface": "web_ui",
            "session_memory": {},
        }
        sid = self.store.upsert_session("user_1", initial_ctx)

        self.assertIsInstance(sid, str)
        self.assertTrue(len(sid) == 36, "session_id should be a UUID4 string")

    def test_retrieve_session_after_create(self) -> None:
        """get_session returns the context that was stored by upsert_session."""
        initial_ctx = {
            "working_memory": [{"role": "user", "content": "Hello"}],
            "source_interface": "telegram",
            "session_memory": {},
        }
        self.store.upsert_session("user_1", initial_ctx)
        retrieved = self.store.get_session("user_1")

        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["source_interface"], "telegram")
        self.assertEqual(len(retrieved["working_memory"]), 1)
        self.assertEqual(retrieved["working_memory"][0]["content"], "Hello")

    def test_update_session_preserves_session_id(self) -> None:
        """
        Updating a session (upsert on existing user) must preserve the
        original session_id — the session identity must be stable.
        """
        ctx_v1 = {"working_memory": [], "source_interface": "web_ui"}
        sid_v1 = self.store.upsert_session("user_1", ctx_v1)

        ctx_v2 = {
            "working_memory": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
            "source_interface": "discord",
        }
        sid_v2 = self.store.upsert_session("user_1", ctx_v2)

        self.assertEqual(sid_v1, sid_v2, "session_id must not change on update")

    def test_update_and_retrieve_new_context(self) -> None:
        """
        Full AC 6.1 scenario:
          1. Create session.
          2. Update with new context.
          3. Retrieve — verify updated context is returned.
        """
        # Step 1 — create
        ctx_v1 = {"working_memory": [], "source_interface": "web_ui", "note": "initial"}
        self.store.upsert_session("user_1", ctx_v1)

        # Step 2 — update
        ctx_v2 = {
            "working_memory": [{"role": "user", "content": "What is the weather?"}],
            "source_interface": "telegram",
            "note": "updated",
        }
        self.store.upsert_session("user_1", ctx_v2)

        # Step 3 — retrieve
        result = self.store.get_session("user_1")

        self.assertIsNotNone(result)
        self.assertEqual(result["note"], "updated", "Context should reflect the latest upsert")
        self.assertEqual(result["source_interface"], "telegram")
        self.assertEqual(result["working_memory"][0]["content"], "What is the weather?")

    # ------------------------------------------------------------------
    # Metadata / _meta injection
    # ------------------------------------------------------------------

    def test_get_session_injects_meta(self) -> None:
        """get_session injects _meta with session_id, created_at, updated_at."""
        self.store.upsert_session("user_1", {"working_memory": []})
        ctx = self.store.get_session("user_1")

        self.assertIn("_meta", ctx)
        meta = ctx["_meta"]
        self.assertIn("session_id", meta)
        self.assertIn("created_at", meta)
        self.assertIn("updated_at", meta)

    def test_meta_not_persisted(self) -> None:
        """
        _meta injected by get_session must not be written back to the DB
        when the caller passes the retrieved context straight into upsert_session.
        """
        self.store.upsert_session("user_1", {"working_memory": []})
        ctx = self.store.get_session("user_1")  # contains _meta

        # Pass _meta-containing dict back into upsert — should not blow up
        self.store.upsert_session("user_1", ctx)

        ctx2 = self.store.get_session("user_1")
        # _meta should still be a single-level injection, not nested
        self.assertNotIn("_meta", ctx2.get("_meta", {}))

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_get_session_unknown_user_returns_none(self) -> None:
        """get_session returns None for a user that has never been seen."""
        result = self.store.get_session("nobody")
        self.assertIsNone(result)

    def test_delete_session(self) -> None:
        """delete_session removes the session; subsequent get returns None."""
        self.store.upsert_session("user_1", {"working_memory": []})
        deleted = self.store.delete_session("user_1")
        self.assertTrue(deleted)

        result = self.store.get_session("user_1")
        self.assertIsNone(result)

    def test_delete_nonexistent_returns_false(self) -> None:
        """Deleting a user that doesn't exist returns False without error."""
        deleted = self.store.delete_session("ghost")
        self.assertFalse(deleted)

    def test_list_users(self) -> None:
        """list_users returns all stored user_ids."""
        self.store.upsert_session("alice", {"working_memory": []})
        self.store.upsert_session("bob", {"working_memory": []})
        users = self.store.list_users()
        self.assertIn("alice", users)
        self.assertIn("bob", users)

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def test_upsert_empty_user_id_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.store.upsert_session("", {"working_memory": []})

    def test_upsert_blank_user_id_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.store.upsert_session("   ", {"working_memory": []})

    def test_upsert_non_dict_context_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.store.upsert_session("user_1", "not a dict")  # type: ignore[arg-type]

    def test_get_empty_user_id_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.store.get_session("")

    # ------------------------------------------------------------------
    # Multi-user isolation
    # ------------------------------------------------------------------

    def test_multiple_users_are_isolated(self) -> None:
        """Two different users must never see each other's context."""
        self.store.upsert_session("alice", {"note": "alice_data"})
        self.store.upsert_session("bob", {"note": "bob_data"})

        alice_ctx = self.store.get_session("alice")
        bob_ctx = self.store.get_session("bob")

        self.assertEqual(alice_ctx["note"], "alice_data")
        self.assertEqual(bob_ctx["note"], "bob_data")

    # ------------------------------------------------------------------
    # Complex context payloads
    # ------------------------------------------------------------------

    def test_nested_context_round_trips(self) -> None:
        """Complex nested context (lists, dicts, None, booleans) must survive JSON round-trip."""
        complex_ctx = {
            "working_memory": [
                {"role": "user", "content": "Hello", "timestamp": "2026-06-09T00:00:00Z"},
                {"role": "assistant", "content": "Hi!", "timestamp": "2026-06-09T00:00:01Z"},
            ],
            "session_memory": {
                "summary": "User greeted the assistant.",
                "turn_count": 2,
            },
            "long_term_memory": [],
            "source_interface": "discord",
            "user_profile": {
                "name": None,
                "preferences": {"lang": "en", "verbose": True},
            },
        }
        self.store.upsert_session("user_1", complex_ctx)
        result = self.store.get_session("user_1")

        self.assertEqual(result["source_interface"], "discord")
        self.assertEqual(result["session_memory"]["turn_count"], 2)
        self.assertIsNone(result["user_profile"]["name"])
        self.assertTrue(result["user_profile"]["preferences"]["verbose"])
        self.assertEqual(len(result["working_memory"]), 2)


# ---------------------------------------------------------------------------
# Allow running the test file directly (without pytest)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestSessionStore)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
