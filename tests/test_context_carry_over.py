"""
Runnable test for ContextCarryOver.save_context (Sub-AC 6.3.1).

Verifies that:
  1. save_context creates the ``context`` table automatically (schema init).
  2. A multi-turn messages list is persisted and the raw JSON payload is
     present in the database when queried via a **direct SQLite connection**
     (not through the ContextCarryOver API).
  3. Repeated calls for the same user_id replace the previous payload (upsert).
  4. load_context correctly deserialises the stored payload.
  5. Input validation raises ValueError for bad arguments.
  6. Multiple users are isolated from each other.

Run with:
    python -m pytest tests/test_context_carry_over.py -v
    # — or — without pytest:
    python tests/test_context_carry_over.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure the package root is importable when running directly from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from assistant.context_carry_over import ContextCarryOver


class TestContextCarryOver(unittest.TestCase):
    """Tests for ContextCarryOver — Sub-AC 6.3.1."""

    # ------------------------------------------------------------------
    # Test fixtures
    # ------------------------------------------------------------------

    def setUp(self) -> None:
        """Each test gets its own isolated temp database file."""
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self.db_path = self._tmp.name
        self.cc = ContextCarryOver(db_path=self.db_path)

    def tearDown(self) -> None:
        Path(self._tmp.name).unlink(missing_ok=True)
        for ext in ("-wal", "-shm"):
            Path(self._tmp.name + ext).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Helper: direct SQLite query (bypass ContextCarryOver API)
    # ------------------------------------------------------------------

    def _raw_query(self, user_id: str) -> str | None:
        """Return the raw ``messages`` column value via a direct SQLite query."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT messages FROM context WHERE user_id = ?",
                (user_id,),
            )
            row = cursor.fetchone()
        finally:
            conn.close()
        return row[0] if row is not None else None

    # ------------------------------------------------------------------
    # Sub-AC 6.3.1 — Primary assertion
    # ------------------------------------------------------------------

    def test_save_context_raw_payload_present_in_db(self) -> None:
        """
        **Primary Sub-AC 6.3.1 assertion.**

        save_context writes a multi-turn messages list and the raw serialised
        payload must be present in the ``context`` table as verified by a
        direct SQLite query (not through ContextCarryOver.load_context).
        """
        user_id = "user_001"
        messages = [
            {"role": "user", "content": "What is the weather today?"},
            {"role": "assistant", "content": "Let me check that for you."},
            {"role": "user", "content": "Also, do I have any meetings?"},
            {"role": "assistant", "content": "You have a stand-up at 10:00 AM."},
        ]

        # Act — persist via the public API
        self.cc.save_context(user_id, messages)

        # Assert — verify using a DIRECT SQLite query (not the API)
        raw_payload = self._raw_query(user_id)

        self.assertIsNotNone(
            raw_payload,
            "A row must exist in the context table after save_context",
        )

        # The raw payload must be valid JSON that round-trips to the original list
        decoded = json.loads(raw_payload)
        self.assertEqual(decoded, messages, "Raw DB payload must match the saved messages list")

        # Spot-check individual messages to confirm data fidelity
        self.assertEqual(decoded[0]["role"], "user")
        self.assertEqual(decoded[0]["content"], "What is the weather today?")
        self.assertEqual(decoded[1]["role"], "assistant")
        self.assertEqual(decoded[3]["content"], "You have a stand-up at 10:00 AM.")

    # ------------------------------------------------------------------
    # Schema initialisation
    # ------------------------------------------------------------------

    def test_context_table_created_on_init(self) -> None:
        """The ``context`` table must exist after ContextCarryOver is constructed."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='context'"
            )
            table = cursor.fetchone()
        finally:
            conn.close()

        self.assertIsNotNone(table, "context table must be created during __init__")

    def test_schema_init_is_idempotent(self) -> None:
        """Creating a second ContextCarryOver on the same db must not raise."""
        cc2 = ContextCarryOver(db_path=self.db_path)
        # If idempotent, no exception is raised and the table still exists
        raw = self._raw_query("nobody")
        self.assertIsNone(raw)

    # ------------------------------------------------------------------
    # Upsert semantics
    # ------------------------------------------------------------------

    def test_save_context_overwrites_previous_payload(self) -> None:
        """A second save_context call for the same user_id must replace the payload."""
        user_id = "user_001"
        messages_v1 = [{"role": "user", "content": "Hello"}]
        messages_v2 = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! How can I help?"},
            {"role": "user", "content": "Tell me a joke."},
        ]

        self.cc.save_context(user_id, messages_v1)
        self.cc.save_context(user_id, messages_v2)

        raw_payload = self._raw_query(user_id)
        decoded = json.loads(raw_payload)

        self.assertEqual(decoded, messages_v2, "Second save must overwrite the first")
        self.assertEqual(len(decoded), 3)

    # ------------------------------------------------------------------
    # load_context round-trip
    # ------------------------------------------------------------------

    def test_load_context_returns_original_messages(self) -> None:
        """load_context must return the same list that was passed to save_context."""
        messages = [
            {"role": "user", "content": "What time is it?", "timestamp": "2026-06-09T08:00:00Z"},
            {"role": "assistant", "content": "It is 8:00 AM UTC.", "timestamp": "2026-06-09T08:00:01Z"},
        ]
        self.cc.save_context("default", messages)
        loaded = self.cc.load_context("default")

        self.assertEqual(loaded, messages)
        self.assertEqual(loaded[0]["timestamp"], "2026-06-09T08:00:00Z")

    def test_load_context_returns_empty_list_for_unknown_user(self) -> None:
        """load_context must return [] (not None) if no context has been saved for user."""
        result = self.cc.load_context("nobody")
        self.assertEqual(result, [])

    # ------------------------------------------------------------------
    # Complex / nested payloads
    # ------------------------------------------------------------------

    def test_nested_message_content_survives_round_trip(self) -> None:
        """Complex message structures (nested dicts, lists, booleans, None) must survive."""
        messages = [
            {
                "role": "user",
                "content": "Run this code",
                "metadata": {"source": "discord", "flags": [1, 2, 3], "urgent": True},
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"tool": "code_execute", "input": {"code": "print('hi')"}}
                ],
            },
        ]
        self.cc.save_context("user_complex", messages)

        raw = self._raw_query("user_complex")
        decoded = json.loads(raw)

        self.assertEqual(decoded[0]["metadata"]["flags"], [1, 2, 3])
        self.assertTrue(decoded[0]["metadata"]["urgent"])
        self.assertIsNone(decoded[1]["content"])
        self.assertEqual(decoded[1]["tool_calls"][0]["tool"], "code_execute")

    def test_empty_messages_list_is_valid(self) -> None:
        """An empty messages list must be stored and retrieved correctly."""
        self.cc.save_context("user_empty", [])

        raw = self._raw_query("user_empty")
        self.assertIsNotNone(raw)

        decoded = json.loads(raw)
        self.assertEqual(decoded, [])

    # ------------------------------------------------------------------
    # Multi-user isolation
    # ------------------------------------------------------------------

    def test_multiple_users_are_isolated(self) -> None:
        """Two different users must not see each other's context."""
        msgs_alice = [{"role": "user", "content": "Alice's message"}]
        msgs_bob = [{"role": "user", "content": "Bob's message"}]

        self.cc.save_context("alice", msgs_alice)
        self.cc.save_context("bob", msgs_bob)

        alice_loaded = self.cc.load_context("alice")
        bob_loaded = self.cc.load_context("bob")

        self.assertEqual(alice_loaded[0]["content"], "Alice's message")
        self.assertEqual(bob_loaded[0]["content"], "Bob's message")

        # Cross-check raw payloads via direct SQLite query
        alice_raw = self._raw_query("alice")
        bob_raw = self._raw_query("bob")
        self.assertNotEqual(alice_raw, bob_raw)

    # ------------------------------------------------------------------
    # delete_context
    # ------------------------------------------------------------------

    def test_delete_context_removes_row(self) -> None:
        """delete_context must remove the row; subsequent load_context returns None."""
        self.cc.save_context("user_del", [{"role": "user", "content": "to be deleted"}])
        deleted = self.cc.delete_context("user_del")
        self.assertTrue(deleted)

        raw = self._raw_query("user_del")
        self.assertIsNone(raw)

    def test_delete_nonexistent_returns_false(self) -> None:
        """delete_context on a user with no saved context returns False."""
        result = self.cc.delete_context("ghost")
        self.assertFalse(result)

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def test_save_empty_user_id_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.cc.save_context("", [])

    def test_save_blank_user_id_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.cc.save_context("   ", [])

    def test_save_non_list_messages_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.cc.save_context("user_1", {"role": "user"})  # type: ignore[arg-type]

    def test_load_empty_user_id_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.cc.load_context("")


# ---------------------------------------------------------------------------
# Sub-AC 6.3.2 — load_context: direct SQLite pre-population + empty-record edge case
# ---------------------------------------------------------------------------

class TestLoadContextSubAC632(unittest.TestCase):
    """
    Dedicated tests for Sub-AC 6.3.2.

    Acceptance criteria:
      - ``load_context(user_id)`` queries SQLite by ``user_id``
      - Deserialises the stored payload back into a Python list of message dicts
      - Returns an **empty list** (not None) when no record exists
      - The test pre-populates the database with a **known serialised payload
        via a direct SQLite INSERT** (not through ContextCarryOver.save_context)
        and asserts the returned list equals the original messages.
    """

    def setUp(self) -> None:
        """Each test gets its own isolated temp database file."""
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self.db_path = self._tmp.name
        self.cc = ContextCarryOver(db_path=self.db_path)

    def tearDown(self) -> None:
        Path(self._tmp.name).unlink(missing_ok=True)
        for ext in ("-wal", "-shm"):
            Path(self._tmp.name + ext).unlink(missing_ok=True)

    def _raw_insert(self, user_id: str, messages_json: str) -> None:
        """Directly INSERT a pre-serialised payload into the context table."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO context (user_id, messages, updated_at)
                VALUES (?, ?, '2026-06-09T00:00:00+00:00')
                ON CONFLICT(user_id) DO UPDATE SET
                    messages   = excluded.messages,
                    updated_at = excluded.updated_at
                """,
                (user_id, messages_json),
            )
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Sub-AC 6.3.2 — Primary assertion
    # ------------------------------------------------------------------

    def test_load_context_deserialises_prepopulated_payload(self) -> None:
        """
        **Primary Sub-AC 6.3.2 assertion.**

        Pre-populate the database with a known serialised payload via a direct
        SQLite INSERT (bypassing ContextCarryOver.save_context), call
        load_context, and assert the returned list equals the original messages.
        """
        user_id = "user_6_3_2"
        original_messages = [
            {"role": "user", "content": "What is the weather today?"},
            {"role": "assistant", "content": "Let me check that for you."},
            {"role": "user", "content": "Also, do I have any meetings?"},
            {"role": "assistant", "content": "You have a stand-up at 10:00 AM."},
        ]

        # Pre-populate via a DIRECT SQLite INSERT — not through save_context
        known_payload = json.dumps(original_messages, ensure_ascii=False)
        self._raw_insert(user_id, known_payload)

        # Act — call load_context
        loaded = self.cc.load_context(user_id)

        # Assert — returned list must equal the original messages exactly
        self.assertEqual(
            loaded,
            original_messages,
            "load_context must deserialise and return the exact pre-populated messages list",
        )

        # Spot-check individual fields for data fidelity
        self.assertEqual(loaded[0]["role"], "user")
        self.assertEqual(loaded[0]["content"], "What is the weather today?")
        self.assertEqual(loaded[1]["role"], "assistant")
        self.assertEqual(loaded[3]["content"], "You have a stand-up at 10:00 AM.")

    def test_load_context_returns_empty_list_for_missing_record(self) -> None:
        """
        **Sub-AC 6.3.2 empty-record edge case.**

        When no row exists for the given user_id, load_context must return
        an empty list — never None.
        """
        result = self.cc.load_context("nonexistent_user_6_3_2")

        self.assertIsInstance(result, list, "load_context must always return a list")
        self.assertEqual(result, [], "load_context must return [] when no record exists")

    def test_load_context_returns_empty_list_not_none(self) -> None:
        """
        Explicitly verify that None is never returned for a missing record.

        This is a type-contract test: the return type is list, not Optional[list].
        """
        result = self.cc.load_context("another_missing_user")
        self.assertIsNotNone(result, "load_context must never return None")
        self.assertIsInstance(result, list)

    def test_load_context_single_message_payload(self) -> None:
        """Pre-populate with a single message and verify correct deserialisation."""
        user_id = "user_single"
        original = [{"role": "user", "content": "Hello, assistant!"}]
        self._raw_insert(user_id, json.dumps(original))

        loaded = self.cc.load_context(user_id)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded, original)

    def test_load_context_complex_nested_payload(self) -> None:
        """Pre-populate with a complex nested payload and verify deep equality."""
        user_id = "user_nested"
        original = [
            {
                "role": "user",
                "content": "Execute this",
                "metadata": {
                    "source": "telegram",
                    "flags": [True, False, 42],
                    "nested": {"deep": "value"},
                },
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"tool": "code_execute", "input": {"code": "print('hello')"}}
                ],
            },
        ]
        self._raw_insert(user_id, json.dumps(original))

        loaded = self.cc.load_context(user_id)

        self.assertEqual(loaded, original)
        self.assertEqual(loaded[0]["metadata"]["flags"], [True, False, 42])
        self.assertIsNone(loaded[1]["content"])
        self.assertEqual(loaded[1]["tool_calls"][0]["tool"], "code_execute")

    def test_load_context_empty_messages_array_payload(self) -> None:
        """Pre-populate with an empty JSON array '[]' and verify [] is returned."""
        user_id = "user_empty_array"
        self._raw_insert(user_id, "[]")

        loaded = self.cc.load_context(user_id)

        self.assertEqual(loaded, [])

    def test_load_context_multiple_users_isolated(self) -> None:
        """Pre-populate two users; each must only see their own messages."""
        msgs_alice = [{"role": "user", "content": "Alice says hi"}]
        msgs_bob = [{"role": "user", "content": "Bob says hello"}]

        self._raw_insert("alice_632", json.dumps(msgs_alice))
        self._raw_insert("bob_632", json.dumps(msgs_bob))

        loaded_alice = self.cc.load_context("alice_632")
        loaded_bob = self.cc.load_context("bob_632")

        self.assertEqual(loaded_alice, msgs_alice)
        self.assertEqual(loaded_bob, msgs_bob)
        self.assertNotEqual(loaded_alice, loaded_bob)


# ---------------------------------------------------------------------------
# Allow running this test file directly (without pytest)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestContextCarryOver))
    suite.addTests(loader.loadTestsFromTestCase(TestLoadContextSubAC632))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
