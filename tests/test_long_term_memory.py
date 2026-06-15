"""
Tests for the /remember long-term memory system (assistant.long_term_memory).

Covers:
  - LongTermMemoryStore CRUD: add_memory, get_memory, list_memories, forget_memory
  - Multi-user isolation
  - Input validation (ValueError on empty user_id / content)
  - format_memories_for_context: empty vs non-empty formatting, data fidelity
  - handle_remember_command: remember / list / forget forms, and error
    handling for malformed commands

Run with:
    python -m pytest tests/test_long_term_memory.py -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from assistant.long_term_memory import (
    LongTermMemoryStore,
    format_memories_for_context,
    handle_remember_command,
)


class TestLongTermMemoryStore(unittest.TestCase):
    """CRUD lifecycle tests for LongTermMemoryStore."""

    def setUp(self) -> None:
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self.store = LongTermMemoryStore(db_path=self._tmp.name)

    def tearDown(self) -> None:
        Path(self._tmp.name).unlink(missing_ok=True)
        for ext in ("-wal", "-shm"):
            Path(self._tmp.name + ext).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # add_memory / get_memory
    # ------------------------------------------------------------------

    def test_add_memory_returns_int_id(self) -> None:
        memory_id = self.store.add_memory("user_1", "Allergic to peanuts")
        self.assertIsInstance(memory_id, int)

    def test_get_memory_returns_stored_content(self) -> None:
        memory_id = self.store.add_memory("user_1", "Allergic to peanuts")
        memory = self.store.get_memory("user_1", memory_id)

        self.assertIsNotNone(memory)
        self.assertEqual(memory["memory_id"], memory_id)
        self.assertEqual(memory["content"], "Allergic to peanuts")
        self.assertIn("created_at", memory)
        self.assertIn("updated_at", memory)

    def test_get_memory_unknown_id_returns_none(self) -> None:
        self.assertIsNone(self.store.get_memory("user_1", 9999))

    def test_get_memory_scoped_to_user(self) -> None:
        memory_id = self.store.add_memory("alice", "Alice's fact")
        self.assertIsNone(self.store.get_memory("bob", memory_id))

    # ------------------------------------------------------------------
    # list_memories
    # ------------------------------------------------------------------

    def test_list_memories_empty_for_new_user(self) -> None:
        self.assertEqual(self.store.list_memories("user_1"), [])

    def test_list_memories_returns_most_recent_first(self) -> None:
        first_id = self.store.add_memory("user_1", "First fact")
        second_id = self.store.add_memory("user_1", "Second fact")

        memories = self.store.list_memories("user_1")

        self.assertEqual([m["memory_id"] for m in memories], [second_id, first_id])

    def test_list_memories_respects_limit(self) -> None:
        for i in range(5):
            self.store.add_memory("user_1", f"Fact {i}")

        memories = self.store.list_memories("user_1", limit=2)

        self.assertEqual(len(memories), 2)

    # ------------------------------------------------------------------
    # forget_memory
    # ------------------------------------------------------------------

    def test_forget_memory_removes_it(self) -> None:
        memory_id = self.store.add_memory("user_1", "Allergic to peanuts")
        forgotten = self.store.forget_memory("user_1", memory_id)

        self.assertTrue(forgotten)
        self.assertIsNone(self.store.get_memory("user_1", memory_id))

    def test_forget_nonexistent_memory_returns_false(self) -> None:
        self.assertFalse(self.store.forget_memory("user_1", 9999))

    # ------------------------------------------------------------------
    # Multi-user isolation
    # ------------------------------------------------------------------

    def test_multiple_users_are_isolated(self) -> None:
        self.store.add_memory("alice", "Alice's fact")
        self.store.add_memory("bob", "Bob's fact")

        alice_memories = self.store.list_memories("alice")
        bob_memories = self.store.list_memories("bob")

        self.assertEqual(len(alice_memories), 1)
        self.assertEqual(len(bob_memories), 1)
        self.assertEqual(alice_memories[0]["content"], "Alice's fact")
        self.assertEqual(bob_memories[0]["content"], "Bob's fact")

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def test_add_memory_empty_user_id_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.store.add_memory("", "content")

    def test_add_memory_empty_content_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.store.add_memory("user_1", "")

    def test_add_memory_whitespace_content_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.store.add_memory("user_1", "   ")

    def test_list_memories_empty_user_id_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.store.list_memories("")


class TestFormatMemoriesForContext(unittest.TestCase):
    """Tests for the LLM context-injection formatter."""

    def setUp(self) -> None:
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self.store = LongTermMemoryStore(db_path=self._tmp.name)

    def tearDown(self) -> None:
        Path(self._tmp.name).unlink(missing_ok=True)
        for ext in ("-wal", "-shm"):
            Path(self._tmp.name + ext).unlink(missing_ok=True)

    def test_empty_returns_empty_string(self) -> None:
        self.assertEqual(format_memories_for_context("user_1", self.store), "")

    def test_non_empty_includes_all_facts(self) -> None:
        self.store.add_memory("user_1", "Allergic to peanuts")
        self.store.add_memory("user_1", "Birthday is March 5th")

        text = format_memories_for_context("user_1", self.store)

        self.assertIn("Allergic to peanuts", text)
        self.assertIn("Birthday is March 5th", text)

    def test_facts_appear_in_chronological_order(self) -> None:
        self.store.add_memory("user_1", "First fact")
        self.store.add_memory("user_1", "Second fact")

        text = format_memories_for_context("user_1", self.store)

        self.assertLess(text.index("First fact"), text.index("Second fact"))

    def test_other_users_facts_excluded(self) -> None:
        self.store.add_memory("alice", "Alice's secret")
        self.store.add_memory("bob", "Bob's secret")

        text = format_memories_for_context("alice", self.store)

        self.assertIn("Alice's secret", text)
        self.assertNotIn("Bob's secret", text)


class TestHandleRememberCommand(unittest.TestCase):
    """Tests for the /remember command handler."""

    def setUp(self) -> None:
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self.store = LongTermMemoryStore(db_path=self._tmp.name)

    def tearDown(self) -> None:
        Path(self._tmp.name).unlink(missing_ok=True)
        for ext in ("-wal", "-shm"):
            Path(self._tmp.name + ext).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Remember (add) form
    # ------------------------------------------------------------------

    def test_remember_returns_confirmation(self) -> None:
        response = handle_remember_command(
            "user_1", "/remember Allergic to peanuts", self.store
        )

        self.assertIn("remember", response.lower())
        memories = self.store.list_memories("user_1")
        self.assertEqual(memories[0]["content"], "Allergic to peanuts")

    def test_remember_confirmation_mentions_id(self) -> None:
        response = handle_remember_command(
            "user_1", "/remember Allergic to peanuts", self.store
        )
        memory_id = self.store.list_memories("user_1")[0]["memory_id"]

        self.assertIn(str(memory_id), response)

    # ------------------------------------------------------------------
    # List form
    # ------------------------------------------------------------------

    def test_list_empty_returns_no_memories_message(self) -> None:
        response = handle_remember_command("user_1", "/remember list", self.store)
        self.assertIn("don't have anything remembered", response)

    def test_list_returns_formatted_memories(self) -> None:
        handle_remember_command("user_1", "/remember Allergic to peanuts", self.store)
        handle_remember_command("user_1", "/remember Birthday is March 5th", self.store)

        response = handle_remember_command("user_1", "/remember list", self.store)

        self.assertIn("Allergic to peanuts", response)
        self.assertIn("Birthday is March 5th", response)

    # ------------------------------------------------------------------
    # Forget form
    # ------------------------------------------------------------------

    def test_forget_existing_memory(self) -> None:
        handle_remember_command("user_1", "/remember Allergic to peanuts", self.store)
        memory_id = self.store.list_memories("user_1")[0]["memory_id"]

        response = handle_remember_command(
            "user_1", f"/remember forget {memory_id}", self.store
        )

        self.assertIn("Forgot", response)
        self.assertEqual(self.store.list_memories("user_1"), [])

    def test_forget_nonexistent_memory(self) -> None:
        response = handle_remember_command("user_1", "/remember forget 9999", self.store)
        self.assertIn("not found", response)

    def test_forget_invalid_id_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            handle_remember_command("user_1", "/remember forget abc", self.store)

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_missing_argument_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            handle_remember_command("user_1", "/remember", self.store)

    def test_whitespace_only_argument_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            handle_remember_command("user_1", "/remember   ", self.store)

    def test_non_remember_command_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            handle_remember_command("user_1", "/note something", self.store)


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestLongTermMemoryStore))
    suite.addTests(loader.loadTestsFromTestCase(TestFormatMemoriesForContext))
    suite.addTests(loader.loadTestsFromTestCase(TestHandleRememberCommand))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
