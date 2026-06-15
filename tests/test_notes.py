"""
Tests for the /note memo system (assistant.notes).

Covers:
  - NoteStore CRUD: add_note, get_note, list_notes, update_note, delete_note
  - Multi-user isolation
  - Input validation (ValueError on empty user_id / content)
  - handle_note_command: add / list / delete forms, and error handling for
    malformed commands

Run with:
    python -m pytest tests/test_notes.py -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from assistant.notes import NoteStore, handle_note_command


class TestNoteStore(unittest.TestCase):
    """CRUD lifecycle tests for NoteStore."""

    def setUp(self) -> None:
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self.store = NoteStore(db_path=self._tmp.name)

    def tearDown(self) -> None:
        Path(self._tmp.name).unlink(missing_ok=True)
        for ext in ("-wal", "-shm"):
            Path(self._tmp.name + ext).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # add_note / get_note
    # ------------------------------------------------------------------

    def test_add_note_returns_int_id(self) -> None:
        note_id = self.store.add_note("user_1", "Buy milk")
        self.assertIsInstance(note_id, int)

    def test_get_note_returns_stored_content(self) -> None:
        note_id = self.store.add_note("user_1", "Buy milk")
        note = self.store.get_note("user_1", note_id)

        self.assertIsNotNone(note)
        self.assertEqual(note["note_id"], note_id)
        self.assertEqual(note["content"], "Buy milk")
        self.assertIn("created_at", note)
        self.assertIn("updated_at", note)

    def test_get_note_unknown_id_returns_none(self) -> None:
        self.assertIsNone(self.store.get_note("user_1", 9999))

    def test_get_note_scoped_to_user(self) -> None:
        note_id = self.store.add_note("alice", "Alice's note")
        self.assertIsNone(self.store.get_note("bob", note_id))

    # ------------------------------------------------------------------
    # list_notes
    # ------------------------------------------------------------------

    def test_list_notes_empty_for_new_user(self) -> None:
        self.assertEqual(self.store.list_notes("user_1"), [])

    def test_list_notes_returns_most_recent_first(self) -> None:
        first_id = self.store.add_note("user_1", "First note")
        second_id = self.store.add_note("user_1", "Second note")

        notes = self.store.list_notes("user_1")

        self.assertEqual([n["note_id"] for n in notes], [second_id, first_id])

    def test_list_notes_respects_limit(self) -> None:
        for i in range(5):
            self.store.add_note("user_1", f"Note {i}")

        notes = self.store.list_notes("user_1", limit=2)

        self.assertEqual(len(notes), 2)

    # ------------------------------------------------------------------
    # update_note
    # ------------------------------------------------------------------

    def test_update_note_changes_content(self) -> None:
        note_id = self.store.add_note("user_1", "Original")
        updated = self.store.update_note("user_1", note_id, "Updated")

        self.assertTrue(updated)
        self.assertEqual(self.store.get_note("user_1", note_id)["content"], "Updated")

    def test_update_note_unknown_id_returns_false(self) -> None:
        self.assertFalse(self.store.update_note("user_1", 9999, "Updated"))

    # ------------------------------------------------------------------
    # delete_note
    # ------------------------------------------------------------------

    def test_delete_note_removes_it(self) -> None:
        note_id = self.store.add_note("user_1", "Buy milk")
        deleted = self.store.delete_note("user_1", note_id)

        self.assertTrue(deleted)
        self.assertIsNone(self.store.get_note("user_1", note_id))

    def test_delete_nonexistent_note_returns_false(self) -> None:
        self.assertFalse(self.store.delete_note("user_1", 9999))

    # ------------------------------------------------------------------
    # Multi-user isolation
    # ------------------------------------------------------------------

    def test_multiple_users_are_isolated(self) -> None:
        self.store.add_note("alice", "Alice's note")
        self.store.add_note("bob", "Bob's note")

        alice_notes = self.store.list_notes("alice")
        bob_notes = self.store.list_notes("bob")

        self.assertEqual(len(alice_notes), 1)
        self.assertEqual(len(bob_notes), 1)
        self.assertEqual(alice_notes[0]["content"], "Alice's note")
        self.assertEqual(bob_notes[0]["content"], "Bob's note")

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def test_add_note_empty_user_id_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.store.add_note("", "content")

    def test_add_note_empty_content_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.store.add_note("user_1", "")

    def test_add_note_whitespace_content_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.store.add_note("user_1", "   ")

    def test_list_notes_empty_user_id_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.store.list_notes("")


class TestHandleNoteCommand(unittest.TestCase):
    """Tests for the /note command handler."""

    def setUp(self) -> None:
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self.store = NoteStore(db_path=self._tmp.name)

    def tearDown(self) -> None:
        Path(self._tmp.name).unlink(missing_ok=True)
        for ext in ("-wal", "-shm"):
            Path(self._tmp.name + ext).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Add form
    # ------------------------------------------------------------------

    def test_add_note_returns_confirmation(self) -> None:
        response = handle_note_command("user_1", "/note Buy milk", self.store)

        self.assertIn("saved", response)
        notes = self.store.list_notes("user_1")
        self.assertEqual(notes[0]["content"], "Buy milk")

    def test_add_note_confirmation_mentions_id(self) -> None:
        response = handle_note_command("user_1", "/note Buy milk", self.store)
        note_id = self.store.list_notes("user_1")[0]["note_id"]

        self.assertIn(str(note_id), response)

    # ------------------------------------------------------------------
    # List form
    # ------------------------------------------------------------------

    def test_list_empty_returns_no_notes_message(self) -> None:
        response = handle_note_command("user_1", "/note list", self.store)
        self.assertEqual(response, "You have no notes.")

    def test_list_returns_formatted_notes(self) -> None:
        handle_note_command("user_1", "/note Buy milk", self.store)
        handle_note_command("user_1", "/note Call dentist", self.store)

        response = handle_note_command("user_1", "/note list", self.store)

        self.assertIn("Buy milk", response)
        self.assertIn("Call dentist", response)

    # ------------------------------------------------------------------
    # Delete form
    # ------------------------------------------------------------------

    def test_delete_existing_note(self) -> None:
        handle_note_command("user_1", "/note Buy milk", self.store)
        note_id = self.store.list_notes("user_1")[0]["note_id"]

        response = handle_note_command("user_1", f"/note delete {note_id}", self.store)

        self.assertIn("deleted", response)
        self.assertEqual(self.store.list_notes("user_1"), [])

    def test_delete_nonexistent_note(self) -> None:
        response = handle_note_command("user_1", "/note delete 9999", self.store)
        self.assertIn("not found", response)

    def test_delete_invalid_id_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            handle_note_command("user_1", "/note delete abc", self.store)

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_missing_argument_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            handle_note_command("user_1", "/note", self.store)

    def test_whitespace_only_argument_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            handle_note_command("user_1", "/note   ", self.store)

    def test_non_note_command_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            handle_note_command("user_1", "/remember something", self.store)


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestNoteStore))
    suite.addTests(loader.loadTestsFromTestCase(TestHandleNoteCommand))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
