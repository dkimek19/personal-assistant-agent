"""
NoteStore — SQLite-backed memo storage for the ``/note`` command.

Stores short free-text notes for the single-user personal assistant agent.
Uses memory.db as the backing store (shared with SessionStore).

Schema
------
notes
  note_id    INTEGER PRIMARY KEY AUTOINCREMENT
  user_id    TEXT NOT NULL
  content    TEXT NOT NULL
  created_at TEXT NOT NULL          -- ISO 8601
  updated_at TEXT NOT NULL          -- ISO 8601
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default database location: ~/assistant/data/memory.db (shared with SessionStore)
_DEFAULT_DB_DIR = Path.home() / "assistant" / "data"
_DEFAULT_DB_PATH = _DEFAULT_DB_DIR / "memory.db"

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS notes (
    note_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def _now_iso() -> str:
    """Return current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


class NoteStore:
    """
    SQLite-backed memo store for the personal assistant agent.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file. Defaults to
        ``~/assistant/data/memory.db``. The parent directory is created
        automatically if it does not exist.

    Usage
    -----
    >>> store = NoteStore(db_path="/tmp/memory.db")
    >>> note_id = store.add_note("user_1", "Buy milk")
    >>> store.list_notes("user_1")[0]["content"]
    'Buy milk'
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            db_path = _DEFAULT_DB_PATH
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _connect(self):
        """Yield a connected, autocommit-on-success SQLite connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Create the notes table if it does not yet exist."""
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE_SQL)
        logger.debug("NoteStore initialised at %s", self.db_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_note(self, user_id: str, content: str) -> int:
        """Create a new note for *user_id* and return its ``note_id``.

        Raises
        ------
        ValueError
            If *user_id* or *content* is empty/whitespace-only.
        """
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content must be a non-empty string")

        now = _now_iso()
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO notes (user_id, content, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (user_id, content, now, now),
            )
            note_id = cursor.lastrowid

        logger.info("add_note: created note %d for user_id=%r", note_id, user_id)
        return note_id

    def get_note(self, user_id: str, note_id: int) -> dict[str, Any] | None:
        """Retrieve a single note by id, scoped to *user_id*.

        Returns ``None`` if no matching note exists.
        """
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")

        with self._connect() as conn:
            row = conn.execute(
                "SELECT note_id, content, created_at, updated_at "
                "FROM notes WHERE user_id = ? AND note_id = ?",
                (user_id, note_id),
            ).fetchone()

        return dict(row) if row is not None else None

    def list_notes(self, user_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        """Return notes for *user_id*, most recently created first.

        Parameters
        ----------
        limit:
            Maximum number of notes to return. ``None`` (default) returns
            all notes.
        """
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")

        sql = (
            "SELECT note_id, content, created_at, updated_at "
            "FROM notes WHERE user_id = ? ORDER BY note_id DESC"
        )
        params: tuple[Any, ...] = (user_id,)
        if limit is not None:
            sql += " LIMIT ?"
            params = (user_id, limit)

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [dict(row) for row in rows]

    def update_note(self, user_id: str, note_id: int, content: str) -> bool:
        """Update the content of an existing note.

        Returns ``True`` if a note was updated, ``False`` if no matching
        note exists for *user_id*.

        Raises
        ------
        ValueError
            If *user_id* or *content* is empty/whitespace-only.
        """
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content must be a non-empty string")

        now = _now_iso()
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE notes SET content = ?, updated_at = ? "
                "WHERE user_id = ? AND note_id = ?",
                (content, now, user_id, note_id),
            )

        return cursor.rowcount > 0

    def delete_note(self, user_id: str, note_id: int) -> bool:
        """Delete a note by id, scoped to *user_id*.

        Returns ``True`` if a note was deleted, ``False`` if no matching
        note exists.
        """
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")

        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM notes WHERE user_id = ? AND note_id = ?",
                (user_id, note_id),
            )

        deleted = cursor.rowcount > 0
        logger.debug(
            "delete_note: user_id=%r note_id=%r deleted=%s", user_id, note_id, deleted
        )
        return deleted


# ---------------------------------------------------------------------------
# /note command handler
# ---------------------------------------------------------------------------


def handle_note_command(user_id: str, command_text: str, store: NoteStore) -> str:
    """Handle a ``/note`` command and return a user-facing response string.

    Supported forms:
        ``/note <text>``          — save *text* as a new note
        ``/note list``             — list all notes, most recent first
        ``/note delete <note_id>`` — delete the note with the given id

    Parameters
    ----------
    user_id:
        Stable identifier for the user issuing the command.
    command_text:
        The raw command text, e.g. ``"/note Buy milk"``.
    store:
        The :class:`NoteStore` instance to operate on.

    Returns
    -------
    str
        A human-readable response to send back to the user.

    Raises
    ------
    ValueError
        If *command_text* does not start with ``"/note"``, has no argument,
        or the id given to ``delete`` is not a valid integer.
    """
    if not command_text.startswith("/note"):
        raise ValueError("command_text must start with '/note'")

    remainder = command_text[len("/note"):].strip()

    if not remainder:
        raise ValueError(
            "/note requires an argument: note text, 'list', or 'delete <id>'"
        )

    if remainder == "list":
        notes = store.list_notes(user_id)
        if not notes:
            return "You have no notes."
        return "\n".join(f"[{n['note_id']}] {n['content']}" for n in notes)

    if remainder.startswith("delete"):
        arg = remainder[len("delete"):].strip()
        try:
            note_id = int(arg)
        except ValueError as exc:
            raise ValueError(f"Invalid note id: {arg!r}") from exc

        if store.delete_note(user_id, note_id):
            return f"Note {note_id} deleted."
        return f"Note {note_id} not found."

    note_id = store.add_note(user_id, remainder)
    return f"Note {note_id} saved."
