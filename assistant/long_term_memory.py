"""
LongTermMemoryStore — SQLite-backed long-term memory for the ``/remember`` command.

Stores durable facts about the user (preferences, recurring details, etc.)
that should persist across sessions and be available to the agent as
background context. Uses ``user.db`` as the backing store — distinct from
``memory.db`` (working/session memory and notes), reflecting that long-term
memories describe the *user* rather than a particular conversation.

Schema
------
memories
  memory_id  INTEGER PRIMARY KEY AUTOINCREMENT
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

# Default database location: ~/assistant/data/user.db
_DEFAULT_DB_DIR = Path.home() / "assistant" / "data"
_DEFAULT_DB_PATH = _DEFAULT_DB_DIR / "user.db"

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    memory_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def _now_iso() -> str:
    """Return current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


class LongTermMemoryStore:
    """
    SQLite-backed long-term memory store for the personal assistant agent.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file. Defaults to
        ``~/assistant/data/user.db``. The parent directory is created
        automatically if it does not exist.

    Usage
    -----
    >>> store = LongTermMemoryStore(db_path="/tmp/user.db")
    >>> memory_id = store.add_memory("user_1", "Allergic to peanuts")
    >>> store.list_memories("user_1")[0]["content"]
    'Allergic to peanuts'
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
        """Create the memories table if it does not yet exist."""
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE_SQL)
        logger.debug("LongTermMemoryStore initialised at %s", self.db_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_memory(self, user_id: str, content: str) -> int:
        """Store a new long-term memory fact for *user_id* and return its id.

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
                "INSERT INTO memories (user_id, content, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (user_id, content, now, now),
            )
            memory_id = cursor.lastrowid

        logger.info("add_memory: created memory %d for user_id=%r", memory_id, user_id)
        return memory_id

    def get_memory(self, user_id: str, memory_id: int) -> dict[str, Any] | None:
        """Retrieve a single memory by id, scoped to *user_id*.

        Returns ``None`` if no matching memory exists.
        """
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")

        with self._connect() as conn:
            row = conn.execute(
                "SELECT memory_id, content, created_at, updated_at "
                "FROM memories WHERE user_id = ? AND memory_id = ?",
                (user_id, memory_id),
            ).fetchone()

        return dict(row) if row is not None else None

    def list_memories(self, user_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        """Return memories for *user_id*, most recently created first.

        Parameters
        ----------
        limit:
            Maximum number of memories to return. ``None`` (default) returns
            all memories.
        """
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")

        sql = (
            "SELECT memory_id, content, created_at, updated_at "
            "FROM memories WHERE user_id = ? ORDER BY memory_id DESC"
        )
        params: tuple[Any, ...] = (user_id,)
        if limit is not None:
            sql += " LIMIT ?"
            params = (user_id, limit)

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [dict(row) for row in rows]

    def forget_memory(self, user_id: str, memory_id: int) -> bool:
        """Delete a memory by id, scoped to *user_id*.

        Returns ``True`` if a memory was deleted, ``False`` if no matching
        memory exists.
        """
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")

        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM memories WHERE user_id = ? AND memory_id = ?",
                (user_id, memory_id),
            )

        deleted = cursor.rowcount > 0
        logger.debug(
            "forget_memory: user_id=%r memory_id=%r deleted=%s",
            user_id,
            memory_id,
            deleted,
        )
        return deleted


# ---------------------------------------------------------------------------
# LLM context injection
# ---------------------------------------------------------------------------


def format_memories_for_context(user_id: str, store: LongTermMemoryStore) -> str:
    """Return all of *user_id*'s long-term memories formatted for LLM context.

    The returned string is suitable for prepending to a system prompt so the
    agent has background knowledge about the user across sessions. Factual
    content is reproduced verbatim — formatting only.

    Returns an empty string if the user has no stored memories.
    """
    memories = store.list_memories(user_id)
    if not memories:
        return ""

    lines = "\n".join(f"- {m['content']}" for m in reversed(memories))
    return f"Things to remember about the user:\n{lines}"


# ---------------------------------------------------------------------------
# /remember command handler
# ---------------------------------------------------------------------------


def handle_remember_command(user_id: str, command_text: str, store: LongTermMemoryStore) -> str:
    """Handle a ``/remember`` command and return a user-facing response string.

    Supported forms:
        ``/remember <fact>``          — store *fact* as a long-term memory
        ``/remember list``             — list all stored memories, most recent first
        ``/remember forget <memory_id>`` — delete the memory with the given id

    Parameters
    ----------
    user_id:
        Stable identifier for the user issuing the command.
    command_text:
        The raw command text, e.g. ``"/remember I'm allergic to peanuts"``.
    store:
        The :class:`LongTermMemoryStore` instance to operate on.

    Returns
    -------
    str
        A human-readable response to send back to the user.

    Raises
    ------
    ValueError
        If *command_text* does not start with ``"/remember"``, has no
        argument, or the id given to ``forget`` is not a valid integer.
    """
    if not command_text.startswith("/remember"):
        raise ValueError("command_text must start with '/remember'")

    remainder = command_text[len("/remember"):].strip()

    if not remainder:
        raise ValueError(
            "/remember requires an argument: a fact to remember, 'list', "
            "or 'forget <id>'"
        )

    if remainder == "list":
        memories = store.list_memories(user_id)
        if not memories:
            return "I don't have anything remembered about you yet."
        return "\n".join(f"[{m['memory_id']}] {m['content']}" for m in memories)

    if remainder.startswith("forget"):
        arg = remainder[len("forget"):].strip()
        try:
            memory_id = int(arg)
        except ValueError as exc:
            raise ValueError(f"Invalid memory id: {arg!r}") from exc

        if store.forget_memory(user_id, memory_id):
            return f"Forgot memory {memory_id}."
        return f"Memory {memory_id} not found."

    memory_id = store.add_memory(user_id, remainder)
    return f"I'll remember that ({memory_id})."
