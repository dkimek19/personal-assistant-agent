"""
ContextCarryOver — SQLite-backed conversation context persistence.

Sub-AC 6.3.1: Persists the current messages list (conversation turns) for a
given user_id so that context can be carried across interface restarts or
reconnections.

Schema (in memory.db)
---------------------
context
  user_id     TEXT PRIMARY KEY   -- keyed by user; single-user system uses "default"
  messages    TEXT NOT NULL      -- JSON-serialised list of message dicts
  updated_at  TEXT NOT NULL      -- ISO 8601 UTC timestamp of last write

Notes
-----
* The table is created automatically on first use (schema initialisation is
  idempotent — safe to call multiple times).
* ``save_context`` is an **upsert**: a second call for the same ``user_id``
  replaces the previous messages list.
* The raw JSON payload stored in the ``messages`` column is intentionally
  unmodified by any LLM processing — it is the verbatim serialisation of the
  ``messages`` argument passed by the caller.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default database location mirrors the rest of the storage layer: memory.db
_DEFAULT_DB_DIR = Path.home() / "assistant" / "data"
_DEFAULT_DB_PATH = _DEFAULT_DB_DIR / "memory.db"

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS context (
    user_id    TEXT PRIMARY KEY,
    messages   TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def _now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


class ContextCarryOver:
    """
    Persist and retrieve conversation message lists across sessions.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Defaults to
        ``~/assistant/data/memory.db``.  The parent directory is created
        automatically if it does not exist.

    Usage
    -----
    >>> cc = ContextCarryOver()
    >>> messages = [
    ...     {"role": "user", "content": "Hello"},
    ...     {"role": "assistant", "content": "Hi there!"},
    ... ]
    >>> cc.save_context("default", messages)
    >>> loaded = cc.load_context("default")
    >>> loaded[0]["content"]
    'Hello'
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            db_path = _DEFAULT_DB_PATH
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _connect(self):
        """Yield a connected SQLite connection; commit on success, rollback on error."""
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

    def _init_schema(self) -> None:
        """Create the ``context`` table if it does not yet exist.

        This method is idempotent — calling it multiple times on the same
        database is safe.
        """
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE_SQL)
        logger.debug("ContextCarryOver schema initialised at %s", self.db_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_context(self, user_id: str, messages: list[dict[str, Any]]) -> None:
        """Serialise *messages* and persist it in the ``context`` table keyed
        by *user_id*.

        If a row already exists for *user_id*, it is replaced (upsert
        semantics).  The schema is created automatically on the first call if
        the table is absent.

        Parameters
        ----------
        user_id:
            Stable identifier for the user.  Must be a non-empty string.
        messages:
            List of message dicts (each typically containing at least
            ``"role"`` and ``"content"`` keys) representing the current
            conversation turns.

        Raises
        ------
        ValueError
            If *user_id* is empty/whitespace, or *messages* is not a list.
        TypeError
            If *messages* contains values that are not JSON-serialisable.

        Notes
        -----
        The ``messages`` column in the database contains the **raw JSON
        string** produced by ``json.dumps(messages)``.  The content is never
        modified by any LLM layer.
        """
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")

        messages_json: str = json.dumps(messages, ensure_ascii=False)
        now: str = _now_iso()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO context (user_id, messages, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    messages   = excluded.messages,
                    updated_at = excluded.updated_at
                """,
                (user_id, messages_json, now),
            )

        logger.debug(
            "save_context: persisted %d message(s) for user_id=%r",
            len(messages),
            user_id,
        )

    def load_context(self, user_id: str) -> list[dict[str, Any]]:
        """Retrieve and deserialise the stored messages list for *user_id*.

        Parameters
        ----------
        user_id:
            The user identifier to look up.

        Returns
        -------
        list
            The deserialised messages list.  Returns an **empty list** (``[]``)
            if no context has been saved for this user yet — never ``None``.

        Raises
        ------
        ValueError
            If *user_id* is empty or not a string.

        Notes
        -----
        Sub-AC 6.3.2: This method queries SQLite by ``user_id``, deserialises
        the stored JSON payload back into a Python list of message dicts, and
        returns an empty list when no record exists.
        """
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")

        with self._connect() as conn:
            row = conn.execute(
                "SELECT messages FROM context WHERE user_id = ?",
                (user_id,),
            ).fetchone()

        if row is None:
            logger.debug("load_context: no context found for user_id=%r — returning []", user_id)
            return []

        result: list[dict[str, Any]] = json.loads(row["messages"])
        logger.debug(
            "load_context: loaded %d message(s) for user_id=%r",
            len(result),
            user_id,
        )
        return result

    def delete_context(self, user_id: str) -> bool:
        """Remove the stored context for *user_id*.

        Returns ``True`` if a row was deleted, ``False`` if none existed.
        """
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")

        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM context WHERE user_id = ?", (user_id,)
            )
        deleted = cursor.rowcount > 0
        logger.debug(
            "delete_context: user_id=%r deleted=%s", user_id, deleted
        )
        return deleted
