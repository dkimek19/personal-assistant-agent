"""
SessionManager — top-level lifecycle manager for the unified cross-interface
session (Web UI, Telegram, Discord).

Sub-AC 6.1.1 (1): SQLite schema initialisation
------------------------------------------------
On construction, :class:`SessionManager` ensures that the backing SQLite
database (default: ``~/assistant/data/memory.db``) exists on disk and that
the ``sessions`` table — the canonical store for the single unified session
shared across all interfaces — has been created with the expected columns:

    sessions
      user_id       TEXT PRIMARY KEY   -- canonical user identifier ("default")
      session_id    TEXT NOT NULL      -- stable UUID4 shared across interfaces
      context       TEXT NOT NULL      -- JSON-serialised session context
      created_at    TEXT NOT NULL      -- ISO 8601 timestamp
      updated_at    TEXT NOT NULL      -- ISO 8601 timestamp

Schema initialisation is **idempotent**: constructing ``SessionManager``
multiple times against the same database file is safe, will not raise, and
will not duplicate or reset the table.

Persistence operations (get / create / update session) are delegated to
:class:`assistant.session_store.SessionStore`, which performs the actual
``CREATE TABLE IF NOT EXISTS`` against the shared ``memory.db`` file. This
keeps the schema definition in a single place while ``SessionManager``
provides the higher-level lifecycle API used by the agent's interfaces.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from assistant.session_store import SessionStore

logger = logging.getLogger(__name__)

# Default database location: ~/assistant/data/memory.db (shared with SessionStore)
_DEFAULT_DB_DIR = Path.home() / "assistant" / "data"
_DEFAULT_DB_PATH = _DEFAULT_DB_DIR / "memory.db"


class SessionManager:
    """
    Manages the lifecycle of the single unified session shared across the
    Web UI, Telegram, and Discord interfaces.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file backing the session store.
        Defaults to ``~/assistant/data/memory.db``. The parent directory and
        the database file (with its ``sessions`` table) are created
        automatically on first run if they do not already exist.

    Attributes
    ----------
    db_path : Path
        Resolved path to the SQLite database file.
    store : SessionStore
        The underlying :class:`~assistant.session_store.SessionStore` used
        for persistence.

    Usage
    -----
    >>> manager = SessionManager(db_path="/tmp/memory.db")
    >>> manager.is_initialized()
    True
    >>> manager.table_exists("sessions")
    True
    >>> sorted(manager.get_columns("sessions"))
    ['context', 'created_at', 'session_id', 'updated_at', 'user_id']
    """

    #: Name of the table created during schema initialisation.
    SESSIONS_TABLE: str = "sessions"

    #: Expected column names of the `sessions` table, in declaration order.
    SESSIONS_COLUMNS: tuple[str, ...] = (
        "user_id",
        "session_id",
        "context",
        "created_at",
        "updated_at",
    )

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            db_path = _DEFAULT_DB_PATH
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # SessionStore.__init__ performs idempotent schema initialisation:
        # it creates the database file (if missing) and runs
        # `CREATE TABLE IF NOT EXISTS sessions (...)`.
        self.store = SessionStore(db_path=self.db_path)

        logger.debug(
            "SessionManager initialised: db_path=%s table=%s",
            self.db_path,
            self.SESSIONS_TABLE,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _connect(self):
        """Yield a connection to the backing SQLite database for introspection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Schema introspection (used by tests / diagnostics)
    # ------------------------------------------------------------------

    def table_exists(self, table_name: str) -> bool:
        """Return ``True`` if *table_name* exists in the backing database."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name = ?",
                (table_name,),
            ).fetchone()
        return row is not None

    def get_columns(self, table_name: str) -> list[str]:
        """
        Return the ordered list of column names for *table_name*.

        Returns an empty list if the table does not exist.
        """
        if not self.table_exists(table_name):
            return []
        with self._connect() as conn:
            # PRAGMA statements do not support parameter binding; table_name
            # is restricted to internally-defined constants (never user input).
            rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        return [row["name"] for row in rows]

    def is_initialized(self) -> bool:
        """
        Return ``True`` if the database file exists on disk and the
        ``sessions`` table has been created with all expected columns.
        """
        if not self.db_path.exists():
            return False
        if not self.table_exists(self.SESSIONS_TABLE):
            return False
        columns = self.get_columns(self.SESSIONS_TABLE)
        return all(col in columns for col in self.SESSIONS_COLUMNS)

    # ------------------------------------------------------------------
    # Session lifecycle (delegated to SessionStore)
    # ------------------------------------------------------------------

    def get_session(self, user_id: str = "default") -> dict[str, Any] | None:
        """Retrieve the stored session context for *user_id*, or ``None``."""
        return self.store.get_session(user_id)

    def create_session(
        self, user_id: str = "default", context: dict[str, Any] | None = None
    ) -> str:
        """
        Create (or update) the session for *user_id*.

        If *context* is omitted, a fresh session is bootstrapped with the
        default hierarchical-memory structure (empty working memory, empty
        session memory, empty long-term memory).

        Returns the ``session_id`` (newly generated, or existing if a
        session for *user_id* already exists).
        """
        if context is None:
            context = {
                "working_memory": [],
                "session_memory": {},
                "long_term_memory": [],
            }
        return self.store.upsert_session(user_id, context)

    def update_session(self, user_id: str, context: dict[str, Any]) -> str:
        """Update the session for *user_id* with new *context*."""
        return self.store.upsert_session(user_id, context)
