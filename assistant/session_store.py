"""
SessionStore — SQLite-backed unified session persistence.

Stores and retrieves cross-interface session context (Web UI, Telegram, Discord)
for the single-user personal assistant agent. Uses memory.db as the backing store.

Schema
------
sessions
  user_id       TEXT PRIMARY KEY
  session_id    TEXT NOT NULL          -- unique UUID per session
  context       TEXT NOT NULL          -- JSON-serialised session context
  created_at    TEXT NOT NULL          -- ISO 8601
  updated_at    TEXT NOT NULL          -- ISO 8601
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default database location: ~/assistant/data/memory.db
_DEFAULT_DB_DIR = Path.home() / "assistant" / "data"
_DEFAULT_DB_PATH = _DEFAULT_DB_DIR / "memory.db"

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    user_id    TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    context    TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def _now_iso() -> str:
    """Return current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


class SessionStore:
    """
    SQLite-backed session store for the personal assistant agent.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Defaults to
        ``~/assistant/data/memory.db``.  The parent directory is created
        automatically if it does not exist.

    Usage
    -----
    >>> store = SessionStore()
    >>> store.upsert_session("user_1", {"working_memory": [], "source_interface": "web_ui"})
    >>> ctx = store.get_session("user_1")
    >>> ctx["source_interface"]
    'web_ui'
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
        # Enable WAL mode for concurrent reader friendliness
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
        """Create the sessions table if it does not yet exist."""
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE_SQL)
        logger.debug("SessionStore initialised at %s", self.db_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_session(self, user_id: str) -> dict[str, Any] | None:
        """
        Retrieve the stored session context for *user_id*.

        Returns
        -------
        dict or None
            The deserialised context dictionary, or ``None`` if no session
            exists yet for this user.

        Raises
        ------
        ValueError
            If *user_id* is empty or not a string.
        """
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")

        with self._connect() as conn:
            row = conn.execute(
                "SELECT context, session_id, created_at, updated_at "
                "FROM sessions WHERE user_id = ?",
                (user_id,),
            ).fetchone()

        if row is None:
            logger.debug("get_session: no session found for user_id=%r", user_id)
            return None

        ctx = json.loads(row["context"])
        # Inject metadata so callers have full picture without a separate query
        ctx.setdefault("_meta", {})
        ctx["_meta"]["session_id"] = row["session_id"]
        ctx["_meta"]["created_at"] = row["created_at"]
        ctx["_meta"]["updated_at"] = row["updated_at"]
        logger.debug("get_session: found session for user_id=%r", user_id)
        return ctx

    def upsert_session(self, user_id: str, context: dict[str, Any]) -> str:
        """
        Insert a new session or update the existing one for *user_id*.

        On insert, a fresh ``session_id`` (UUID4) is generated and stored.
        On update, the existing ``session_id`` is preserved so that the
        session identity is stable across context updates.

        Parameters
        ----------
        user_id:
            Stable identifier for the user (single-user system typically
            passes ``"default"`` or a constant string).
        context:
            Arbitrary JSON-serialisable dict representing the session state
            (working_memory, session_memory, source_interface, etc.).

        Returns
        -------
        str
            The ``session_id`` associated with this session (new or existing).

        Raises
        ------
        ValueError
            If *user_id* is empty or *context* is not a dict.
        TypeError
            If *context* contains non-JSON-serialisable values.
        """
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")
        if not isinstance(context, dict):
            raise ValueError("context must be a dict")

        # Strip internal _meta before persisting (it's injected on read)
        payload = {k: v for k, v in context.items() if k != "_meta"}
        context_json = json.dumps(payload, ensure_ascii=False)
        now = _now_iso()

        with self._connect() as conn:
            existing = conn.execute(
                "SELECT session_id FROM sessions WHERE user_id = ?",
                (user_id,),
            ).fetchone()

            if existing is None:
                session_id = str(uuid.uuid4())
                conn.execute(
                    """
                    INSERT INTO sessions (user_id, session_id, context, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (user_id, session_id, context_json, now, now),
                )
                logger.info(
                    "upsert_session: created new session %s for user_id=%r",
                    session_id,
                    user_id,
                )
            else:
                session_id = existing["session_id"]
                conn.execute(
                    """
                    UPDATE sessions
                    SET context = ?, updated_at = ?
                    WHERE user_id = ?
                    """,
                    (context_json, now, user_id),
                )
                logger.info(
                    "upsert_session: updated session %s for user_id=%r",
                    session_id,
                    user_id,
                )

        return session_id

    def delete_session(self, user_id: str) -> bool:
        """
        Remove the session for *user_id* from the store.

        Returns ``True`` if a row was deleted, ``False`` if none existed.
        Provided for completeness; not required by AC 6.1 but useful in tests.
        """
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")

        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM sessions WHERE user_id = ?", (user_id,)
            )
        deleted = cursor.rowcount > 0
        logger.debug(
            "delete_session: user_id=%r deleted=%s", user_id, deleted
        )
        return deleted

    def list_users(self) -> list[str]:
        """Return all stored user_ids (debugging / admin helper)."""
        with self._connect() as conn:
            rows = conn.execute("SELECT user_id FROM sessions ORDER BY user_id").fetchall()
        return [r["user_id"] for r in rows]

    # ------------------------------------------------------------------
    # Sub-AC 6.3.3 — cross-interface-label save / load
    # ------------------------------------------------------------------

    def save_context(
        self,
        user_id: str,
        messages: list[dict[str, Any]],
        interface_label: str,
    ) -> str:
        """
        Persist a list of conversation messages for *user_id*.

        The storage key is **always** ``user_id`` alone.  The
        ``interface_label`` (e.g. ``'web'``, ``'telegram'``, ``'discord'``)
        is recorded inside the context payload as ``source_interface`` for
        informational purposes but has **no effect** on the lookup key.

        Parameters
        ----------
        user_id:
            Stable user identifier (interface-agnostic).
        messages:
            List of message dicts, each expected to contain at least
            ``{"role": ..., "content": ...}``.
        interface_label:
            The originating interface name.  Stored as metadata only.

        Returns
        -------
        str
            The ``session_id`` for this user's session.

        Raises
        ------
        ValueError
            If *user_id* is empty, *messages* is not a list, or
            *interface_label* is not a non-empty string.
        """
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")
        if not isinstance(interface_label, str) or not interface_label.strip():
            raise ValueError("interface_label must be a non-empty string")

        # Retrieve existing context so that other fields are not clobbered.
        existing_ctx = self.get_session(user_id) or {}
        # Strip _meta injected by get_session before merging
        existing_ctx.pop("_meta", None)

        context = {
            **existing_ctx,
            "working_memory": messages,
            "source_interface": interface_label,
        }
        return self.upsert_session(user_id, context)

    def load_context(
        self,
        user_id: str,
        interface_label: str,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the stored conversation messages for *user_id*.

        The lookup is performed **exclusively on** ``user_id`` — the
        ``interface_label`` argument is accepted for API symmetry and may be
        used by callers to annotate the retrieval, but it does **not** filter
        or alter which row is returned.  This is the core cross-interface
        guarantee: a context saved under label A is fully accessible under
        label B for the same ``user_id``.

        Parameters
        ----------
        user_id:
            Stable user identifier (interface-agnostic).
        interface_label:
            The interface performing the load.  Accepted but **ignored** for
            the database query.

        Returns
        -------
        list
            The ``working_memory`` list from the stored context, or an empty
            list if no session exists for this user.

        Raises
        ------
        ValueError
            If *user_id* is empty or *interface_label* is not a non-empty string.
        """
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")
        if not isinstance(interface_label, str) or not interface_label.strip():
            raise ValueError("interface_label must be a non-empty string")

        ctx = self.get_session(user_id)
        if ctx is None:
            logger.debug(
                "load_context: no session for user_id=%r (interface_label=%r)",
                user_id,
                interface_label,
            )
            return []

        messages = ctx.get("working_memory", [])
        logger.debug(
            "load_context: user_id=%r interface_label=%r → %d messages",
            user_id,
            interface_label,
            len(messages),
        )
        return messages
