"""
session_id_resolver — maps incoming interface requests to a canonical session.

Sub-AC 6.2: Unified identity key across Web UI, Telegram, and Discord interfaces.

Design
------
This is a **single-user** system.  Every interface (web_ui, telegram, discord)
maps to the same canonical ``user_id`` (``"default"``), which in turn is tied
to a single stable ``session_id`` stored in the SessionStore.

Conceptually:
    web_ui    ─┐
    telegram  ─┼──► canonical user_id="default" ──► session_id (stable UUID)
    discord   ─┘

The ``interface_user_id`` parameter accepts interface-specific identifiers
(e.g. Telegram chat IDs, Discord member IDs, browser fingerprints) for
forward-compatibility, but they do not influence routing in a single-user model.
"""

from __future__ import annotations

from typing import NamedTuple

from assistant.session_store import SessionStore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: The single canonical user_id for the single-user assistant.
CANONICAL_USER_ID: str = "default"

#: All recognised interface names.
VALID_INTERFACES: frozenset[str] = frozenset({"web_ui", "telegram", "discord"})


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class ResolvedSession(NamedTuple):
    """Resolved identity for an incoming request.

    Attributes
    ----------
    user_id:
        Canonical user identifier (always ``"default"`` in the single-user
        model).
    session_id:
        Stable UUID tied to *user_id* in the :class:`~assistant.session_store.SessionStore`.
        Guaranteed to be the same for all interfaces that share the same
        ``user_id``.
    """

    user_id: str
    session_id: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def session_id_resolver(
    source_interface: str,
    interface_user_id: str | None = None,
    *,
    store: SessionStore | None = None,
) -> ResolvedSession:
    """Map an incoming request from any interface to the canonical session.

    This function is the **unified identity key** for the personal assistant
    agent.  Because the system is single-user, ``web_ui``, ``telegram``, and
    ``discord`` all resolve to the same ``user_id`` (``"default"``) and
    therefore to the same ``session_id``.

    Parameters
    ----------
    source_interface:
        The originating interface.  Must be one of ``"web_ui"``,
        ``"telegram"``, or ``"discord"``.
    interface_user_id:
        Optional interface-specific user identifier (e.g. Telegram chat ID,
        Discord member ID, web client token).  Accepted for forward-
        compatibility but ignored in the single-user routing model.
    store:
        Optional :class:`~assistant.session_store.SessionStore` instance.
        If *None*, a default store is created pointing at
        ``~/assistant/data/memory.db``.  Pass an explicit *store* in tests
        to use an isolated temporary database.

    Returns
    -------
    ResolvedSession
        A named-tuple with two fields:

        * ``user_id``    — always ``"default"`` in the single-user model.
        * ``session_id`` — the stable UUID from the session store, shared
          across all interfaces.

    Raises
    ------
    ValueError
        If *source_interface* is not one of the recognised interface names.

    Examples
    --------
    >>> from assistant.session_resolver import session_id_resolver
    >>> r1 = session_id_resolver("web_ui")
    >>> r2 = session_id_resolver("telegram")
    >>> r3 = session_id_resolver("discord")
    >>> assert r1.session_id == r2.session_id == r3.session_id
    """
    if source_interface not in VALID_INTERFACES:
        raise ValueError(
            f"Unknown interface: {source_interface!r}. "
            f"Must be one of {sorted(VALID_INTERFACES)}"
        )

    if store is None:
        store = SessionStore()

    # Single-user: all interfaces resolve to the canonical user_id.
    existing = store.get_session(CANONICAL_USER_ID)
    if existing is None:
        # Bootstrap a fresh session with minimal context.
        session_id = store.upsert_session(
            CANONICAL_USER_ID,
            {
                "working_memory": [],
                "source_interface": source_interface,
                "session_memory": {},
                "long_term_memory": [],
            },
        )
    else:
        session_id = existing["_meta"]["session_id"]

    return ResolvedSession(user_id=CANONICAL_USER_ID, session_id=session_id)
