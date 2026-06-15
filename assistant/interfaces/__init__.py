"""Interface adapters (Web UI, Telegram, Discord) for the personal assistant agent.

Each adapter is a thin wrapper around
:func:`assistant.agent_core.handle_user_message`, which integrates with the
shared :class:`~assistant.session_store.SessionStore` so conversation turns
from any interface are persisted to, and loaded from, the same canonical
session (AC6.2-6.5).
"""

from __future__ import annotations
