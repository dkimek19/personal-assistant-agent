"""End-to-end cross-interface context continuity test (AC6.5).

Drives a single conversation through all three real interface adapter entry
points -- the Web UI's ``POST /chat`` endpoint, the Telegram
``handle_message`` callback, and the Discord ``handle_message`` callback --
backed by one shared :class:`~assistant.session_store.SessionStore`, and
verifies that:

- All three interfaces resolve to the same canonical session
  (``session_id``).
- ``working_memory`` accumulates across interfaces in turn order, each
  message tagged with the originating ``source_interface``.
- Each adapter's responder sees the *full* conversation history so far,
  including turns from other interfaces -- i.e. a conversation begun on one
  interface continues seamlessly on another.

The Telegram and Discord adapters call
:func:`assistant.agent_core.handle_user_message` with their production
defaults (no injectable store/responder). To exercise the *real*
``handle_user_message`` logic against the test's shared store and a
deterministic stub responder, this test patches each adapter's
``handle_user_message`` reference with a thin wrapper that forwards to the
real function with ``store`` and ``responder`` supplied. The Web UI adapter
is exercised via its existing FastAPI dependency overrides.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from assistant.agent_core import handle_user_message as real_handle_user_message
from assistant.interfaces import discord_bot, telegram_bot
from assistant.interfaces.web_ui import app, get_responder, get_store
from assistant.session_resolver import session_id_resolver
from assistant.session_store import SessionStore


@pytest.fixture
def store(tmp_path):
    return SessionStore(db_path=tmp_path / "memory.db")


def _stub_responder(messages):
    return f"reply to: {messages[-1]['content']}"


def _forwarding_handler(store, responder):
    """A drop-in replacement for handle_user_message that injects store/responder."""

    def _handler(source_interface, message_text):
        return real_handle_user_message(
            source_interface, message_text, store=store, responder=responder
        )

    return _handler


@pytest.fixture
def web_client(store):
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_responder] = lambda: _stub_responder
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


async def _send_telegram_message(store, text: str) -> str:
    update = MagicMock()
    update.message = MagicMock()
    update.message.text = text
    update.message.reply_text = AsyncMock()

    with patch.object(telegram_bot, "handle_user_message", side_effect=_forwarding_handler(store, _stub_responder)):
        await telegram_bot.handle_message(update, MagicMock())

    return update.message.reply_text.call_args.args[0]


async def _send_discord_message(store, text: str) -> str:
    client = MagicMock()
    client.user = MagicMock(name="bot_user")

    message = MagicMock()
    message.content = text
    message.author = MagicMock(name="human_user")
    message.channel.send = AsyncMock()

    with patch.object(discord_bot, "handle_user_message", side_effect=_forwarding_handler(store, _stub_responder)):
        await discord_bot.handle_message(message, client)

    return message.channel.send.call_args.args[0]


class TestCrossInterfaceContinuity:
    def test_all_interfaces_resolve_to_same_session_id(self, store):
        web = session_id_resolver("web_ui", store=store)
        telegram = session_id_resolver("telegram", store=store)
        discord_session = session_id_resolver("discord", store=store)

        assert web.session_id == telegram.session_id == discord_session.session_id
        assert web.user_id == telegram.user_id == discord_session.user_id == "default"

    async def test_conversation_continues_seamlessly_across_all_three_interfaces(self, store, web_client):
        web_response = web_client.post("/chat", json={"message": "Hello from web"})
        assert web_response.status_code == 200
        assert web_response.json()["reply"] == "reply to: Hello from web"

        telegram_reply = await _send_telegram_message(store, "Hello from telegram")
        assert telegram_reply == "reply to: Hello from telegram"

        discord_reply = await _send_discord_message(store, "Hello from discord")
        assert discord_reply == "reply to: Hello from discord"

        ctx = store.get_session("default")
        working_memory = ctx["working_memory"]

        assert len(working_memory) == 6
        assert [m["source_interface"] for m in working_memory] == [
            "web_ui",
            "web_ui",
            "telegram",
            "telegram",
            "discord",
            "discord",
        ]
        assert [m["role"] for m in working_memory] == [
            "user",
            "assistant",
            "user",
            "assistant",
            "user",
            "assistant",
        ]
        assert working_memory[0]["content"] == "Hello from web"
        assert working_memory[2]["content"] == "Hello from telegram"
        assert working_memory[4]["content"] == "Hello from discord"

        # The session itself is the single canonical session shared by all interfaces.
        assert ctx["_meta"]["session_id"] == session_id_resolver("web_ui", store=store).session_id

    async def test_later_interface_responder_sees_earlier_interfaces_history(self, store, web_client):
        seen_histories: list[list[dict]] = []

        def recording_responder(messages):
            seen_histories.append([dict(m) for m in messages])
            return _stub_responder(messages)

        app.dependency_overrides[get_responder] = lambda: recording_responder
        web_client.post("/chat", json={"message": "My favorite color is blue"})

        with patch.object(
            telegram_bot,
            "handle_user_message",
            side_effect=_forwarding_handler(store, recording_responder),
        ):
            update = MagicMock()
            update.message = MagicMock()
            update.message.text = "What's my favorite color?"
            update.message.reply_text = AsyncMock()
            await telegram_bot.handle_message(update, MagicMock())

        # The Telegram-side responder call must see the Web-originated turn.
        telegram_history = seen_histories[-1]
        assert telegram_history[0]["content"] == "My favorite color is blue"
        assert telegram_history[0]["source_interface"] == "web_ui"
        assert telegram_history[-1]["content"] == "What's my favorite color?"
        assert telegram_history[-1]["source_interface"] == "telegram"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
