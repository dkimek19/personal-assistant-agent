"""Tests for the Discord bot interface adapter (assistant.interfaces.discord_bot).

Covers:
- handle_message: delegates to handle_user_message("discord", ...) and
  replies in the same channel; ignores the bot's own messages and
  empty/whitespace-only content; replies with a friendly error message
  (instead of crashing) on OllamaError.
- build_client: registers an on_message handler.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from assistant.interfaces.discord_bot import (
    _LLM_UNAVAILABLE_MESSAGE,
    build_client,
    handle_message,
)
from assistant.llm import OllamaError


def _make_message(content: str | None, author=None):
    message = MagicMock()
    message.content = content
    message.author = author if author is not None else MagicMock()
    message.channel.send = AsyncMock()
    return message


def _make_client():
    client = MagicMock()
    client.user = MagicMock(name="bot_user")
    return client


class TestHandleMessage:
    @patch("assistant.interfaces.discord_bot.handle_user_message")
    async def test_replies_with_result(self, mock_handle):
        mock_handle.return_value = "Hi there!"
        client = _make_client()
        message = _make_message("Hello")

        await handle_message(message, client)

        mock_handle.assert_called_once_with("discord", "Hello")
        message.channel.send.assert_awaited_once_with("Hi there!")

    @patch("assistant.interfaces.discord_bot.handle_user_message")
    async def test_ignores_message_from_bot_itself(self, mock_handle):
        client = _make_client()
        message = _make_message("Hi there!", author=client.user)

        await handle_message(message, client)

        mock_handle.assert_not_called()
        message.channel.send.assert_not_called()

    @patch("assistant.interfaces.discord_bot.handle_user_message")
    async def test_ignores_whitespace_only_content(self, mock_handle):
        client = _make_client()
        message = _make_message("   ")

        await handle_message(message, client)

        mock_handle.assert_not_called()
        message.channel.send.assert_not_called()

    @patch("assistant.interfaces.discord_bot.handle_user_message")
    async def test_ignores_empty_content(self, mock_handle):
        client = _make_client()
        message = _make_message("")

        await handle_message(message, client)

        mock_handle.assert_not_called()
        message.channel.send.assert_not_called()

    @patch("assistant.interfaces.discord_bot.handle_user_message")
    async def test_replies_with_friendly_message_on_ollama_error(self, mock_handle):
        mock_handle.side_effect = OllamaError("connection refused")
        client = _make_client()
        message = _make_message("Hello")

        await handle_message(message, client)

        message.channel.send.assert_awaited_once_with(_LLM_UNAVAILABLE_MESSAGE)


class TestBuildClient:
    def test_returns_client_with_on_message_handler(self):
        client = build_client()

        assert callable(client.on_message)

    def test_message_content_intent_enabled(self):
        client = build_client()

        assert client.intents.message_content is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
