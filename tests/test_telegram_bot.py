"""Tests for the Telegram bot interface adapter (assistant.interfaces.telegram_bot).

Covers:
- handle_message: extracts text, delegates to
  handle_user_message("telegram", ...), and replies with the result;
  ignores updates with no message / empty / whitespace-only text; replies
  with a friendly error message (instead of crashing) on OllamaError.
- build_application: input validation, registers the message handler.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from assistant.interfaces.telegram_bot import (
    _LLM_UNAVAILABLE_MESSAGE,
    build_application,
    handle_message,
    send_message,
)
from assistant.llm import OllamaError


def _make_update(text: str | None):
    update = MagicMock()
    if text is None:
        update.message = None
    else:
        update.message = MagicMock()
        update.message.text = text
        update.message.reply_text = AsyncMock()
    return update


class TestHandleMessage:
    @patch("assistant.interfaces.telegram_bot.handle_user_message")
    async def test_replies_with_result(self, mock_handle):
        mock_handle.return_value = "Hi there!"
        update = _make_update("Hello")
        context = MagicMock()

        await handle_message(update, context)

        mock_handle.assert_called_once_with("telegram", "Hello")
        update.message.reply_text.assert_awaited_once_with("Hi there!")

    @patch("assistant.interfaces.telegram_bot.handle_user_message")
    async def test_ignores_whitespace_only_text(self, mock_handle):
        update = _make_update("   ")
        context = MagicMock()

        await handle_message(update, context)

        mock_handle.assert_not_called()
        update.message.reply_text.assert_not_called()

    @patch("assistant.interfaces.telegram_bot.handle_user_message")
    async def test_ignores_update_without_message(self, mock_handle):
        update = _make_update(None)
        context = MagicMock()

        await handle_message(update, context)

        mock_handle.assert_not_called()

    @patch("assistant.interfaces.telegram_bot.handle_user_message")
    async def test_ignores_update_with_no_text(self, mock_handle):
        update = MagicMock()
        update.message = MagicMock()
        update.message.text = None
        update.message.reply_text = AsyncMock()
        context = MagicMock()

        await handle_message(update, context)

        mock_handle.assert_not_called()
        update.message.reply_text.assert_not_called()

    @patch("assistant.interfaces.telegram_bot.handle_user_message")
    async def test_replies_with_friendly_message_on_ollama_error(self, mock_handle):
        mock_handle.side_effect = OllamaError("connection refused")
        update = _make_update("Hello")
        context = MagicMock()

        await handle_message(update, context)

        update.message.reply_text.assert_awaited_once_with(_LLM_UNAVAILABLE_MESSAGE)


class TestSendMessage:
    @patch("assistant.interfaces.telegram_bot.Bot")
    async def test_sends_message_via_bot(self, mock_bot_cls):
        mock_bot = MagicMock()
        mock_bot.send_message = AsyncMock()
        mock_bot_cls.return_value = mock_bot

        await send_message(12345, "Hello there", token="123:fake-token")

        mock_bot_cls.assert_called_once_with(token="123:fake-token")
        mock_bot.send_message.assert_awaited_once_with(chat_id=12345, text="Hello there")

    @patch("assistant.interfaces.telegram_bot.Bot")
    async def test_uses_token_env_var_when_not_provided(self, mock_bot_cls, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "env-token")
        mock_bot = MagicMock()
        mock_bot.send_message = AsyncMock()
        mock_bot_cls.return_value = mock_bot

        await send_message(12345, "Hello there")

        mock_bot_cls.assert_called_once_with(token="env-token")

    async def test_no_token_raises_runtime_error(self, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)

        with pytest.raises(RuntimeError):
            await send_message(12345, "Hello there")


class TestBuildApplication:
    def test_empty_token_raises_value_error(self):
        with pytest.raises(ValueError):
            build_application("")

    def test_whitespace_token_raises_value_error(self):
        with pytest.raises(ValueError):
            build_application("   ")

    def test_returns_application_with_handler_registered(self):
        application = build_application("123:fake-token")

        handlers = application.handlers[0]
        assert len(handlers) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
