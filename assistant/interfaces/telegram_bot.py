"""Telegram bot interface adapter (AC6.3).

Provides:

- :func:`handle_message` -- a python-telegram-bot ``MessageHandler``
  callback that delegates incoming text messages to
  :func:`assistant.agent_core.handle_user_message` with
  ``source_interface="telegram"``, and replies with the result.
- :func:`build_application` -- constructs a configured
  :class:`telegram.ext.Application` with :func:`handle_message` registered
  for incoming text messages.
- :func:`send_message` -- sends a proactive (non-reply) message to a chat,
  used by background jobs such as calendar alerts (AC21) and disk usage
  warnings (AC22).
- :func:`main` -- entry point that reads ``TELEGRAM_BOT_TOKEN`` from the
  environment and runs the bot via long polling.

Conversation turns are persisted to (and loaded from) the same
:class:`~assistant.session_store.SessionStore` used by the Web UI and
Discord adapters, giving cross-interface context continuity (AC6.5).
"""

from __future__ import annotations

import logging
import os

from telegram import Bot, Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from assistant.agent_core import handle_user_message
from assistant.llm import OllamaError

logger = logging.getLogger(__name__)

_LLM_UNAVAILABLE_MESSAGE = (
    "Sorry, I'm having trouble reaching my brain right now. Please try again in a moment."
)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle an incoming Telegram text message.

    Extracts the message text, runs it through the shared
    session-integration core (``source_interface="telegram"``), and replies
    with the result. Updates without a text message (e.g. photos, stickers)
    and empty/whitespace-only text are ignored.

    Args:
        update: The incoming Telegram update.
        context: The python-telegram-bot callback context (unused).
    """
    message = update.message
    if message is None or not message.text or not message.text.strip():
        return

    try:
        reply = handle_user_message("telegram", message.text)
    except OllamaError as exc:
        logger.error("handle_message: Ollama error: %s", exc)
        await message.reply_text(_LLM_UNAVAILABLE_MESSAGE)
        return

    await message.reply_text(reply)


def build_application(token: str) -> Application:
    """Build a Telegram :class:`~telegram.ext.Application` with the message handler registered.

    Args:
        token: The Telegram bot token (from @BotFather).

    Returns:
        A configured :class:`~telegram.ext.Application`, ready for
        :meth:`~telegram.ext.Application.run_polling`.

    Raises:
        ValueError: If *token* is empty or whitespace-only.
    """
    if not token or not token.strip():
        raise ValueError("token must be a non-empty string")

    application = Application.builder().token(token).build()
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    return application


async def send_message(chat_id: int | str, text: str, *, token: str | None = None) -> None:
    """Send a proactive (non-reply) message to a Telegram chat.

    Unlike :func:`handle_message`, this is not triggered by an incoming
    update -- it is used by background jobs (calendar alerts, AC21; disk
    usage warnings, AC22) to push a notification to the user.

    Args:
        chat_id: The Telegram chat ID to send the message to.
        text: The message text.
        token: The Telegram bot token. Defaults to the
            ``TELEGRAM_BOT_TOKEN`` environment variable.

    Raises:
        RuntimeError: If no token is provided and ``TELEGRAM_BOT_TOKEN`` is
            not set.
    """
    token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN environment variable is not set")

    bot = Bot(token=token)
    await bot.send_message(chat_id=chat_id, text=text)


def main() -> None:
    """Run the Telegram bot via long polling.

    Reads the bot token from the ``TELEGRAM_BOT_TOKEN`` environment variable.

    Raises:
        RuntimeError: If ``TELEGRAM_BOT_TOKEN`` is not set.
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN environment variable is not set")

    application = build_application(token)
    logger.info("Starting Telegram bot (long polling)...")
    application.run_polling()


if __name__ == "__main__":
    main()
