"""Discord bot interface adapter (AC6.4).

Provides:

- :func:`handle_message` -- processes an incoming :class:`discord.Message`,
  delegating to :func:`assistant.agent_core.handle_user_message` with
  ``source_interface="discord"``, and replies in the same channel.
- :func:`build_client` -- constructs a configured :class:`discord.Client`
  with :func:`handle_message` wired to the ``on_message`` event.
- :func:`main` -- entry point that reads ``DISCORD_BOT_TOKEN`` from the
  environment and runs the client.

Conversation turns are persisted to (and loaded from) the same
:class:`~assistant.session_store.SessionStore` used by the Web UI and
Telegram adapters, giving cross-interface context continuity (AC6.5).
"""

from __future__ import annotations

import logging
import os

import discord

from assistant.agent_core import handle_user_message
from assistant.llm import OllamaError

logger = logging.getLogger(__name__)

_LLM_UNAVAILABLE_MESSAGE = (
    "Sorry, I'm having trouble reaching my brain right now. Please try again in a moment."
)


async def handle_message(message: discord.Message, client: discord.Client) -> None:
    """Handle an incoming Discord message.

    Messages sent by the bot itself are ignored (to avoid response loops),
    as are empty/whitespace-only messages. Otherwise, the message content is
    run through the shared session-integration core
    (``source_interface="discord"``) and the reply is sent to the same
    channel.

    Args:
        message: The incoming Discord message.
        client: The bot's :class:`discord.Client`, used to identify and
            ignore the bot's own messages.
    """
    if message.author == client.user:
        return
    if not message.content or not message.content.strip():
        return

    try:
        reply = handle_user_message("discord", message.content)
    except OllamaError as exc:
        logger.error("handle_message: Ollama error: %s", exc)
        await message.channel.send(_LLM_UNAVAILABLE_MESSAGE)
        return

    await message.channel.send(reply)


def build_client() -> discord.Client:
    """Build a :class:`discord.Client` with the message handler wired to ``on_message``.

    Returns:
        A configured :class:`discord.Client`, ready for
        :meth:`~discord.Client.run`.
    """
    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)

    @client.event
    async def on_message(message: discord.Message) -> None:
        await handle_message(message, client)

    return client


def main() -> None:
    """Run the Discord bot.

    Reads the bot token from the ``DISCORD_BOT_TOKEN`` environment variable.

    Raises:
        RuntimeError: If ``DISCORD_BOT_TOKEN`` is not set.
    """
    token = os.environ.get("DISCORD_BOT_TOKEN")
    if not token:
        raise RuntimeError("DISCORD_BOT_TOKEN environment variable is not set")

    client = build_client()
    logger.info("Starting Discord bot...")
    client.run(token)


if __name__ == "__main__":
    main()
