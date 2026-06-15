"""Working memory compression (assistant.compression).

Implements:

- **AC16** -- the ``/compress`` command: manually fold older messages from
  ``working_memory`` into a running ``session_memory["summary"]``.
- **AC17** -- automatic working-memory compression: when
  ``working_memory`` grows past :data:`_AUTO_COMPRESS_THRESHOLD` messages,
  the oldest messages are folded into the summary the same way.

In both cases, the most recent messages are kept verbatim in
``working_memory`` (data-fidelity for recent context) while older messages
are summarized via the local LLM and merged into any existing summary, so
context keeps shrinking back down rather than growing unbounded.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

import httpx

from assistant.llm import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OllamaError,
    OllamaParseError,
    parse_ollama_summary,
)
from assistant.session_store import SessionStore

logger = logging.getLogger(__name__)

_MAX_RETRIES: int = 2
_BACKOFF_BASE_SECONDS: float = 1.0
_REQUEST_TIMEOUT_SECONDS: float = 30.0

#: AC17 -- auto-compression triggers once working_memory exceeds this many messages.
_AUTO_COMPRESS_THRESHOLD: int = 20

#: Number of most-recent messages retained in working_memory after compression.
_KEEP_RECENT_MESSAGES: int = 6


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_compression_prompt(messages: list[dict[str, Any]], previous_summary: str = "") -> str:
    """Build an LLM prompt asking for a summary of *messages*.

    Pure function -- builds prompt text only, no I/O.

    Args:
        messages: Conversation messages to summarize, each expected to have
            ``"role"`` and ``"content"`` keys. Reproduced verbatim in the
            prompt (data-fidelity).
        previous_summary: An existing running summary to fold into, if any.

    Returns:
        The prompt string to send to the LLM.
    """
    lines: list[str] = []

    if previous_summary:
        lines.append("Existing summary of earlier conversation:")
        lines.append(previous_summary)
        lines.append("")

    lines.append(
        "Summarize the following conversation between a user and an AI "
        "assistant. Preserve important facts, decisions, action items, and "
        "any data the user shared (names, dates, preferences, numbers). "
        "Combine this with the existing summary above (if any) into a single "
        "updated summary. Be concise but do not omit information that may be "
        "needed later."
    )
    lines.append("")
    lines.append("Conversation:")
    for message in messages:
        role = message.get("role", "unknown")
        content = message.get("content", "")
        lines.append(f"{role}: {content}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM-backed summarization
# ---------------------------------------------------------------------------


def summarize_messages(
    messages: list[dict[str, Any]],
    previous_summary: str = "",
    *,
    base_url: str = OLLAMA_BASE_URL,
    model: str = OLLAMA_MODEL,
    max_retries: int = _MAX_RETRIES,
    base_backoff: float = _BACKOFF_BASE_SECONDS,
) -> str:
    """Summarize *messages* (folding in *previous_summary*) using the local LLM.

    Retry policy: up to *max_retries* retries with exponential backoff on
    connection errors and HTTP error responses. A response that cannot be
    parsed (:class:`~assistant.llm.OllamaParseError`) is non-retryable.

    Args:
        messages: Messages to summarize. If empty, *previous_summary* is
            returned unchanged and no LLM call is made.
        previous_summary: An existing running summary to fold into, if any.
        base_url: Base URL of the Ollama server.
        model: Name of the Ollama model to use.
        max_retries: Number of retries on transient errors.
        base_backoff: Base delay (seconds) for exponential backoff.

    Returns:
        The updated summary text, verbatim from the LLM (data-fidelity).

    Raises:
        OllamaError: When all retry attempts are exhausted, or when the
            response payload cannot be parsed.
    """
    if not messages:
        return previous_summary

    prompt = build_compression_prompt(messages, previous_summary)
    url = f"{base_url.rstrip('/')}/api/generate"
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }

    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        if attempt > 0:
            delay = base_backoff * (2 ** (attempt - 1))
            logger.warning(
                "summarize_messages: attempt %d failed (%s), retrying in %.1fs",
                attempt,
                last_error,
                delay,
            )
            time.sleep(delay)

        try:
            response = httpx.post(url, json=payload, timeout=_REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            data: dict[str, Any] = response.json()
            return parse_ollama_summary(data)
        except httpx.HTTPStatusError as exc:
            last_error = exc
            logger.error("summarize_messages: HTTP error: %s", exc)
        except httpx.RequestError as exc:
            last_error = exc
            logger.error("summarize_messages: connection error: %s", exc)
        except OllamaParseError as exc:
            raise OllamaError(
                f"Working memory compression failed: unexpected response "
                f"format from Ollama — {exc}"
            ) from exc
        except ValueError as exc:
            raise OllamaError(
                f"Working memory compression failed: could not decode Ollama "
                f"response as JSON — {exc}"
            ) from exc

    raise OllamaError(
        f"Working memory compression failed after {max_retries + 1} attempts. "
        f"(Last error: {last_error})"
    )


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------


def compress_working_memory(
    context: dict[str, Any],
    *,
    keep_recent: int = _KEEP_RECENT_MESSAGES,
    summarizer: Callable[[list[dict[str, Any]], str], str] = summarize_messages,
) -> dict[str, Any]:
    """Fold older messages from ``context["working_memory"]`` into a summary.

    The most recent *keep_recent* messages are kept verbatim in
    ``working_memory``. Everything older is summarized via *summarizer* and
    merged into ``context["session_memory"]["summary"]``.

    Args:
        context: Session context dict (as returned by
            :meth:`assistant.session_store.SessionStore.get_session`,
            without its injected ``"_meta"`` key).
        keep_recent: Number of most-recent messages to keep verbatim. If
            ``len(working_memory) <= keep_recent``, *context* is returned
            unchanged (nothing to compress).
        summarizer: Callable taking ``(messages_to_summarize,
            previous_summary)`` and returning the updated summary text.
            Defaults to :func:`summarize_messages`.

    Returns:
        A new context dict with ``working_memory`` trimmed and
        ``session_memory["summary"]`` updated. *context* itself is not
        mutated. If there was nothing to compress, *context* is returned
        as-is.
    """
    working_memory = context.get("working_memory", [])

    if len(working_memory) <= keep_recent:
        return context

    if keep_recent > 0:
        to_summarize = working_memory[:-keep_recent]
        recent = working_memory[-keep_recent:]
    else:
        to_summarize = working_memory
        recent = []

    session_memory = dict(context.get("session_memory", {}))
    previous_summary = session_memory.get("summary", "")
    session_memory["summary"] = summarizer(to_summarize, previous_summary)

    new_context = dict(context)
    new_context["working_memory"] = recent
    new_context["session_memory"] = session_memory

    logger.info(
        "compress_working_memory: folded %d message(s) into summary, kept %d",
        len(to_summarize),
        len(recent),
    )
    return new_context


def maybe_auto_compress(
    context: dict[str, Any],
    *,
    threshold: int = _AUTO_COMPRESS_THRESHOLD,
    keep_recent: int = _KEEP_RECENT_MESSAGES,
    summarizer: Callable[[list[dict[str, Any]], str], str] = summarize_messages,
) -> dict[str, Any]:
    """AC17: compress ``working_memory`` if it exceeds *threshold* messages.

    Args:
        context: Session context dict.
        threshold: If ``len(working_memory) <= threshold``, *context* is
            returned unchanged and *summarizer* is never called.
        keep_recent: Forwarded to :func:`compress_working_memory`.
        summarizer: Forwarded to :func:`compress_working_memory`.

    Returns:
        The (possibly compressed) context dict.
    """
    working_memory = context.get("working_memory", [])
    if len(working_memory) <= threshold:
        return context

    return compress_working_memory(context, keep_recent=keep_recent, summarizer=summarizer)


# ---------------------------------------------------------------------------
# /compress command handler
# ---------------------------------------------------------------------------


def handle_compress_command(
    user_id: str,
    command_text: str,
    store: SessionStore,
    *,
    keep_recent: int = _KEEP_RECENT_MESSAGES,
    summarizer: Callable[[list[dict[str, Any]], str], str] = summarize_messages,
) -> str:
    """Handle a ``/compress`` command and return a user-facing response string.

    Loads the session for *user_id*, folds older ``working_memory`` messages
    into ``session_memory["summary"]`` (keeping the *keep_recent* most recent
    messages verbatim), persists the updated session, and reports how many
    messages were compressed.

    Args:
        user_id: Stable identifier for the user issuing the command.
        command_text: The raw command text, e.g. ``"/compress"``.
        store: The :class:`~assistant.session_store.SessionStore` instance
            to operate on.
        keep_recent: Forwarded to :func:`compress_working_memory`.
        summarizer: Forwarded to :func:`compress_working_memory`.

    Returns:
        A human-readable response to send back to the user.

    Raises:
        ValueError: If *command_text* does not start with ``"/compress"``.
    """
    if not command_text.startswith("/compress"):
        raise ValueError("command_text must start with '/compress'")

    context = store.get_session(user_id)
    if context is None:
        return "There is no conversation history to compress yet."

    context = dict(context)
    context.pop("_meta", None)
    working_memory = context.get("working_memory", [])

    if len(working_memory) <= keep_recent:
        return "There isn't enough conversation history to compress yet."

    new_context = compress_working_memory(context, keep_recent=keep_recent, summarizer=summarizer)
    store.upsert_session(user_id, new_context)

    removed = len(working_memory) - len(new_context["working_memory"])
    return f"Compressed {removed} older message(s) into the session summary."
