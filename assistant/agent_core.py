"""Shared session-integration core for interface adapters (AC6.2-6.5).

:func:`handle_user_message` is the single entry point used by every
interface adapter (Web UI, Telegram, Discord). It:

1. Resolves the canonical session via
   :func:`assistant.session_resolver.session_id_resolver`.
2. If the message is a recognised slash command (``/note``, ``/remember``,
   ``/compress`` -- AC14-16), routes it directly to the matching handler and
   returns its response, without involving the LLM.
3. Otherwise, loads the shared ``working_memory`` from the
   :class:`~assistant.session_store.SessionStore`, appends the incoming user
   message, and generates a reply via an injectable *responder* (default:
   :func:`default_responder`).

   :func:`default_responder` calls the local Ollama ``/api/chat`` endpoint
   with :data:`assistant.tools.definitions.TOOL_DEFINITIONS`. If the model
   responds with one or more ``tool_calls``, each is executed via
   :func:`assistant.tools.dispatch.dispatch_tool`, the assistant's tool-call
   message and the tool results are appended to ``working_memory``, and the
   LLM is re-called for a final natural-language reply -- looping until the
   model stops requesting tools or :data:`_MAX_TOOL_ITERATIONS` is reached
   (AC1-3). A tool that fails reports ``tool_status="failed"`` and
   ``tool_error_message`` in its result message, so the LLM can surface a
   clear error to the user (AC23).
4. Appends the reply (command response or LLM reply) to ``working_memory``.
5. Auto-compresses ``working_memory`` if it has grown too large (AC17).
6. Persists the updated context back to the store.

Because every interface resolves to the same canonical ``user_id`` /
``session_id`` and always loads-then-saves the *shared* ``working_memory``,
a conversation begun on one interface continues seamlessly on another
(AC6.5).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

import httpx

from assistant.compression import handle_compress_command, maybe_auto_compress
from assistant.llm import OLLAMA_BASE_URL, OLLAMA_MODEL, OllamaError
from assistant.long_term_memory import LongTermMemoryStore, handle_remember_command
from assistant.notes import NoteStore, handle_note_command
from assistant.session_resolver import session_id_resolver
from assistant.session_store import SessionStore
from assistant.tools.definitions import TOOL_DEFINITIONS
from assistant.tools.dispatch import dispatch_tool

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT_SECONDS: float = 30.0

#: AC1-3 -- maximum tool-call round trips per user message before forcing a
#: final, tool-free reply. Bounds worst-case latency on a model that keeps
#: requesting tools.
_MAX_TOOL_ITERATIONS: int = 5

#: A responder takes the current `working_memory` (including the just-appended
#: user message) and returns the assistant's reply text. It may append
#: additional messages (tool calls/results) to `working_memory` in place.
Responder = Callable[[list[dict[str, Any]]], str]


def _to_ollama_message(message: dict[str, Any]) -> dict[str, Any]:
    """Project a `working_memory` entry to the keys Ollama's ``/api/chat`` expects.

    Drops bookkeeping keys such as ``"source_interface"`` while preserving
    ``"tool_calls"``/``"name"``/``"tool_call_id"`` when present, so
    assistant tool-call messages and tool-result messages survive a
    round trip through `working_memory`.
    """
    out: dict[str, Any] = {"role": message["role"], "content": message.get("content") or ""}
    for key in ("tool_calls", "name", "tool_call_id"):
        if key in message:
            out[key] = message[key]
    return out


def _chat(
    messages: list[dict[str, Any]],
    *,
    base_url: str,
    model: str,
    tools: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """POST *messages* to Ollama's ``/api/chat`` and return the raw assistant message dict.

    Args:
        messages: Conversation messages, most recent last.
        base_url: Base URL of the Ollama server.
        model: Name of the Ollama model to use.
        tools: Tool definitions to offer the model, or ``None``/empty to
            disable tool calling for this request.

    Returns:
        The ``"message"`` object from the Ollama response, e.g.
        ``{"role": "assistant", "content": "...", "tool_calls": [...]}``.

    Raises:
        OllamaError: If the request fails, the response is not valid JSON,
            or the response does not contain a ``"message"`` object.
    """
    url = f"{base_url.rstrip('/')}/api/chat"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [_to_ollama_message(m) for m in messages],
        "stream": False,
    }
    if tools:
        payload["tools"] = tools

    try:
        response = httpx.post(url, json=payload, timeout=_REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        data: dict[str, Any] = response.json()
    except httpx.HTTPError as exc:
        raise OllamaError(f"Chat request to Ollama failed: {exc}") from exc
    except ValueError as exc:
        raise OllamaError(f"Could not decode Ollama response as JSON: {exc}") from exc

    message = data.get("message")
    if not isinstance(message, dict):
        raise OllamaError(f"Unexpected response from Ollama: missing 'message' object. Received: {data!r}")
    return message


def _execute_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run *tool_calls* via :func:`dispatch_tool` and build ``"tool"`` result messages.

    Args:
        tool_calls: The model's requested tool calls, each shaped like
            ``{"function": {"name": ..., "arguments": {...}}}``.

    Returns:
        One ``{"role": "tool", "name": ..., "content": ...}`` message per
        tool call, with ``"content"`` holding the JSON-serialised result of
        :func:`dispatch_tool` (including ``tool_status`` and, on failure,
        ``tool_error_message`` -- AC23).
    """
    results = []
    for call in tool_calls:
        function = call.get("function", {})
        tool_name = function.get("name", "")
        tool_input = function.get("arguments", {})
        if isinstance(tool_input, str):
            try:
                tool_input = json.loads(tool_input)
            except ValueError:
                tool_input = {}

        result = dispatch_tool(tool_name, tool_input)
        results.append({"role": "tool", "name": tool_name, "content": json.dumps(result)})
    return results


def default_responder(
    messages: list[dict[str, Any]],
    *,
    base_url: str = OLLAMA_BASE_URL,
    model: str = OLLAMA_MODEL,
    tools: list[dict[str, Any]] | None = TOOL_DEFINITIONS,
    max_tool_iterations: int = _MAX_TOOL_ITERATIONS,
) -> str:
    """Generate a reply for *messages* via the local Ollama ``/api/chat`` endpoint.

    If the model responds with ``tool_calls``, each is executed via
    :func:`dispatch_tool`, the assistant's tool-call message and the tool
    results are appended to *messages* **in place**, and the LLM is
    re-queried for a final reply -- looping until the model stops requesting
    tools or *max_tool_iterations* round trips have been made (AC1-3, AC23).

    Args:
        messages: Conversation messages (each with ``"role"`` and
            ``"content"`` keys), most recent last. Mutated in place to record
            any tool calls/results made while generating the reply.
        base_url: Base URL of the Ollama server.
        model: Name of the Ollama model to use.
        tools: Tool definitions to offer the model. Defaults to
            :data:`assistant.tools.definitions.TOOL_DEFINITIONS`. Pass
            ``None`` to disable tool calling.
        max_tool_iterations: Maximum number of tool-call round trips before
            forcing a final, tool-free reply.

    Returns:
        The assistant's final reply text, verbatim from the LLM
        (data-fidelity).

    Raises:
        OllamaError: If a request fails, or a response cannot be parsed.
    """
    for _ in range(max_tool_iterations):
        message = _chat(messages, base_url=base_url, model=model, tools=tools)
        tool_calls = message.get("tool_calls") or []

        if not tool_calls:
            return message.get("content") or ""

        messages.append({"role": "assistant", "content": message.get("content") or "", "tool_calls": tool_calls})
        messages.extend(_execute_tool_calls(tool_calls))

    # Exceeded max_tool_iterations: ask once more without tools to force a
    # final natural-language reply instead of looping forever.
    message = _chat(messages, base_url=base_url, model=model, tools=None)
    return message.get("content") or ""


#: Slash commands handled by :func:`_dispatch_slash_command` instead of the LLM.
_SLASH_COMMAND_PREFIXES: tuple[str, ...] = ("/note", "/remember", "/compress")


def _load_context(store: SessionStore, user_id: str) -> dict[str, Any]:
    """Load *user_id*'s session context, or a fresh empty one if none exists."""
    context = store.get_session(user_id) or {
        "working_memory": [],
        "session_memory": {},
        "long_term_memory": [],
    }
    context = dict(context)
    context.pop("_meta", None)
    return context


def _dispatch_slash_command(
    user_id: str,
    message_text: str,
    *,
    session_store: SessionStore,
    note_store: NoteStore | None,
    long_term_memory_store: LongTermMemoryStore | None,
) -> str | None:
    """Route *message_text* to its slash-command handler (AC14-16).

    Returns the handler's response text, or *None* if *message_text* is not
    one of ``/note``, ``/remember``, or ``/compress``. A ``ValueError`` raised
    by a handler (e.g. a missing argument) is caught and returned as the
    response, so the user sees a clear message instead of an exception.
    """
    command = message_text.split(maxsplit=1)[0]
    if command not in _SLASH_COMMAND_PREFIXES:
        return None

    try:
        if command == "/note":
            return handle_note_command(user_id, message_text, note_store or NoteStore())
        if command == "/remember":
            return handle_remember_command(
                user_id, message_text, long_term_memory_store or LongTermMemoryStore()
            )
        return handle_compress_command(user_id, message_text, session_store)
    except ValueError as exc:
        return str(exc)


def handle_user_message(
    source_interface: str,
    message_text: str,
    *,
    store: SessionStore | None = None,
    responder: Responder = default_responder,
    note_store: NoteStore | None = None,
    long_term_memory_store: LongTermMemoryStore | None = None,
) -> str:
    """Process an incoming message from any interface and return the reply.

    Args:
        source_interface: The originating interface. Must be one of
            ``"web_ui"``, ``"telegram"``, or ``"discord"`` (see
            :data:`assistant.session_resolver.VALID_INTERFACES`).
        message_text: The user's message text.
        store: Optional :class:`~assistant.session_store.SessionStore`.
            If *None*, a default store is created. Pass an explicit *store*
            in tests to use an isolated database.
        responder: Callable that generates the assistant's reply from the
            current ``working_memory``. Defaults to
            :func:`default_responder`. Not called for slash commands.
        note_store: Optional :class:`~assistant.notes.NoteStore` used to
            handle ``/note`` commands (AC14). If *None*, a default store is
            created.
        long_term_memory_store: Optional
            :class:`~assistant.long_term_memory.LongTermMemoryStore` used to
            handle ``/remember`` commands (AC15). If *None*, a default store
            is created.

    Returns:
        The assistant's reply text.

    Raises:
        ValueError: If *message_text* is empty/whitespace, or
            *source_interface* is not recognised.
    """
    if not isinstance(message_text, str) or not message_text.strip():
        raise ValueError("message_text must be a non-empty string")

    if store is None:
        store = SessionStore()

    resolved = session_id_resolver(source_interface, store=store)

    # Slash commands (/note, /remember, /compress) bypass the LLM entirely
    # (AC14-16). /compress mutates and persists the session itself, so it
    # runs before the context below is loaded.
    command_reply = _dispatch_slash_command(
        resolved.user_id,
        message_text,
        session_store=store,
        note_store=note_store,
        long_term_memory_store=long_term_memory_store,
    )

    context = _load_context(store, resolved.user_id)

    working_memory = list(context.get("working_memory", []))
    working_memory.append(
        {"role": "user", "content": message_text, "source_interface": source_interface}
    )

    if command_reply is not None:
        reply = command_reply
    else:
        # Persist the user message before the LLM call so it is not lost if the call fails.
        context["working_memory"] = working_memory
        context["source_interface"] = source_interface
        store.upsert_session(resolved.user_id, context)
        reply = responder(working_memory)

    working_memory.append(
        {"role": "assistant", "content": reply, "source_interface": source_interface}
    )

    context["working_memory"] = working_memory
    context["source_interface"] = source_interface
    try:
        context = maybe_auto_compress(context)
    except OllamaError:
        # Auto-compression is best-effort housekeeping: if the LLM call
        # fails (e.g. times out), don't drop the reply that was already
        # computed -- persist the uncompressed context and try again on
        # a later turn.
        logger.warning(
            "handle_user_message: maybe_auto_compress failed, skipping compression for this turn",
            exc_info=True,
        )

    store.upsert_session(resolved.user_id, context)

    logger.info(
        "handle_user_message: source_interface=%r user_id=%r working_memory=%d message(s)",
        source_interface,
        resolved.user_id,
        len(context["working_memory"]),
    )
    return reply
