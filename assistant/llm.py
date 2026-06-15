"""LLM integration helpers for the Personal Assistant Agent.

This module provides pure-function utilities for interacting with the local
Ollama instance (gemma4:12b-mlx) and parsing its JSON response payloads.

All functions here are pure (no side effects, same output for same input)
unless explicitly documented otherwise, enabling straightforward unit testing
without mocking external services.

Data-fidelity principle
-----------------------
Factual content returned by the LLM (summaries, answers) must be passed
through verbatim to callers.  These helpers extract text; they never alter it.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "gemma4:12b-mlx")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class OllamaParseError(ValueError):
    """Raised when an Ollama response payload cannot be parsed.

    Inherits from :class:`ValueError` so callers that already catch
    ``ValueError`` get coverage without a new import.
    """


class OllamaError(RuntimeError):
    """Raised when a request to the local Ollama server fails after retries."""


# ---------------------------------------------------------------------------
# parse_ollama_summary
# ---------------------------------------------------------------------------


def parse_ollama_summary(response: dict) -> str:  # noqa: C901 — complexity is intentional
    """Extract the summary string from an Ollama JSON response payload.

    This is a **pure function** — it has no side effects and always produces
    the same output for the same input.  It is intentionally decoupled from
    any HTTP layer so it can be tested in complete isolation.

    Supported Ollama API response shapes
    -------------------------------------
    The function handles both common Ollama endpoint response formats:

    1. ``/api/generate`` — synchronous generation::

           {
               "model": "gemma4:12b-mlx",
               "created_at": "2026-06-09T00:00:00Z",
               "response": "The summary text produced by the LLM.",
               "done": true,
               ...
           }

       The summary is extracted from the top-level ``"response"`` key.

    2. ``/api/chat`` — chat completion::

           {
               "model": "gemma4:12b-mlx",
               "created_at": "2026-06-09T00:00:00Z",
               "message": {
                   "role": "assistant",
                   "content": "The summary text produced by the LLM."
               },
               "done": true,
               ...
           }

       The summary is extracted from ``response["message"]["content"]``.

    Field-traversal priority
    ------------------------
    If the response contains **both** a top-level ``"response"`` key and a
    ``"message"`` object, the ``"response"`` key takes priority (it is the
    primary ``/api/generate`` shape used by the summarisation pipeline).

    Empty-content handling
    ----------------------
    If the target field is present but its value is ``None`` or an empty
    string, the function returns an empty string ``""`` rather than raising.
    This covers valid (but degenerate) LLM outputs such as an empty reply.

    Args:
        response: A ``dict`` representing the parsed JSON body returned by
                  the Ollama API.  Must be a ``dict``; other types raise
                  immediately.

    Returns:
        The extracted summary string.  May be ``""`` when the content field
        is present but empty or ``None``.

    Raises:
        OllamaParseError: When *response* is not a ``dict``, or when neither
                          the ``"response"`` key nor the ``"message.content"``
                          path is present in the payload.  The exception
                          message describes the specific problem so callers
                          can surface a meaningful error to the user.

    Examples:
        >>> parse_ollama_summary({"response": "Paris is the capital of France."})
        'Paris is the capital of France.'

        >>> parse_ollama_summary({
        ...     "message": {"role": "assistant", "content": "42 is the answer."}
        ... })
        '42 is the answer.'

        >>> parse_ollama_summary({"response": ""})
        ''

        >>> parse_ollama_summary({})
        Traceback (most recent call last):
            ...
        assistant.llm.OllamaParseError: ...
    """
    # ------------------------------------------------------------------
    # 1. Type guard — must be a dict
    # ------------------------------------------------------------------
    if not isinstance(response, dict):
        raise OllamaParseError(
            f"Ollama response must be a dict, got {type(response).__name__!r}. "
            f"Received: {response!r}"
        )

    # ------------------------------------------------------------------
    # 2. Primary path: /api/generate — top-level "response" key
    # ------------------------------------------------------------------
    if "response" in response:
        value = response["response"]
        if value is None:
            logger.debug(
                "parse_ollama_summary: 'response' key is None — returning empty string"
            )
            return ""
        summary = str(value)
        logger.debug(
            "parse_ollama_summary: extracted %d chars from 'response' key",
            len(summary),
        )
        return summary

    # ------------------------------------------------------------------
    # 3. Secondary path: /api/chat — "message.content" nested path
    # ------------------------------------------------------------------
    if "message" in response:
        message = response["message"]

        if message is None:
            raise OllamaParseError(
                "Ollama response has 'message' key but its value is None. "
                "Expected a dict with a 'content' field."
            )

        if not isinstance(message, dict):
            raise OllamaParseError(
                f"Ollama response 'message' must be a dict, "
                f"got {type(message).__name__!r}."
            )

        if "content" not in message:
            raise OllamaParseError(
                "Ollama response 'message' dict is missing the 'content' key. "
                f"Available keys: {list(message.keys())!r}"
            )

        value = message["content"]
        if value is None:
            logger.debug(
                "parse_ollama_summary: 'message.content' is None — returning empty string"
            )
            return ""

        summary = str(value)
        logger.debug(
            "parse_ollama_summary: extracted %d chars from 'message.content' path",
            len(summary),
        )
        return summary

    # ------------------------------------------------------------------
    # 4. Neither expected path found — raise with diagnostic detail
    # ------------------------------------------------------------------
    available_keys = list(response.keys())
    raise OllamaParseError(
        "Ollama response does not contain a recognised content field. "
        "Expected either a top-level 'response' key (for /api/generate) "
        "or a 'message.content' path (for /api/chat). "
        f"Available top-level keys: {available_keys!r}"
    )
