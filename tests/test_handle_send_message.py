"""DOM-based test for the chat composer "send" pipeline (Sub-AC 1, governed
dispatch).

Sub-AC 1
--------
``assistant/static/index.html`` must define a function that, given the chat
input field, (a) captures the user's typed text, (b) appends it to the chat
log as a user message bubble, and (c) clears the input field.

This behavior is implemented by the top-level ``handleSendMessage(inputElement,
pipelineFn)`` function composed with ``appendChatMessage(role, htmlContent)``
and ``escapeHtml(value)`` -- the same composition the production submit
handler uses (``handleSendMessage(chatInput, sendMessage)``, where
``sendMessage`` begins by calling ``appendMessage('user', text)`` ==
``appendChatMessage('user', escapeHtml(text))``):

- ``handleSendMessage`` reads ``inputElement.value``, trims it, clears
  ``inputElement.value`` to ``''``, and invokes the given pipeline function
  with the captured (trimmed) text.
- The pipeline function appends a new ``.message.message-user`` bubble
  (containing the HTML-escaped captured text) to the chat message list via
  ``appendChatMessage``.

This is verified end-to-end by extracting ``handleSendMessage``,
``appendChatMessage``, and ``escapeHtml`` from ``index.html`` and executing
them under Node.js against a minimal DOM mock (mirroring
``test_append_chat_message.py`` / ``test_render_history.py``: a
``#chat-messages`` container with an ``#empty-state`` placeholder and a
``#message-list`` child, plus an ``escapeHtml``-compatible ``textContent`` ->
``innerHTML`` link), asserting both the resulting new message element
(``#message-list`` contents) and the emptied ``#chat-input`` value.

All three functions are defined at the top level of the app's second
``<script>`` block (outside the boot IIFE), specifically so they can be
evaluated standalone like this.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

INDEX_HTML_PATH = (
    Path(__file__).resolve().parent.parent / "assistant" / "static" / "index.html"
)

NODE_BIN = shutil.which("node")

pytestmark = pytest.mark.skipif(
    NODE_BIN is None,
    reason="Node.js is required to execute handleSendMessage() for this test",
)


def _extract_function_source(js_text: str, func_name: str) -> str:
    """Extract a top-level ``function <func_name>(...) { ... }`` declaration.

    Uses brace-counting (rather than a single regex) so the extraction is
    robust to nested braces/braces-in-strings within the function body.
    """
    header_re = re.compile(r"function\s+" + re.escape(func_name) + r"\s*\([^)]*\)\s*\{")
    match = header_re.search(js_text)
    assert match, f"could not find `function {func_name}(...)` in source"

    brace_start = match.end() - 1  # index of the opening '{'
    depth = 0
    for i in range(brace_start, len(js_text)):
        ch = js_text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return js_text[match.start() : i + 1]

    raise AssertionError(f"unbalanced braces while extracting `{func_name}`")


@pytest.fixture(scope="module")
def html_text() -> str:
    return INDEX_HTML_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def app_script(html_text: str) -> str:
    # The app logic lives in the second <script> block (the first is the
    # inlined marked.js library); pick whichever block defines the function.
    script_blocks = re.findall(r"<script>(.*?)</script>", html_text, re.DOTALL)
    candidates = [block for block in script_blocks if "function handleSendMessage" in block]
    assert candidates, "expected a <script> block defining `handleSendMessage`"
    return candidates[0]


@pytest.fixture(scope="module")
def send_pipeline_source(app_script: str) -> str:
    """Concatenate ``handleSendMessage`` and the functions its production
    pipeline (``sendMessage`` -> ``appendMessage('user', text)``) relies on
    to append the user's message bubble: ``appendChatMessage`` and
    ``escapeHtml``.

    Function declarations are hoisted, so declaration order doesn't matter
    for execution.
    """
    parts = [
        _extract_function_source(app_script, "escapeHtml"),
        _extract_function_source(app_script, "appendChatMessage"),
        _extract_function_source(app_script, "handleSendMessage"),
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Minimal DOM mock
# ---------------------------------------------------------------------------
#
# Mirrors `test_render_history.py`'s DOM mock: `document.getElementById` /
# `document.createElement` backed by a small tree of mock elements with
# working `appendChild`/`remove`, plus an `escapeHtml`-compatible
# `textContent` -> `innerHTML` link on every created element. Seeds
# `#chat-messages` containing `#empty-state` and `#message-list` siblings,
# mirroring index.html's markup before the first message is sent.
_DOM_MOCK_PRELUDE = """
function __escapeForInnerHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function makeElement(tagName) {
  var _textContent = '';
  var _innerHTML = '';
  var el = {
    tagName: String(tagName || 'div').toUpperCase(),
    className: '',
    children: [],
    parentNode: null,
    scrollTop: 0,
    scrollHeight: 100,
    appendChild: function (child) {
      child.parentNode = this;
      this.children.push(child);
      return child;
    },
    remove: function () {
      if (this.parentNode) {
        var idx = this.parentNode.children.indexOf(this);
        if (idx !== -1) {
          this.parentNode.children.splice(idx, 1);
        }
        this.parentNode = null;
      }
    }
  };
  Object.defineProperty(el, 'textContent', {
    get: function () { return _textContent; },
    set: function (value) {
      _textContent = value;
      _innerHTML = __escapeForInnerHtml(value);
    }
  });
  Object.defineProperty(el, 'innerHTML', {
    get: function () { return _innerHTML; },
    set: function (value) { _innerHTML = value; }
  });
  return el;
}

var chatMessages = makeElement('div');
var emptyState = makeElement('div');
var messageList = makeElement('div');
chatMessages.appendChild(emptyState);
chatMessages.appendChild(messageList);

var __elementsById = {
  'chat-messages': chatMessages,
  'empty-state': emptyState,
  'message-list': messageList
};

global.document = {
  getElementById: function (id) {
    return __elementsById[id] || null;
  },
  createElement: function (tag) {
    return makeElement(tag);
  }
};
global.window = global;

// The real chat composer's text input -- a plain object exposing `.value`,
// matching `#chat-input`'s relevant surface for `handleSendMessage`.
var chatInput = { value: '' };

// Mirrors the production pipeline: `sendMessage(text)` begins with
// `appendMessage('user', text)`, which is `appendChatMessage('user',
// escapeHtml(text))`.
function userMessagePipeline(text) {
  appendChatMessage('user', escapeHtml(text));
}
"""


def _run_node_harness(send_pipeline_source: str, harness_body: str) -> dict:
    """Run the DOM mock + escapeHtml + appendChatMessage + handleSendMessage
    + harness_body under Node.js.

    `harness_body` should `console.log(JSON.stringify(...))` its result.
    Returns the parsed JSON object printed by the script.
    """
    script = "\n".join([_DOM_MOCK_PRELUDE, send_pipeline_source, harness_body])

    proc = subprocess.run(
        [NODE_BIN, "-e", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"node harness failed: {proc.stderr}"

    lines = [line for line in proc.stdout.strip().splitlines() if line.strip()]
    assert lines, f"expected JSON output on stdout, got: {proc.stdout!r} / {proc.stderr!r}"
    return json.loads(lines[-1])


class TestHandleSendMessageAppendsUserBubble:
    """Captured text is appended to the chat log as a `.message-user` bubble."""

    def test_appends_new_message_element_with_input_text(self, send_pipeline_source):
        harness = """
chatInput.value = 'Hello there';
handleSendMessage(chatInput, userMessagePipeline);
console.log(JSON.stringify({
  childCount: messageList.children.length,
  wrapperClassName: messageList.children[0].className,
  bubbleClassName: messageList.children[0].children[0].className,
  bubbleHtml: messageList.children[0].children[0].innerHTML
}));
"""
        output = _run_node_harness(send_pipeline_source, harness)

        assert output["childCount"] == 1
        assert output["wrapperClassName"] == "message message-user"
        assert output["bubbleClassName"] == "message-bubble"
        assert output["bubbleHtml"] == "Hello there"

    def test_removes_empty_state_placeholder_on_first_message(self, send_pipeline_source):
        harness = """
chatInput.value = 'First message';
handleSendMessage(chatInput, userMessagePipeline);
console.log(JSON.stringify({
  emptyStateRemoved: chatMessages.children.indexOf(emptyState) === -1,
  messageListInChatMessages: chatMessages.children.indexOf(messageList) !== -1
}));
"""
        output = _run_node_harness(send_pipeline_source, harness)

        assert output["emptyStateRemoved"] is True
        assert output["messageListInChatMessages"] is True

    def test_appends_multiple_messages_in_order(self, send_pipeline_source):
        harness = """
chatInput.value = 'first';
handleSendMessage(chatInput, userMessagePipeline);
chatInput.value = 'second';
handleSendMessage(chatInput, userMessagePipeline);
console.log(JSON.stringify({
  childCount: messageList.children.length,
  contents: messageList.children.map(function (el) { return el.children[0].innerHTML; })
}));
"""
        output = _run_node_harness(send_pipeline_source, harness)

        assert output["childCount"] == 2
        assert output["contents"] == ["first", "second"]

    def test_escapes_html_special_characters_in_appended_bubble(self, send_pipeline_source):
        harness = """
chatInput.value = '<b>bold</b> & "quotes"';
handleSendMessage(chatInput, userMessagePipeline);
console.log(JSON.stringify({
  bubbleHtml: messageList.children[0].children[0].innerHTML
}));
"""
        output = _run_node_harness(send_pipeline_source, harness)

        assert "<b>" not in output["bubbleHtml"]
        assert output["bubbleHtml"] == "&lt;b&gt;bold&lt;/b&gt; &amp; &quot;quotes&quot;"


class TestHandleSendMessageClearsInput:
    """The input field is emptied after the message is captured."""

    def test_clears_input_value_after_send(self, send_pipeline_source):
        harness = """
chatInput.value = 'Hello there';
handleSendMessage(chatInput, userMessagePipeline);
console.log(JSON.stringify({ inputValue: chatInput.value }));
"""
        output = _run_node_harness(send_pipeline_source, harness)

        assert output["inputValue"] == ""

    def test_trims_whitespace_from_captured_text_but_clears_raw_input(
        self, send_pipeline_source
    ):
        harness = """
chatInput.value = '  spaced out message  \\n';
var captured = handleSendMessage(chatInput, userMessagePipeline);
console.log(JSON.stringify({
  inputValue: chatInput.value,
  captured: captured,
  bubbleHtml: messageList.children[0].children[0].innerHTML
}));
"""
        output = _run_node_harness(send_pipeline_source, harness)

        assert output["inputValue"] == ""
        assert output["captured"] == "spaced out message"
        assert output["bubbleHtml"] == "spaced out message"


class TestHandleSendMessageReturnValueAndPipeline:
    """The function returns the captured text and invokes the pipeline once."""

    def test_returns_captured_trimmed_text(self, send_pipeline_source):
        harness = """
chatInput.value = '  hi  ';
var result = handleSendMessage(chatInput, userMessagePipeline);
console.log(JSON.stringify({ result: result }));
"""
        output = _run_node_harness(send_pipeline_source, harness)

        assert output["result"] == "hi"

    def test_invokes_pipeline_exactly_once_with_captured_text(self, send_pipeline_source):
        harness = """
var calls = [];
chatInput.value = 'ping';
handleSendMessage(chatInput, function (text) { calls.push(text); });
console.log(JSON.stringify({ calls: calls }));
"""
        output = _run_node_harness(send_pipeline_source, harness)

        assert output["calls"] == ["ping"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
