"""Behavioral test for the ``appendChatMessage(role, htmlContent)`` JS function.

Sub-AC 3 (governed dispatch)
-----------------------------
``assistant/static/index.html`` must define a top-level
``appendChatMessage(role, htmlContent)`` function that inserts a new chat
message bubble (the ontology's ``chat_message`` -- ``{role, content}``) into
the chat messages container (``#chat-messages``):

- a wrapper element with class ``message message-<role>`` (``user`` or
  ``assistant``) is appended to ``#chat-messages``, and
- a child ``.message-bubble`` element whose ``innerHTML`` is set to the given
  ``htmlContent`` is appended to that wrapper.

This is verified end-to-end by extracting the function's source from
``index.html`` and executing it under Node.js against a minimal DOM mock
(``document.getElementById`` / ``document.createElement`` backed by a small
tree of mock elements with working ``appendChild``/``remove``), then
asserting the resulting DOM contains the new element with the correct
``message-<role>`` class and bubble content.

``appendChatMessage`` is defined at the top level of the app's second
``<script>`` block (outside the boot IIFE), alongside ``renderMarkdown`` and
``escapeHtml``, specifically so it can be evaluated standalone like this.
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
    NODE_BIN is None, reason="Node.js is required to execute appendChatMessage() for this test"
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
def append_chat_message_source() -> str:
    html_text = INDEX_HTML_PATH.read_text(encoding="utf-8")

    # The app logic lives in the second <script> block (the first is the
    # inlined marked.js library); pick whichever block defines the function.
    script_blocks = re.findall(r"<script>(.*?)</script>", html_text, re.DOTALL)
    candidates = [block for block in script_blocks if "function appendChatMessage" in block]
    assert candidates, "expected a <script> block defining `appendChatMessage`"

    return _extract_function_source(candidates[0], "appendChatMessage")


# ---------------------------------------------------------------------------
# Minimal DOM mock
# ---------------------------------------------------------------------------
#
# `appendChatMessage` only needs `document.getElementById` and
# `document.createElement`, plus working `appendChild`/`remove` so the test
# can walk the resulting tree. Seeds a `#chat-messages` container that
# initially contains a single `#empty-state` placeholder child, mirroring
# index.html's markup before the first message is sent.
_DOM_MOCK_PRELUDE = """
function makeElement(tagName) {
  return {
    tagName: String(tagName || 'div').toUpperCase(),
    className: '',
    innerHTML: '',
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
}

var chatMessages = makeElement('div');
var emptyState = makeElement('div');
chatMessages.appendChild(emptyState);

var __elementsById = {
  'chat-messages': chatMessages,
  'empty-state': emptyState
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
"""


def _run_node_harness(append_chat_message_source: str, harness_body: str) -> dict:
    """Run the DOM mock + appendChatMessage + harness_body under Node.js.

    `harness_body` should `console.log(JSON.stringify(...))` its result.
    Returns the parsed JSON object printed by the script.
    """
    script = "\n".join([_DOM_MOCK_PRELUDE, append_chat_message_source, harness_body])

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


class TestAppendChatMessageUserBubble:
    """A user message bubble is inserted with the `message-user` class."""

    def test_inserts_wrapper_with_message_user_class_and_content(
        self, append_chat_message_source
    ):
        harness = """
var wrapper = appendChatMessage('user', 'Hello there');
console.log(JSON.stringify({
  wrapperClassName: wrapper.className,
  inContainer: chatMessages.children.indexOf(wrapper) !== -1,
  bubbleClassName: wrapper.children[0].className,
  bubbleHtml: wrapper.children[0].innerHTML
}));
"""
        output = _run_node_harness(append_chat_message_source, harness)

        assert output["wrapperClassName"] == "message message-user"
        assert output["inContainer"] is True
        assert output["bubbleClassName"] == "message-bubble"
        assert output["bubbleHtml"] == "Hello there"


class TestAppendChatMessageAssistantBubble:
    """An assistant message bubble is inserted with the `message-assistant` class."""

    def test_inserts_wrapper_with_message_assistant_class_and_html_content(
        self, append_chat_message_source
    ):
        harness = """
var wrapper = appendChatMessage('assistant', '<p><strong>bold</strong> reply</p>');
console.log(JSON.stringify({
  wrapperClassName: wrapper.className,
  inContainer: chatMessages.children.indexOf(wrapper) !== -1,
  bubbleClassName: wrapper.children[0].className,
  bubbleHtml: wrapper.children[0].innerHTML
}));
"""
        output = _run_node_harness(append_chat_message_source, harness)

        assert output["wrapperClassName"] == "message message-assistant"
        assert output["inContainer"] is True
        assert output["bubbleClassName"] == "message-bubble"
        assert output["bubbleHtml"] == "<p><strong>bold</strong> reply</p>"


class TestAppendChatMessageContainerBehavior:
    """Side-effects on `#chat-messages`: empty-state removal, order, scroll."""

    def test_removes_empty_state_placeholder_on_first_message(self, append_chat_message_source):
        harness = """
appendChatMessage('user', 'first message');
console.log(JSON.stringify({
  emptyStateRemoved: chatMessages.children.indexOf(emptyState) === -1,
  childCount: chatMessages.children.length
}));
"""
        output = _run_node_harness(append_chat_message_source, harness)

        assert output["emptyStateRemoved"] is True
        assert output["childCount"] == 1

    def test_appends_messages_in_order_with_correct_roles(self, append_chat_message_source):
        harness = """
var userWrapper = appendChatMessage('user', 'question');
var assistantWrapper = appendChatMessage('assistant', '<p>answer</p>');
console.log(JSON.stringify({
  childCount: chatMessages.children.length,
  classes: chatMessages.children.map(function (el) { return el.className; }),
  contents: chatMessages.children.map(function (el) { return el.children[0].innerHTML; })
}));
"""
        output = _run_node_harness(append_chat_message_source, harness)

        assert output["childCount"] == 2
        assert output["classes"] == ["message message-user", "message message-assistant"]
        assert output["contents"] == ["question", "<p>answer</p>"]

    def test_scrolls_container_to_bottom_after_insert(self, append_chat_message_source):
        harness = """
chatMessages.scrollTop = 0;
chatMessages.scrollHeight = 500;
appendChatMessage('assistant', 'reply');
console.log(JSON.stringify({ scrollTop: chatMessages.scrollTop }));
"""
        output = _run_node_harness(append_chat_message_source, harness)

        assert output["scrollTop"] == 500


class TestAppendChatMessageEdgeCases:
    def test_treats_null_html_content_as_empty_string(self, append_chat_message_source):
        harness = """
var wrapper = appendChatMessage('assistant', null);
console.log(JSON.stringify({ bubbleHtml: wrapper.children[0].innerHTML }));
"""
        output = _run_node_harness(append_chat_message_source, harness)

        assert output["bubbleHtml"] == ""

    def test_treats_undefined_html_content_as_empty_string(self, append_chat_message_source):
        harness = """
var wrapper = appendChatMessage('user', undefined);
console.log(JSON.stringify({ bubbleHtml: wrapper.children[0].innerHTML }));
"""
        output = _run_node_harness(append_chat_message_source, harness)

        assert output["bubbleHtml"] == ""

    def test_returns_the_created_wrapper_element(self, append_chat_message_source):
        harness = """
var wrapper = appendChatMessage('user', 'hi');
console.log(JSON.stringify({
  isReturnedWrapperInContainer: chatMessages.children[chatMessages.children.length - 1] === wrapper
}));
"""
        output = _run_node_harness(append_chat_message_source, harness)

        assert output["isReturnedWrapperInContainer"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
