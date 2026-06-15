"""Behavioral / DOM tests for the ``renderHistory(messages)`` JS function.

Sub-AC 3 (governed dispatch)
-----------------------------
``assistant/static/index.html`` must define a top-level
``renderHistory(messages)`` function that appends each historical
``chat_message`` object (``{role, content}`` -- the ontology's
``chat_history``, e.g. the array resolved by ``fetchHistory()`` from
``GET /api/history``) to the chat messages container (``#chat-messages``), in
order, with markdown rendering applied:

- assistant messages are rendered as markdown (sanitized HTML) via
  ``renderMarkdown()``, and
- user messages are HTML-escaped via ``escapeHtml()``,

then appended as new ``.message-bubble`` children via
``appendChatMessage(role, html)``, mirroring the live single-message
rendering pipeline (``appendMessage``) used for new chat turns.

This is verified end-to-end by extracting ``renderHistory`` and its
dependencies (``appendChatMessage``, ``renderMarkdown``, ``escapeHtml``) --
plus the inlined ``marked.js`` library -- from ``index.html`` and executing
them under Node.js against a minimal DOM mock (mirroring
``test_append_chat_message.py``'s ``appendChatMessage`` DOM mock, extended
with an ``escapeHtml``-compatible ``textContent`` -> ``innerHTML`` link,
mirroring ``test_render_weather_widget.py``), asserting the resulting
``#chat-messages`` contents for a sample ``messages`` array.

``renderHistory`` is defined at the top level of the second ``<script>``
block (outside the app's DOM-setup IIFE), alongside ``appendChatMessage``, so
it has no dependencies beyond ``document`` and the inlined ``marked`` library
and can be evaluated standalone like this.
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
    NODE_BIN is None, reason="Node.js is required to execute renderHistory() for this test"
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
def script_blocks(html_text: str) -> list[str]:
    return re.findall(r"<script>(.*?)</script>", html_text, re.DOTALL)


@pytest.fixture(scope="module")
def marked_source(script_blocks: list[str]) -> str:
    # The first <script> block is the inlined, offline copy of marked.js
    # (the project's "no CDN" markdown library requirement).
    candidates = [block for block in script_blocks if "a markdown parser" in block]
    assert candidates, "expected a <script> block containing the inlined marked.js library"
    return candidates[0]


@pytest.fixture(scope="module")
def app_script(script_blocks: list[str]) -> str:
    candidates = [block for block in script_blocks if "function renderHistory" in block]
    assert candidates, "expected a <script> block defining `renderHistory`"
    return candidates[0]


@pytest.fixture(scope="module")
def render_history_dependencies_source(app_script: str) -> str:
    """Concatenate ``renderHistory`` and the dependency functions it calls.

    ``renderHistory`` calls ``renderMarkdown`` (assistant messages),
    ``escapeHtml`` (user messages), and ``appendChatMessage`` (DOM insertion).
    These declarations are not contiguous in ``index.html`` (other top-level
    helpers -- ``fetchWeather``, the weather/calendar/notes widget renderers,
    etc. -- sit between them), so each is extracted individually via
    brace-counting and concatenated. Function declarations are hoisted, so
    declaration order doesn't matter for execution.
    """
    parts = [
        _extract_function_source(app_script, "renderMarkdown"),
        _extract_function_source(app_script, "escapeHtml"),
        _extract_function_source(app_script, "appendChatMessage"),
        _extract_function_source(app_script, "renderHistory"),
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Minimal DOM mock
# ---------------------------------------------------------------------------
#
# Combines the `appendChatMessage` DOM mock from `test_append_chat_message.py`
# (`document.getElementById` / `document.createElement` backed by a small
# tree of mock elements with working `appendChild`/`remove`) with an
# `escapeHtml`-compatible `textContent` -> `innerHTML` link on every created
# element, mirroring `test_render_weather_widget.py`. Seeds a
# `#chat-messages` container that initially contains a single `#empty-state`
# placeholder child, mirroring index.html's markup before the first message
# is rendered.
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


def _run_node_harness(
    marked_source: str, render_history_dependencies_source: str, harness_body: str
) -> dict:
    """Run the DOM mock + marked.js + renderHistory (+ deps) + harness_body.

    `harness_body` should `console.log(JSON.stringify(...))` its result.
    Returns the parsed JSON object printed by the script.
    """
    script = "\n".join(
        [
            marked_source,
            # The UMD wrapper assigns the marked module to `module.exports`
            # when run under Node/CommonJS (which `node -e` provides). Make
            # it available as `window.marked` (== `global.marked`, since
            # `global.window = global`), mirroring how the browser exposes
            # `window.marked` from the same inlined script.
            "if (typeof module !== 'undefined' && module.exports && module.exports.marked) {"
            " global.marked = module.exports.marked; }",
            _DOM_MOCK_PRELUDE,
            render_history_dependencies_source,
            harness_body,
        ]
    )

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


# A representative `messages` array, matching the shape of the `history`
# field returned by GET /api/history (HistoryResponse: {"history": [{"role",
# "content"}, ...]}) and resolved by `fetchHistory()`.
_SAMPLE_MESSAGES = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there! How can I help?"},
]


class TestRenderHistoryOrderingAndRoles:
    """Each message in `messages` is appended in order with the correct role."""

    def test_appends_each_message_in_order_with_correct_roles(
        self, marked_source, render_history_dependencies_source
    ):
        harness = (
            "var result = renderHistory(" + json.dumps(_SAMPLE_MESSAGES) + ");\n"
            "console.log(JSON.stringify({\n"
            "  childCount: chatMessages.children.length,\n"
            "  classes: chatMessages.children.map(function (el) { return el.className; }),\n"
            "  contents: chatMessages.children.map(function (el) { return el.children[0].innerHTML; }),\n"
            "  returnedCount: result.length\n"
            "}));"
        )
        output = _run_node_harness(marked_source, render_history_dependencies_source, harness)

        assert output["childCount"] == 2
        assert output["classes"] == ["message message-user", "message message-assistant"]
        assert output["contents"] == ["Hello", "<p>Hi there! How can I help?</p>\n"]
        assert output["returnedCount"] == 2

    def test_removes_empty_state_placeholder_on_first_history_message(
        self, marked_source, render_history_dependencies_source
    ):
        harness = (
            "renderHistory(" + json.dumps(_SAMPLE_MESSAGES[:1]) + ");\n"
            "console.log(JSON.stringify({\n"
            "  emptyStateRemoved: chatMessages.children.indexOf(emptyState) === -1,\n"
            "  childCount: chatMessages.children.length\n"
            "}));"
        )
        output = _run_node_harness(marked_source, render_history_dependencies_source, harness)

        assert output["emptyStateRemoved"] is True
        assert output["childCount"] == 1

    def test_returns_wrapper_elements_appended_to_container_in_order(
        self, marked_source, render_history_dependencies_source
    ):
        harness = (
            "var result = renderHistory(" + json.dumps(_SAMPLE_MESSAGES) + ");\n"
            "console.log(JSON.stringify({\n"
            "  matchesContainerOrder: result.every(function (el, i) { return el === chatMessages.children[i]; })\n"
            "}));"
        )
        output = _run_node_harness(marked_source, render_history_dependencies_source, harness)

        assert output["matchesContainerOrder"] is True


class TestRenderHistoryMarkdownRendering:
    """Assistant messages are markdown-rendered; user messages are escaped."""

    def test_renders_assistant_message_content_as_markdown(
        self, marked_source, render_history_dependencies_source
    ):
        messages = [{"role": "assistant", "content": "Here is **bold** text"}]
        harness = (
            "renderHistory(" + json.dumps(messages) + ");\n"
            "console.log(JSON.stringify({\n"
            "  bubbleHtml: chatMessages.children[0].children[0].innerHTML\n"
            "}));"
        )
        output = _run_node_harness(marked_source, render_history_dependencies_source, harness)

        assert output["bubbleHtml"] == "<p>Here is <strong>bold</strong> text</p>\n"

    def test_renders_assistant_markdown_list(
        self, marked_source, render_history_dependencies_source
    ):
        messages = [{"role": "assistant", "content": "- alpha\n- beta"}]
        harness = (
            "renderHistory(" + json.dumps(messages) + ");\n"
            "console.log(JSON.stringify({\n"
            "  bubbleHtml: chatMessages.children[0].children[0].innerHTML\n"
            "}));"
        )
        output = _run_node_harness(marked_source, render_history_dependencies_source, harness)

        assert output["bubbleHtml"] == "<ul>\n<li>alpha</li>\n<li>beta</li>\n</ul>\n"

    def test_escapes_user_message_content_instead_of_rendering_markdown(
        self, marked_source, render_history_dependencies_source
    ):
        # User-authored history content is plain text and must be
        # HTML-escaped (not interpreted as markdown/HTML) when re-rendered.
        messages = [{"role": "user", "content": "<b>not bold</b> & **not markdown**"}]
        harness = (
            "renderHistory(" + json.dumps(messages) + ");\n"
            "console.log(JSON.stringify({\n"
            "  bubbleHtml: chatMessages.children[0].children[0].innerHTML\n"
            "}));"
        )
        output = _run_node_harness(marked_source, render_history_dependencies_source, harness)

        assert output["bubbleHtml"] == "&lt;b&gt;not bold&lt;/b&gt; &amp; **not markdown**"
        assert "<b>" not in output["bubbleHtml"]


class TestRenderHistoryEdgeCases:
    def test_renders_nothing_for_empty_messages_array(
        self, marked_source, render_history_dependencies_source
    ):
        harness = """
var result = renderHistory([]);
console.log(JSON.stringify({
  childCount: chatMessages.children.length,
  emptyStatePresent: chatMessages.children.indexOf(emptyState) !== -1,
  result: result
}));
"""
        output = _run_node_harness(marked_source, render_history_dependencies_source, harness)

        assert output["childCount"] == 1
        assert output["emptyStatePresent"] is True
        assert output["result"] == []

    def test_treats_null_messages_as_empty_array(
        self, marked_source, render_history_dependencies_source
    ):
        harness = """
var result = renderHistory(null);
console.log(JSON.stringify({
  childCount: chatMessages.children.length,
  result: result
}));
"""
        output = _run_node_harness(marked_source, render_history_dependencies_source, harness)

        assert output["childCount"] == 1
        assert output["result"] == []

    def test_treats_missing_content_as_empty_string(
        self, marked_source, render_history_dependencies_source
    ):
        messages = [{"role": "user"}]
        harness = (
            "renderHistory(" + json.dumps(messages) + ");\n"
            "console.log(JSON.stringify({\n"
            "  bubbleHtml: chatMessages.children[0].children[0].innerHTML\n"
            "}));"
        )
        output = _run_node_harness(marked_source, render_history_dependencies_source, harness)

        assert output["bubbleHtml"] == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
