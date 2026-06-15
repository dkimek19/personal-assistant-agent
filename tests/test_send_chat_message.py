"""Behavioral test for the ``sendChatMessage(message)`` JS function.

Sub-AC 1 (governed dispatch)
-----------------------------
``assistant/static/index.html`` must define a top-level ``sendChatMessage(message)``
function that:

- POSTs ``{"message": message}`` (JSON) to ``/chat``, and
- resolves with the parsed JSON response body (the ontology's
  ``chat_response``, e.g. ``{"reply": "..."}``).

This is verified end-to-end by extracting the function's source from
``index.html`` and executing it under Node.js with a mocked ``global.fetch``,
asserting both the recorded request payload (URL, method, headers, body) and
the value the function resolves with.

The function is defined outside the app's DOM-setup IIFE specifically so it
has no ``document``/``window`` dependencies and can be evaluated standalone
like this.
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
    NODE_BIN is None, reason="Node.js is required to execute sendChatMessage() for this test"
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
def send_chat_message_source() -> str:
    html_text = INDEX_HTML_PATH.read_text(encoding="utf-8")

    # The app logic lives in the second <script> block (the first is the
    # inlined marked.js library); pick whichever block defines the function.
    script_blocks = re.findall(r"<script>(.*?)</script>", html_text, re.DOTALL)
    candidates = [block for block in script_blocks if "function sendChatMessage" in block]
    assert candidates, "expected a <script> block defining `sendChatMessage`"

    return _extract_function_source(candidates[0], "sendChatMessage")


def _run_node_harness(send_chat_message_source: str, harness_body: str) -> dict:
    """Run a small Node.js script: mocked fetch + sendChatMessage + harness_body.

    `harness_body` should `console.log(JSON.stringify(...))` its result.
    Returns the parsed JSON object printed by the script.
    """
    script = "\n".join(
        [
            "global.__calls = [];",
            "global.fetch = function (url, options) {",
            "  __calls.push({ url: url, options: options });",
            "  return global.__mockResponse(url, options);",
            "};",
            send_chat_message_source,
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


class TestSendChatMessageRequest:
    """The function POSTs `{message}` as JSON to `/chat`."""

    def test_posts_message_payload_to_chat_endpoint(self, send_chat_message_source):
        harness = """
global.__mockResponse = function () {
  return Promise.resolve({
    ok: true,
    status: 200,
    json: function () { return Promise.resolve({ reply: 'Hi **there**' }); }
  });
};

sendChatMessage('hello world').then(function (data) {
  console.log(JSON.stringify({ calls: __calls, result: data }));
}).catch(function (err) {
  console.log(JSON.stringify({ error: err.message }));
});
"""
        output = _run_node_harness(send_chat_message_source, harness)

        assert "error" not in output, output
        assert len(output["calls"]) == 1

        call = output["calls"][0]
        assert call["url"] == "/chat"

        options = call["options"]
        assert options["method"] == "POST"
        assert options["headers"]["Content-Type"] == "application/json"

        body = json.loads(options["body"])
        assert body == {"message": "hello world"}


class TestSendChatMessageResponse:
    """The function resolves with the parsed JSON response body."""

    def test_resolves_with_parsed_json_reply(self, send_chat_message_source):
        harness = """
global.__mockResponse = function () {
  return Promise.resolve({
    ok: true,
    status: 200,
    json: function () { return Promise.resolve({ reply: '# Markdown reply\\n\\nSome *text*.' }); }
  });
};

sendChatMessage('what is up?').then(function (data) {
  console.log(JSON.stringify({ result: data }));
}).catch(function (err) {
  console.log(JSON.stringify({ error: err.message }));
});
"""
        output = _run_node_harness(send_chat_message_source, harness)

        assert "error" not in output, output
        assert output["result"] == {"reply": "# Markdown reply\n\nSome *text*."}

    def test_rejects_with_server_detail_on_non_ok_response(self, send_chat_message_source):
        harness = """
global.__mockResponse = function () {
  return Promise.resolve({
    ok: false,
    status: 503,
    json: function () { return Promise.resolve({ detail: 'LLM backend unreachable' }); }
  });
};

sendChatMessage('hello').then(function (data) {
  console.log(JSON.stringify({ result: data }));
}).catch(function (err) {
  console.log(JSON.stringify({ error: err.message }));
});
"""
        output = _run_node_harness(send_chat_message_source, harness)

        assert output.get("error") == "LLM backend unreachable"
