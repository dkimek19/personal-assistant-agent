"""Behavioral test for the ``fetchHistory()`` JS function (chat history).

Sub-AC 1 (governed dispatch)
-----------------------------
``assistant/static/index.html`` must define a top-level ``fetchHistory()``
function -- the chat panel's history-loading function -- that:

- GETs ``/api/history`` (the new backend pass-through endpoint backing the
  ontology's ``chat_history`` concept), and
- resolves with the parsed JSON response's ``history`` array (an ordered
  list of ``chat_message`` objects, ``{role, content}``).

This is verified end-to-end by extracting the function's source from
``index.html`` and executing it under Node.js with a mocked
``global.fetch``, asserting both the recorded request URL and the value the
function resolves with, mirroring ``tests/test_weather_widget_fetch.py``'s
treatment of ``fetchWeather``.

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
    NODE_BIN is None, reason="Node.js is required to execute fetchHistory() for this test"
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
def fetch_history_source() -> str:
    html_text = INDEX_HTML_PATH.read_text(encoding="utf-8")

    # The app logic lives in the second <script> block (the first is the
    # inlined marked.js library); pick whichever block defines the function.
    script_blocks = re.findall(r"<script>(.*?)</script>", html_text, re.DOTALL)
    candidates = [block for block in script_blocks if "function fetchHistory" in block]
    assert candidates, "expected a <script> block defining `fetchHistory`"

    return _extract_function_source(candidates[0], "fetchHistory")


def _run_node_harness(fetch_history_source: str, harness_body: str) -> dict:
    """Run a small Node.js script: mocked fetch + fetchHistory + harness_body.

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
            fetch_history_source,
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


# A representative `history` payload, matching the shape returned by
# GET /api/history (HistoryResponse: {"history": [{"role", "content"}, ...]}).
_SAMPLE_HISTORY_RESPONSE = {
    "history": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there! How can I help?"},
    ]
}


class TestFetchHistoryRequest:
    """The function calls the backend history endpoint via GET."""

    def test_calls_history_endpoint(self, fetch_history_source):
        harness = """
global.__mockResponse = function () {
  return Promise.resolve({
    ok: true,
    status: 200,
    json: function () { return Promise.resolve(""" + json.dumps(_SAMPLE_HISTORY_RESPONSE) + """); }
  });
};

fetchHistory().then(function (data) {
  console.log(JSON.stringify({ calls: __calls, result: data }));
}).catch(function (err) {
  console.log(JSON.stringify({ error: err.message }));
});
"""
        output = _run_node_harness(fetch_history_source, harness)

        assert "error" not in output, output
        assert len(output["calls"]) == 1

        call = output["calls"][0]
        assert call["url"] == "/api/history"


class TestFetchHistoryResponse:
    """The function resolves with the parsed `history` array on success."""

    def test_resolves_with_history_array_on_success(self, fetch_history_source):
        harness = """
global.__mockResponse = function () {
  return Promise.resolve({
    ok: true,
    status: 200,
    json: function () { return Promise.resolve(""" + json.dumps(_SAMPLE_HISTORY_RESPONSE) + """); }
  });
};

fetchHistory().then(function (data) {
  console.log(JSON.stringify({ result: data }));
}).catch(function (err) {
  console.log(JSON.stringify({ error: err.message }));
});
"""
        output = _run_node_harness(fetch_history_source, harness)

        assert "error" not in output, output
        assert output["result"] == _SAMPLE_HISTORY_RESPONSE["history"]

    def test_resolves_with_empty_array_when_history_missing(self, fetch_history_source):
        harness = """
global.__mockResponse = function () {
  return Promise.resolve({
    ok: true,
    status: 200,
    json: function () { return Promise.resolve({}); }
  });
};

fetchHistory().then(function (data) {
  console.log(JSON.stringify({ result: data }));
}).catch(function (err) {
  console.log(JSON.stringify({ error: err.message }));
});
"""
        output = _run_node_harness(fetch_history_source, harness)

        assert "error" not in output, output
        assert output["result"] == []

    def test_rejects_with_server_detail_on_non_ok_response(self, fetch_history_source):
        harness = """
global.__mockResponse = function () {
  return Promise.resolve({
    ok: false,
    status: 500,
    json: function () { return Promise.resolve({ detail: 'Session store unavailable' }); }
  });
};

fetchHistory().then(function (data) {
  console.log(JSON.stringify({ result: data }));
}).catch(function (err) {
  console.log(JSON.stringify({ error: err.message }));
});
"""
        output = _run_node_harness(fetch_history_source, harness)

        assert output.get("error") == "Session store unavailable"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
