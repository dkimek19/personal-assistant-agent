"""Behavioral / DOM tests for the ``renderWeatherUnavailable()`` JS function.

Sub-AC 2c (governed dispatch)
------------------------------
``assistant/static/index.html`` must define a top-level
``renderWeatherUnavailable()`` function that renders a graceful inline
"unavailable" status message into the sidebar weather widget DOM
(``#weather-body``) -- the ontology's ``widget_state`` "error"/"empty" case
for the ``weather_data`` concept -- for use by ``renderWeatherWidget(data)``
when given null/undefined/error/empty input.

This is verified end-to-end by extracting ``renderWeatherUnavailable`` --
along with the contiguous top-level helper declarations that precede it
(``escapeHtml``, ``WEATHER_ICONS``, ``weatherIcon``, ``renderWeather``,
``renderWeatherSuccess``) -- from ``index.html`` and executing it under
Node.js against a minimal ``document`` shim whose ``getElementById`` returns
a *persistent* mock element per id (so the ``#weather-body`` mutation made by
``renderWeatherUnavailable`` is observable afterwards) and whose
``createElement`` backs ``escapeHtml``'s ``textContent`` -> ``innerHTML``
escaping. The test then invokes ``renderWeatherUnavailable()`` directly (with
no arguments) and asserts the fallback "unavailable" message appears both in
the function's return value and in the resulting ``#weather-body`` innerHTML,
mirroring ``test_render_weather_success.py``'s treatment of
``renderWeatherSuccess``.
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
    reason="Node.js is required to execute renderWeatherUnavailable() for this test",
)


def _find_function_span(js_text: str, func_name: str) -> tuple[int, int]:
    """Locate a top-level ``function <func_name>(...) { ... }`` declaration.

    Uses brace-counting (rather than a single regex) so the extraction is
    robust to nested braces/braces-in-strings within the function body.
    Returns the ``(start, end)`` character offsets of the declaration
    (``end`` is exclusive, i.e. one past the closing ``}``).
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
                return match.start(), i + 1

    raise AssertionError(f"unbalanced braces while extracting `{func_name}`")


def _extract_dependency_chain_source(js_text: str, first_func: str, last_func: str) -> str:
    """Extract source spanning from the start of ``function <first_func>``
    through the end of ``function <last_func>``'s body.

    Used to extract a contiguous group of top-level declarations
    (``escapeHtml``, ``WEATHER_ICONS``, ``weatherIcon``, ``renderWeather``,
    ``renderWeatherSuccess``, ``renderWeatherUnavailable``) that are declared
    one after another at the top level of the app's ``<script>`` block.
    """
    start, _ = _find_function_span(js_text, first_func)
    _, end = _find_function_span(js_text, last_func)
    return js_text[start:end]


@pytest.fixture(scope="module")
def render_weather_unavailable_source() -> str:
    html_text = INDEX_HTML_PATH.read_text(encoding="utf-8")

    # The app logic lives in the second <script> block (the first is the
    # inlined marked.js library); pick whichever block defines the function.
    script_blocks = re.findall(r"<script>(.*?)</script>", html_text, re.DOTALL)
    candidates = [
        block for block in script_blocks if "function renderWeatherUnavailable" in block
    ]
    assert candidates, "expected a <script> block defining `renderWeatherUnavailable`"

    return _extract_dependency_chain_source(
        candidates[0], "escapeHtml", "renderWeatherUnavailable"
    )


# ---------------------------------------------------------------------------
# Mocks for document
# ---------------------------------------------------------------------------
#
# `document.getElementById` returns a *persistent* element per id (recorded
# in `global.__elements`), so the test harness can inspect the same element
# object that `renderWeatherUnavailable()` mutated (e.g.
# `bodyEl.innerHTML = ...`). The initial `#weather-body` innerHTML is seeded
# with the "Loading..." placeholder so the test can confirm it gets replaced.
_MOCK_PRELUDE = """
function __escapeForInnerHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function __makeMockElement(id) {
  var _textContent = '';
  var el = {
    id: id,
    innerHTML: '<p class="widget-status">Loading&hellip;</p>'
  };
  Object.defineProperty(el, 'textContent', {
    get: function () { return _textContent; },
    set: function (value) {
      _textContent = value;
      el.innerHTML = __escapeForInnerHtml(value === null || value === undefined ? '' : value);
    }
  });
  return el;
}

global.__elements = {};
global.document = {
  getElementById: function (id) {
    if (!global.__elements[id]) {
      global.__elements[id] = __makeMockElement(id);
    }
    return global.__elements[id];
  },
  createElement: function () { return __makeMockElement(); }
};
global.window = global;
"""


def _run_node_harness(render_weather_unavailable_source: str, harness_body: str) -> dict:
    """Run the mocked prelude + renderWeatherUnavailable (+ deps) + harness body.

    `harness_body` should `console.log(JSON.stringify(...))` its result,
    which is parsed and returned as a dict.
    """
    script = "\n".join([_MOCK_PRELUDE, render_weather_unavailable_source, harness_body])

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


class TestRenderWeatherUnavailable:
    """`renderWeatherUnavailable()` renders an inline "unavailable" status
    message into `#weather-body`, for use by `renderWeatherWidget(data)` in
    error/empty `weather_data` cases."""

    def _render(self, render_weather_unavailable_source):
        harness = (
            "var html = renderWeatherUnavailable();\n"
            "console.log(JSON.stringify({\n"
            "  returned: html,\n"
            "  bodyInnerHTML: global.__elements['weather-body'].innerHTML\n"
            "}));\n"
        )
        return _run_node_harness(render_weather_unavailable_source, harness)

    def test_returns_html_with_unavailable_message(self, render_weather_unavailable_source):
        output = self._render(render_weather_unavailable_source)
        html = output["returned"]

        assert "unavailable" in html.lower(), html
        assert "widget-status" in html

    def test_writes_unavailable_message_into_weather_body(
        self, render_weather_unavailable_source
    ):
        output = self._render(render_weather_unavailable_source)

        assert output["bodyInnerHTML"] == output["returned"]
        assert "unavailable" in output["bodyInnerHTML"].lower()

    def test_replaces_loading_placeholder(self, render_weather_unavailable_source):
        output = self._render(render_weather_unavailable_source)

        assert "Loading" not in output["bodyInnerHTML"]

    def test_does_not_leak_live_weather_markup(self, render_weather_unavailable_source):
        output = self._render(render_weather_unavailable_source)
        html = output["returned"]

        assert "weather-main" not in html
        assert "weather-temp" not in html

    def test_is_callable_with_no_arguments_and_idempotent(
        self, render_weather_unavailable_source
    ):
        # Call twice with no arguments to confirm the function does not
        # require any input and consistently re-renders the same fallback.
        harness = (
            "var first = renderWeatherUnavailable();\n"
            "var second = renderWeatherUnavailable();\n"
            "console.log(JSON.stringify({\n"
            "  first: first,\n"
            "  second: second,\n"
            "  bodyInnerHTML: global.__elements['weather-body'].innerHTML\n"
            "}));\n"
        )
        output = _run_node_harness(render_weather_unavailable_source, harness)

        assert output["first"] == output["second"]
        assert output["bodyInnerHTML"] == output["first"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
