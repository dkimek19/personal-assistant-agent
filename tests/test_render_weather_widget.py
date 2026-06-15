"""Behavioral / DOM tests for the ``renderWeatherWidget(data)`` JS function.

Sub-AC 2 (governed dispatch)
-----------------------------
``assistant/static/index.html`` must define a top-level
``renderWeatherWidget(data)`` function that:

- populates the weather widget body (``#weather-body``) with live values
  (icon, rounded temperature, description, location, rounded wind speed --
  the ontology's ``weather_data`` concept) when given a valid
  ``weather_data`` payload, and
- renders an inline "unavailable" status message into ``#weather-body`` when
  given null/undefined/error/empty input, leaving the widget card's
  ``widget-card`` / ``widget-body`` markup stable (the ontology's
  ``widget_state`` concept).

This is verified end-to-end by extracting ``renderWeatherWidget`` -- along
with the contiguous top-level helper declarations it depends on
(``escapeHtml``, ``WEATHER_ICONS``, ``weatherIcon``, ``renderWeather``) --
from ``index.html`` and executing it under Node.js against a minimal
``document`` shim whose ``getElementById`` returns a *persistent* mock
element per id (so the ``#weather-body`` mutation made by
``renderWeatherWidget`` is observable afterwards) and whose
``createElement`` backs ``escapeHtml``'s ``textContent`` -> ``innerHTML``
escaping. The test then asserts on both the function's return value and the
resulting ``#weather-body`` innerHTML for a sample ``weather_data`` payload
and for error/empty inputs.

``renderWeatherWidget`` and its dependencies are defined at the top level of
the second ``<script>`` block (outside the app's DOM-setup IIFE) specifically
so this contiguous span has no dependencies beyond ``document`` and can be
evaluated standalone like this, mirroring ``test_weather_widget_fetch.py``'s
treatment of ``fetchWeather``.
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
    reason="Node.js is required to execute renderWeatherWidget() for this test",
)


# A representative `weather_data` payload, matching the shape returned by
# GET /api/weather (WeatherResponse / WeatherReport.to_dict()).
_SAMPLE_WEATHER_DATA = {
    "location_name": "Seoul",
    "latitude": 37.5683,
    "longitude": 126.9778,
    "temperature_c": 21.5,
    "wind_speed_kmh": 9.4,
    "weather_code": 1,
    "weather_description": "Mainly clear",
    "observation_time": "2026-06-13T12:00",
}


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
    ``renderWeatherWidget``) that are declared one after another at the top
    level of the app's ``<script>`` block, where later declarations depend on
    earlier ones.
    """
    start, _ = _find_function_span(js_text, first_func)
    _, end = _find_function_span(js_text, last_func)
    return js_text[start:end]


@pytest.fixture(scope="module")
def render_weather_widget_source() -> str:
    html_text = INDEX_HTML_PATH.read_text(encoding="utf-8")

    # The app logic lives in the second <script> block (the first is the
    # inlined marked.js library); pick whichever block defines the function.
    script_blocks = re.findall(r"<script>(.*?)</script>", html_text, re.DOTALL)
    candidates = [block for block in script_blocks if "function renderWeatherWidget" in block]
    assert candidates, "expected a <script> block defining `renderWeatherWidget`"

    return _extract_dependency_chain_source(candidates[0], "escapeHtml", "renderWeatherWidget")


# ---------------------------------------------------------------------------
# Mocks for document
# ---------------------------------------------------------------------------
#
# `document.getElementById` returns a *persistent* element per id (recorded
# in `global.__elements`), so the test harness can inspect the same element
# object that `renderWeatherWidget()` mutated (e.g. `bodyEl.innerHTML = ...`).
#
# `escapeHtml()` (a dependency of `renderWeather`) creates a throwaway <div>
# via `document.createElement`, sets its `textContent`, then reads back
# `innerHTML` to obtain HTML-escaped text; the mock element links those two
# properties via a getter/setter so `escapeHtml()` behaves the same as it
# would in a browser.
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
    innerHTML: ''
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


def _run_node_harness(render_weather_widget_source: str, harness_body: str) -> dict:
    """Run the mocked prelude + renderWeatherWidget (+ deps) + harness body.

    `harness_body` should `console.log(JSON.stringify(...))` its result,
    which is parsed and returned as a dict.
    """
    script = "\n".join([_MOCK_PRELUDE, render_weather_widget_source, harness_body])

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


class TestRenderWeatherWidgetWithValidData:
    """`renderWeatherWidget(data)` populates `#weather-body` with live
    values when given a valid `weather_data` payload."""

    def _render(self, render_weather_widget_source):
        harness = (
            "var html = renderWeatherWidget(" + json.dumps(_SAMPLE_WEATHER_DATA) + ");\n"
            "console.log(JSON.stringify({\n"
            "  returned: html,\n"
            "  bodyInnerHTML: global.__elements['weather-body'].innerHTML\n"
            "}));\n"
        )
        return _run_node_harness(render_weather_widget_source, harness)

    def test_returns_html_with_temperature_description_and_location(
        self, render_weather_widget_source
    ):
        output = self._render(render_weather_widget_source)
        html = output["returned"]

        assert "22°C" in html, html
        assert "Mainly clear" in html
        assert "Seoul" in html
        assert "Wind 9 km/h" in html

    def test_returns_html_with_weather_icon_for_weather_code(self, render_weather_widget_source):
        output = self._render(render_weather_widget_source)
        assert "\U0001F324️" in output["returned"], output["returned"]

    def test_returned_html_does_not_contain_unavailable_status(self, render_weather_widget_source):
        output = self._render(render_weather_widget_source)
        html = output["returned"]
        assert "widget-status" not in html
        assert "unavailable" not in html.lower()

    def test_populates_weather_body_element_with_returned_html(self, render_weather_widget_source):
        output = self._render(render_weather_widget_source)
        assert output["bodyInnerHTML"] == output["returned"]
        assert "Loading" not in output["bodyInnerHTML"]


class TestRenderWeatherWidgetWithNullOrErrorData:
    """`renderWeatherWidget(data)` renders an inline "unavailable" status
    message into `#weather-body` when given null/undefined/error/empty
    input, instead of throwing or leaving the loading placeholder."""

    @pytest.mark.parametrize(
        "harness_arg",
        [
            "null",
            "undefined",
            "{}",
            json.dumps({"detail": "Failed to retrieve weather data"}),
            json.dumps({"error": "boom"}),
        ],
        ids=["null", "undefined", "empty-object", "detail-error", "error-shape"],
    )
    def test_renders_unavailable_message_for_invalid_input(
        self, render_weather_widget_source, harness_arg
    ):
        harness = (
            "var html = renderWeatherWidget(" + harness_arg + ");\n"
            "console.log(JSON.stringify({\n"
            "  returned: html,\n"
            "  bodyInnerHTML: global.__elements['weather-body'].innerHTML\n"
            "}));\n"
        )
        output = _run_node_harness(render_weather_widget_source, harness)

        html = output["returned"]
        assert "unavailable" in html.lower(), html
        assert "widget-status" in html

        # The DOM is updated with the same "unavailable" message.
        assert output["bodyInnerHTML"] == html

        # No live weather values leak into the unavailable state.
        assert "weather-main" not in html
        assert "weather-temp" not in html


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
