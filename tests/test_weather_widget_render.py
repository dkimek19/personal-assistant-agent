"""DOM rendering test for the weather sidebar widget in
``assistant/static/index.html``.

Sub-AC 2a (governed dispatch)
------------------------------
The weather widget module (``loadWeather``, wired up in the boot IIFE of the
app's ``<script>`` block) must:

- fetch data from its backend endpoint (``GET /api/weather``) on page load,
  and
- render the returned ``weather_data`` (the ontology's ``weather_data``
  concept) into the sidebar's ``#weather-body`` element, replacing the
  initial "Loading..." placeholder.

This is verified end-to-end by extracting the app's ``<script>`` block (the
one defining ``sendChatMessage``, ``renderMarkdown``, and the boot IIFE) from
index.html and executing it under Node.js with:

- a mocked ``global.fetch`` that resolves ``GET /api/weather`` with a
  representative ``weather_data`` payload without hitting the network,
- a mocked ``global.setInterval`` so the 60s polling registration doesn't
  schedule real timers, and
- a minimal ``document``/``window`` shim whose ``getElementById`` returns a
  *persistent* element per id (stored in ``global.__elements``), so mutations
  made by ``loadWeather()`` (e.g. ``bodyEl.innerHTML = ...``) are observable
  by the test harness after the boot IIFE runs.

Because ``loadWeather()``'s fetch-then-render chain resolves via
microtasks, the harness defers its assertions to a ``setTimeout(fn, 0)``
macrotask, which Node guarantees runs only after the microtask queue (and
thus the render) has fully drained.
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
    reason="Node.js is required to execute the boot/polling IIFE for this test",
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


# ---------------------------------------------------------------------------
# Mocks for fetch / setInterval / document / window
# ---------------------------------------------------------------------------
#
# `GET /api/weather` resolves with `_SAMPLE_WEATHER_DATA`; any other URL
# resolves with an empty object (the calendar/notes widgets aren't under
# test here, but the boot IIFE still calls their loaders).
#
# `document.getElementById` returns a *persistent* element per id, recorded
# in `global.__elements`, so the test harness can inspect the same element
# object that `loadWeather()` mutated (e.g. `bodyEl.innerHTML = ...`).
_MOCK_PRELUDE = (
    """
global.__fetchCalls = [];
global.__weatherData = """
    + json.dumps(_SAMPLE_WEATHER_DATA)
    + """;

global.fetch = function (url, options) {
  global.__fetchCalls.push({ url: url, options: options || null });
  if (url === '/api/weather') {
    return Promise.resolve({
      ok: true,
      status: 200,
      json: function () { return Promise.resolve(global.__weatherData); }
    });
  }
  return Promise.resolve({
    ok: true,
    status: 200,
    json: function () { return Promise.resolve({}); }
  });
};

global.__intervals = [];
global.setInterval = function (fn, ms) {
  var id = global.__intervals.length + 1;
  global.__intervals.push({ fn: fn, ms: ms, id: id });
  return id;
};
global.clearInterval = function () {};

global.__elements = {};

// `escapeHtml()` in the app creates a throwaway <div>, sets its
// `textContent`, then reads back `innerHTML` to obtain HTML-escaped text.
// Real DOM elements link those two properties; this mock element does the
// same via a getter/setter so `escapeHtml()` (and thus `renderWeather()`)
// behaves the same as it would in a browser.
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
    innerHTML: '',
    value: '',
    style: {},
    disabled: false,
    scrollTop: 0,
    scrollHeight: 0,
    classList: {
      add: function () {},
      remove: function () {},
      contains: function () { return false; }
    },
    addEventListener: function () {},
    appendChild: function (child) { return child; },
    remove: function () {},
    focus: function () {}
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
)


@pytest.fixture(scope="module")
def app_script_source() -> str:
    """Source of the `<script>` block defining the boot IIFE.

    This is the second inlined `<script>` block (the first is the marked.js
    library): it defines `sendChatMessage`, `renderMarkdown`, and the
    top-level IIFE that wires up the sidebar widgets (including
    `loadWeather`) on load.
    """
    html_text = INDEX_HTML_PATH.read_text(encoding="utf-8")
    script_blocks = re.findall(r"<script>(.*?)</script>", html_text, re.DOTALL)
    candidates = [block for block in script_blocks if "function sendChatMessage" in block]
    assert candidates, "expected a <script> block defining `sendChatMessage` and the boot IIFE"
    return candidates[0]


def _run_node_harness(app_script_source: str, harness_body: str) -> dict:
    """Run the mocked prelude + app script + harness body under Node.js.

    `harness_body` should `console.log(JSON.stringify(...))` its result,
    which is parsed and returned as a dict.
    """
    script = "\n".join([_MOCK_PRELUDE, app_script_source, harness_body])

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


# Defer assertions to a macrotask (setTimeout) so the microtask queue --
# i.e. the fetch().then(...).then(...) chain inside loadWeather() -- has
# fully drained and the DOM mutation has happened by the time we inspect it.
_AFTER_RENDER_HARNESS = """
setTimeout(function () {
  var body = global.__elements['weather-body'];
  var updated = global.__elements['weather-updated'];
  console.log(JSON.stringify({
    weatherCalls: global.__fetchCalls.filter(function (c) { return c.url === '/api/weather'; }).length,
    bodyInnerHTML: body ? body.innerHTML : null,
    updatedText: updated ? updated.textContent : null
  }));
}, 0);
"""


class TestWeatherWidgetRendersFetchedDataIntoSidebar:
    """`loadWeather()` fetches `/api/weather` on load and renders the result
    into `#weather-body`, replacing the initial loading placeholder."""

    def test_renders_temperature_description_and_location_into_weather_body(
        self, app_script_source
    ):
        output = _run_node_harness(app_script_source, _AFTER_RENDER_HARNESS)

        assert output["weatherCalls"] == 1, "expected one GET /api/weather call on load"

        body_html = output["bodyInnerHTML"]
        assert body_html, "expected #weather-body to have rendered content"

        # The initial "Loading..." placeholder must be replaced.
        assert "Loading" not in body_html
        assert "widget-status" not in body_html, (
            "a successful response should render weather data, not a status message"
        )

        # Temperature is rounded and rendered in degrees Celsius.
        assert "22°C" in body_html, body_html

        # Weather description and location name are rendered verbatim.
        assert "Mainly clear" in body_html
        assert "Seoul" in body_html

        # Wind speed is rounded and rendered alongside the location.
        assert "Wind 9 km/h" in body_html

    def test_renders_weather_icon_for_weather_code(self, app_script_source):
        output = _run_node_harness(app_script_source, _AFTER_RENDER_HARNESS)

        body_html = output["bodyInnerHTML"]
        # weather_code 1 maps to the "mainly clear" icon.
        assert "\U0001F324️" in body_html, body_html

    def test_updates_the_last_updated_timestamp(self, app_script_source):
        output = _run_node_harness(app_script_source, _AFTER_RENDER_HARNESS)

        updated_text = output["updatedText"]
        assert updated_text, "expected #weather-updated to be populated after render"
        assert updated_text.startswith("Updated "), updated_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
