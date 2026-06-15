"""Behavioral test for the ``fetchWeatherData()`` JS function (weather widget).

Sub-AC 2a (governed dispatch)
------------------------------
``assistant/static/index.html`` must define a top-level ``fetchWeatherData()``
function -- a "never-throw" sibling of ``fetchWeather()`` -- that:

- GETs ``/api/weather`` (the new backend pass-through endpoint backing the
  ontology's ``weather_data`` concept), and
- resolves with the parsed JSON response body on a successful response, or
- resolves with a normalized ``null`` result (instead of rejecting) when the
  request fails: a rejected ``fetch()`` call (network error), a non-OK HTTP
  response, or a response body that does not parse as JSON / is empty.

This is verified end-to-end by extracting the function's source from
``index.html`` and executing it under Node.js with a mocked
``global.fetch``, asserting both the recorded request URL and the value the
function resolves with for success, error (rejected/non-OK), and empty
response scenarios -- mirroring ``tests/test_weather_widget_fetch.py``'s
treatment of ``fetchWeather`` and ``tests/test_fetch_history.py``'s treatment
of ``fetchHistory``.

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
    NODE_BIN is None, reason="Node.js is required to execute fetchWeatherData() for this test"
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
def fetch_weather_data_source() -> str:
    html_text = INDEX_HTML_PATH.read_text(encoding="utf-8")

    # The app logic lives in the second <script> block (the first is the
    # inlined marked.js library); pick whichever block defines the function.
    script_blocks = re.findall(r"<script>(.*?)</script>", html_text, re.DOTALL)
    candidates = [block for block in script_blocks if "function fetchWeatherData" in block]
    assert candidates, "expected a <script> block defining `fetchWeatherData`"

    return _extract_function_source(candidates[0], "fetchWeatherData")


def _run_node_harness(fetch_weather_data_source: str, harness_body: str) -> dict:
    """Run a small Node.js script: mocked fetch + fetchWeatherData + harness_body.

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
            fetch_weather_data_source,
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


_HARNESS_TEMPLATE = """
global.__mockResponse = function () {{
  {mock_response_body}
}};

fetchWeatherData().then(function (data) {{
  console.log(JSON.stringify({{ calls: __calls, result: data }}));
}}).catch(function (err) {{
  console.log(JSON.stringify({{ error: err.message }}));
}});
"""


class TestFetchWeatherDataRequest:
    """The function calls the backend weather endpoint via GET."""

    def test_calls_weather_endpoint(self, fetch_weather_data_source):
        harness = _HARNESS_TEMPLATE.format(
            mock_response_body=(
                "return Promise.resolve({ ok: true, status: 200, "
                "json: function () { return Promise.resolve("
                + json.dumps(_SAMPLE_WEATHER_DATA)
                + "); } });"
            )
        )
        output = _run_node_harness(fetch_weather_data_source, harness)

        assert "error" not in output, output
        assert len(output["calls"]) == 1
        assert output["calls"][0]["url"] == "/api/weather"


class TestFetchWeatherDataSuccess:
    """On a successful response, resolves with the parsed JSON body."""

    def test_resolves_with_parsed_weather_data_on_success(self, fetch_weather_data_source):
        harness = _HARNESS_TEMPLATE.format(
            mock_response_body=(
                "return Promise.resolve({ ok: true, status: 200, "
                "json: function () { return Promise.resolve("
                + json.dumps(_SAMPLE_WEATHER_DATA)
                + "); } });"
            )
        )
        output = _run_node_harness(fetch_weather_data_source, harness)

        assert "error" not in output, output
        assert output["result"] == _SAMPLE_WEATHER_DATA


class TestFetchWeatherDataFailureNormalization:
    """On any failure, resolves (never rejects) with a normalized `null`."""

    def test_resolves_with_null_on_non_ok_response(self, fetch_weather_data_source):
        harness = _HARNESS_TEMPLATE.format(
            mock_response_body=(
                "return Promise.resolve({ ok: false, status: 502, "
                "json: function () { return Promise.resolve({ detail: 'Failed to retrieve weather data' }); } });"
            )
        )
        output = _run_node_harness(fetch_weather_data_source, harness)

        assert "error" not in output, output
        assert output["result"] is None

    def test_resolves_with_null_when_fetch_itself_rejects(self, fetch_weather_data_source):
        """A network-level failure (the `fetch()` call itself rejects, e.g.
        the browser couldn't reach the backend at all) must be normalized to
        a resolved `null` -- not a rejected promise -- so callers never need
        a `.catch()`.
        """
        harness = _HARNESS_TEMPLATE.format(
            mock_response_body="return Promise.reject(new TypeError('Failed to fetch'));"
        )
        output = _run_node_harness(fetch_weather_data_source, harness)

        assert "error" not in output, output
        assert output["result"] is None

    def test_resolves_with_null_when_response_body_is_not_json(self, fetch_weather_data_source):
        """An ok response whose body cannot be parsed as JSON (e.g. an empty
        body) must also normalize to `null` rather than rejecting.
        """
        harness = _HARNESS_TEMPLATE.format(
            mock_response_body=(
                "return Promise.resolve({ ok: true, status: 200, "
                "json: function () { return Promise.reject(new SyntaxError('Unexpected end of JSON input')); } });"
            )
        )
        output = _run_node_harness(fetch_weather_data_source, harness)

        assert "error" not in output, output
        assert output["result"] is None


class TestFetchWeatherDataEmptyResponse:
    """An ok response with an empty JSON object is still "success"."""

    def test_resolves_with_empty_object_on_empty_response(self, fetch_weather_data_source):
        harness = _HARNESS_TEMPLATE.format(
            mock_response_body=(
                "return Promise.resolve({ ok: true, status: 200, "
                "json: function () { return Promise.resolve({}); } });"
            )
        )
        output = _run_node_harness(fetch_weather_data_source, harness)

        assert "error" not in output, output
        assert output["result"] == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
