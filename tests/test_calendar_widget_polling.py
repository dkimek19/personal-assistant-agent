"""Polling behavior test for the calendar sidebar widget in
``assistant/static/index.html``.

Sub-AC 2 (governed dispatch)
-----------------------------
The calendar widget's polling function (``loadCalendar``, wired up in the
boot IIFE of the app's ``<script>`` block) must:

- call ``GET /api/calendar/today`` immediately on page load (the ontology's
  ``calendar_events`` fetch), and
- register a ``setInterval`` with a delay of ``60000`` ms (the ontology's
  ``polling_interval``) whose callback re-fetches ``/api/calendar/today``.

This is verified end-to-end by extracting the app's ``<script>`` block (the
one defining ``sendChatMessage``, ``renderMarkdown``, and the boot IIFE) from
index.html and executing it under Node.js with:

- a mocked ``global.fetch`` that records every call (URL + options) without
  hitting the network,
- a mocked ``global.setInterval`` that records ``(callback, delay)`` pairs
  instead of scheduling real timers, and
- a minimal ``document``/``window`` shim so the boot IIFE's DOM lookups
  (``document.getElementById``, etc.) resolve to harmless stub elements.

The test then asserts on the recorded fetch call counts (before/after
manually invoking the captured interval callbacks) and on the recorded
interval delay, without needing real wall-clock time to elapse.
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


# ---------------------------------------------------------------------------
# Mocks for fetch / setInterval / document / window
# ---------------------------------------------------------------------------
#
# Records every fetch() call (so we can count/inspect calls to
# `/api/calendar/today`) and every setInterval() registration (so we can
# inspect the delay and manually invoke the callback to simulate elapsed
# time, instead of waiting on real timers).
_MOCK_PRELUDE = """
global.__fetchCalls = [];
global.fetch = function (url, options) {
  global.__fetchCalls.push({ url: url, options: options || null });
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

function __makeMockElement(id) {
  return {
    id: id,
    innerHTML: '',
    textContent: '',
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
}

global.document = {
  getElementById: function (id) { return __makeMockElement(id); },
  createElement: function () { return __makeMockElement(); }
};

global.window = global;
"""


@pytest.fixture(scope="module")
def app_script_source() -> str:
    """Source of the `<script>` block defining the boot IIFE.

    This is the second inlined `<script>` block (the first is the marked.js
    library): it defines `sendChatMessage`, `renderMarkdown`, and the
    top-level IIFE that wires up the sidebar widgets' polling on load.
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


class TestCalendarWidgetCallsEndpointImmediatelyOnLoad:
    """`loadCalendar()` (and thus `GET /api/calendar/today`) runs once during boot."""

    def test_fetches_calendar_endpoint_exactly_once_on_load(self, app_script_source):
        harness = """
console.log(JSON.stringify({
  calendarCalls: global.__fetchCalls.filter(function (c) { return c.url === '/api/calendar/today'; }).length,
  intervalCount: global.__intervals.length,
  intervalDelays: global.__intervals.map(function (iv) { return iv.ms; })
}));
"""
        output = _run_node_harness(app_script_source, harness)

        assert output["calendarCalls"] == 1, (
            "expected exactly one GET /api/calendar/today call immediately on load, "
            f"got {output['calendarCalls']}"
        )
        assert output["intervalCount"] >= 1, "expected at least one setInterval registration"


class TestCalendarWidgetPollsEvery60Seconds:
    """A `setInterval(..., 60000)` registration re-fetches `/api/calendar/today`."""

    def test_one_registered_interval_has_60_second_delay_and_refetches_calendar(
        self, app_script_source
    ):
        harness = """
var results = global.__intervals.map(function (iv) {
  var before = global.__fetchCalls.filter(function (c) { return c.url === '/api/calendar/today'; }).length;
  iv.fn();
  var after = global.__fetchCalls.filter(function (c) { return c.url === '/api/calendar/today'; }).length;
  return { ms: iv.ms, calendarCallDelta: after - before };
});
console.log(JSON.stringify({ results: results }));
"""
        output = _run_node_harness(app_script_source, harness)

        calendar_pollers = [r for r in output["results"] if r["calendarCallDelta"] == 1]
        assert len(calendar_pollers) == 1, (
            "expected exactly one registered setInterval whose callback re-fetches "
            f"/api/calendar/today, got: {output['results']!r}"
        )
        assert calendar_pollers[0]["ms"] == 60000, (
            "expected the calendar polling interval delay to be 60000ms "
            f"(60s), got {calendar_pollers[0]['ms']!r}"
        )

    def test_calls_calendar_endpoint_again_after_interval_fires(self, app_script_source):
        harness = """
var before = global.__fetchCalls.filter(function (c) { return c.url === '/api/calendar/today'; }).length;
global.__intervals.forEach(function (iv) { iv.fn(); });
var after = global.__fetchCalls.filter(function (c) { return c.url === '/api/calendar/today'; }).length;
console.log(JSON.stringify({ before: before, after: after }));
"""
        output = _run_node_harness(app_script_source, harness)

        assert output["before"] == 1, "expected one /api/calendar/today call before any interval fires"
        assert output["after"] == 2, (
            "expected a second /api/calendar/today call after the 60s interval callback "
            "fires (simulated)"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
