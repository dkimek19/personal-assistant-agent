"""Behavioral tests for the ``sanitizeHtml(htmlString)`` JS function.

Sub-AC 2 (governed dispatch)
-----------------------------
``assistant/static/index.html`` must define a top-level
``sanitizeHtml(htmlString)`` function that strips unsafe elements (e.g.
``<script>`` tags and their contents, inline event-handler attributes such as
``onerror=``, and ``javascript:``/``vbscript:``/``data:`` URLs) from a raw HTML
string while preserving allowed/benign markup (``<p>``, ``<strong>``, ``<em>``,
headings, lists, ``<a>``, ``<img>``, ``<pre>``/``<code>``, etc.) untouched.

This is verified end-to-end by extracting the function's source from
``index.html`` and executing it under Node.js, asserting the exact sanitized
HTML produced for inputs containing both ``<script>``/other dangerous markup
and benign tags.

``sanitizeHtml`` is defined at the top level of the second ``<script>`` block
(outside the app's DOM-setup IIFE), fully self-contained -- no
``document``/``window``/``marked`` dependencies -- specifically so it can be
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
    NODE_BIN is None, reason="Node.js is required to execute sanitizeHtml() for this test"
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
def sanitize_html_source(script_blocks: list[str]) -> str:
    candidates = [block for block in script_blocks if "function sanitizeHtml" in block]
    assert candidates, "expected a <script> block defining `sanitizeHtml`"
    return _extract_function_source(candidates[0], "sanitizeHtml")


def _run_node_harness(sanitize_html_source: str, harness_body: str) -> dict:
    """Run a small Node.js script: sanitizeHtml + harness_body.

    `harness_body` should `console.log(JSON.stringify(...))` its result.
    Returns the parsed JSON object printed by the script.
    """
    script = "\n".join([sanitize_html_source, harness_body])

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


class TestSanitizeHtmlStripsScriptTags:
    """The AC's named example: ``<script>`` tags."""

    def test_strips_script_tag_and_contents(self, sanitize_html_source):
        harness = """
console.log(JSON.stringify({
  result: sanitizeHtml('<p>Hello <script>alert(1)</script> world</p>')
}));
"""
        output = _run_node_harness(sanitize_html_source, harness)
        assert "<script" not in output["result"]
        assert "alert(1)" not in output["result"]
        assert output["result"] == "<p>Hello  world</p>"

    def test_strips_uppercase_script_tag(self, sanitize_html_source):
        harness = """
console.log(JSON.stringify({
  result: sanitizeHtml('before <SCRIPT>alert(1)</SCRIPT> after')
}));
"""
        output = _run_node_harness(sanitize_html_source, harness)
        assert "SCRIPT" not in output["result"]
        assert "alert(1)" not in output["result"]
        assert output["result"] == "before  after"

    def test_strips_script_with_src_attribute(self, sanitize_html_source):
        harness = """
console.log(JSON.stringify({
  result: sanitizeHtml('<p>safe</p><script src="evil.js"></script>')
}));
"""
        output = _run_node_harness(sanitize_html_source, harness)
        assert "<script" not in output["result"]
        assert output["result"] == "<p>safe</p>"

    def test_strips_multiple_script_tags(self, sanitize_html_source):
        harness = """
console.log(JSON.stringify({
  result: sanitizeHtml('<script>one()</script><p>mid</p><script>two()</script>')
}));
"""
        output = _run_node_harness(sanitize_html_source, harness)
        assert "<script" not in output["result"]
        assert "one()" not in output["result"]
        assert "two()" not in output["result"]
        assert output["result"] == "<p>mid</p>"


class TestSanitizeHtmlPreservesBenignMarkup:
    """Allowed formatting tags and their safe attributes pass through untouched."""

    def test_preserves_paragraph_and_inline_formatting(self, sanitize_html_source):
        harness = """
console.log(JSON.stringify({
  result: sanitizeHtml('<p>Hello <strong>bold</strong> and <em>italic</em> text</p>')
}));
"""
        output = _run_node_harness(sanitize_html_source, harness)
        assert output["result"] == "<p>Hello <strong>bold</strong> and <em>italic</em> text</p>"

    def test_preserves_headings_and_lists(self, sanitize_html_source):
        harness = """
console.log(JSON.stringify({
  result: sanitizeHtml('<h1>Title</h1><ul><li>one</li><li>two</li></ul>')
}));
"""
        output = _run_node_harness(sanitize_html_source, harness)
        assert output["result"] == "<h1>Title</h1><ul><li>one</li><li>two</li></ul>"

    def test_preserves_code_blocks(self, sanitize_html_source):
        harness = """
console.log(JSON.stringify({
  result: sanitizeHtml('<pre><code>const x = 1;</code></pre>')
}));
"""
        output = _run_node_harness(sanitize_html_source, harness)
        assert output["result"] == "<pre><code>const x = 1;</code></pre>"

    def test_preserves_safe_link_and_image(self, sanitize_html_source):
        harness = """
console.log(JSON.stringify({
  result: sanitizeHtml('<a href="https://example.com">link</a> <img src="pic.png" alt="pic">')
}));
"""
        output = _run_node_harness(sanitize_html_source, harness)
        assert output["result"] == (
            '<a href="https://example.com">link</a> <img src="pic.png" alt="pic">'
        )


class TestSanitizeHtmlStripsOtherDangerousElements:
    """``<style>``/``<iframe>``/``<object>``/``<embed>``/``<form>`` and stray
    void/dangerous tags are removed."""

    def test_strips_style_tag_and_contents(self, sanitize_html_source):
        harness = """
console.log(JSON.stringify({
  result: sanitizeHtml('<style>body{color:red}</style><p>text</p>')
}));
"""
        output = _run_node_harness(sanitize_html_source, harness)
        assert "<style" not in output["result"]
        assert "color:red" not in output["result"]
        assert output["result"] == "<p>text</p>"

    def test_strips_iframe(self, sanitize_html_source):
        harness = """
console.log(JSON.stringify({
  result: sanitizeHtml('<p>before</p><iframe src="https://evil.example"></iframe><p>after</p>')
}));
"""
        output = _run_node_harness(sanitize_html_source, harness)
        assert "<iframe" not in output["result"]
        assert output["result"] == "<p>before</p><p>after</p>"

    def test_strips_form_and_input(self, sanitize_html_source):
        harness = """
console.log(JSON.stringify({
  result: sanitizeHtml('<form action="https://evil.example"><input type="text" name="x"></form><p>safe</p>')
}));
"""
        output = _run_node_harness(sanitize_html_source, harness)
        assert "<form" not in output["result"]
        assert "<input" not in output["result"]
        assert output["result"] == "<p>safe</p>"

    def test_strips_stray_link_meta_base_tags(self, sanitize_html_source):
        harness = """
console.log(JSON.stringify({
  result: sanitizeHtml('<link rel="stylesheet" href="evil.css"><meta http-equiv="refresh" content="0;url=evil"><base href="https://evil.example"><p>text</p>')
}));
"""
        output = _run_node_harness(sanitize_html_source, harness)
        assert "<link" not in output["result"]
        assert "<meta" not in output["result"]
        assert "<base" not in output["result"]
        assert output["result"] == "<p>text</p>"

    def test_strips_button_tags_but_keeps_text(self, sanitize_html_source):
        harness = """
console.log(JSON.stringify({
  result: sanitizeHtml('<button onclick="evil()">Click</button>')
}));
"""
        output = _run_node_harness(sanitize_html_source, harness)
        assert "<button" not in output["result"]
        assert "onclick" not in output["result"]
        assert output["result"] == "Click"


class TestSanitizeHtmlStripsEventHandlerAttributes:
    def test_strips_onerror_attribute(self, sanitize_html_source):
        harness = """
console.log(JSON.stringify({
  result: sanitizeHtml('<img src="x.png" onerror="alert(1)">')
}));
"""
        output = _run_node_harness(sanitize_html_source, harness)
        assert "onerror" not in output["result"]
        assert output["result"] == '<img src="x.png">'

    def test_strips_onclick_attribute_preserving_other_attributes(self, sanitize_html_source):
        harness = """
console.log(JSON.stringify({
  result: sanitizeHtml('<a href="https://example.com" onclick="evil()">link</a>')
}));
"""
        output = _run_node_harness(sanitize_html_source, harness)
        assert "onclick" not in output["result"]
        assert output["result"] == '<a href="https://example.com">link</a>'


class TestSanitizeHtmlNeutralizesDangerousUrls:
    def test_neutralizes_javascript_protocol_href(self, sanitize_html_source):
        harness = """
console.log(JSON.stringify({
  result: sanitizeHtml('<a href="javascript:alert(1)">click me</a>')
}));
"""
        output = _run_node_harness(sanitize_html_source, harness)
        assert "javascript:" not in output["result"]
        assert output["result"] == "<a>click me</a>"

    def test_neutralizes_vbscript_protocol_href(self, sanitize_html_source):
        harness = """
console.log(JSON.stringify({
  result: sanitizeHtml('<a href="vbscript:msgbox(1)">click</a>')
}));
"""
        output = _run_node_harness(sanitize_html_source, harness)
        assert "vbscript:" not in output["result"]
        assert output["result"] == "<a>click</a>"

    def test_neutralizes_data_protocol_src(self, sanitize_html_source):
        harness = """
console.log(JSON.stringify({
  result: sanitizeHtml('<img src="data:text/html;base64,QUJD" alt="x">')
}));
"""
        output = _run_node_harness(sanitize_html_source, harness)
        assert "data:" not in output["result"]
        assert output["result"] == '<img alt="x">'

    def test_preserves_safe_https_url(self, sanitize_html_source):
        harness = """
console.log(JSON.stringify({
  result: sanitizeHtml('<a href="https://example.com/path?q=1">safe</a>')
}));
"""
        output = _run_node_harness(sanitize_html_source, harness)
        assert output["result"] == '<a href="https://example.com/path?q=1">safe</a>'


class TestSanitizeHtmlEdgeCases:
    def test_returns_empty_string_for_null_input(self, sanitize_html_source):
        harness = """
console.log(JSON.stringify({ result: sanitizeHtml(null) }));
"""
        output = _run_node_harness(sanitize_html_source, harness)
        assert output["result"] == ""

    def test_returns_empty_string_for_undefined_input(self, sanitize_html_source):
        harness = """
console.log(JSON.stringify({ result: sanitizeHtml(undefined) }));
"""
        output = _run_node_harness(sanitize_html_source, harness)
        assert output["result"] == ""

    def test_returns_empty_string_for_empty_input(self, sanitize_html_source):
        harness = """
console.log(JSON.stringify({ result: sanitizeHtml('') }));
"""
        output = _run_node_harness(sanitize_html_source, harness)
        assert output["result"] == ""

    def test_passes_through_plain_text_unchanged(self, sanitize_html_source):
        harness = """
console.log(JSON.stringify({ result: sanitizeHtml('just plain text') }));
"""
        output = _run_node_harness(sanitize_html_source, harness)
        assert output["result"] == "just plain text"

    def test_mixed_malicious_and_benign_markup(self, sanitize_html_source):
        # A single input combining script tags, an event handler, a
        # javascript: URL, and benign formatting -- the malicious content is
        # removed while the benign markup is fully preserved.
        harness = """
console.log(JSON.stringify({
  result: sanitizeHtml(
    '<h1>Report</h1>' +
    '<script>steal()</script>' +
    '<p>Hello <strong>world</strong>, <a href="javascript:evil()">click</a> or ' +
    '<a href="https://example.com">visit</a>.</p>' +
    '<img src="ok.png" onerror="evil()" alt="ok">'
  )
}));
"""
        output = _run_node_harness(sanitize_html_source, harness)
        result = output["result"]
        assert "<script" not in result
        assert "steal()" not in result
        assert "javascript:" not in result
        assert "onerror" not in result
        assert result == (
            "<h1>Report</h1>"
            "<p>Hello <strong>world</strong>, <a>click</a> or "
            '<a href="https://example.com">visit</a>.</p>'
            '<img src="ok.png" alt="ok">'
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
