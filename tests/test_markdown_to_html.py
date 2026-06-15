"""Behavioral tests for the ``markdownToHtml(markdownText)`` JS function.

Sub-AC 1 (governed dispatch)
-----------------------------
``assistant/static/index.html`` must define a top-level
``markdownToHtml(markdownText)`` function that converts markdown syntax --
headings, lists, bold/italic, inline code, and fenced code blocks -- into the
corresponding raw HTML elements via the inlined marked.js library (offline,
no CDN).

Unlike ``renderMarkdown(text)`` (covered by ``test_render_markdown.py``),
``markdownToHtml`` performs *no* sanitization: its output is the raw,
unsanitized HTML structure produced by marked.js. This is verified here by
extracting the function's source from ``index.html`` (along with the inlined
``marked.js`` library it depends on) and executing it under Node.js, asserting
the exact HTML produced for sample markdown inputs covering each syntax type
named in the AC, plus an edge case demonstrating that raw HTML (e.g. a
``<script>`` tag) passes through unmodified -- i.e. the structure is
unsanitized, in contrast to ``renderMarkdown``.

``markdownToHtml`` is defined at the top level of the second ``<script>``
block (outside the app's DOM-setup IIFE), self-contained (no dependency on
``renderMarkdown``), specifically so it has no ``document``/``window``
dependencies beyond the inlined ``marked`` library and can be evaluated
standalone like this.
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
    NODE_BIN is None, reason="Node.js is required to execute markdownToHtml() for this test"
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
def markdown_to_html_source(script_blocks: list[str]) -> str:
    candidates = [block for block in script_blocks if "function markdownToHtml" in block]
    assert candidates, "expected a <script> block defining `markdownToHtml`"
    return _extract_function_source(candidates[0], "markdownToHtml")


def _run_node_harness(marked_source: str, markdown_to_html_source: str, harness_body: str) -> dict:
    """Run a small Node.js script: inlined marked.js + markdownToHtml + harness_body.

    `harness_body` should `console.log(JSON.stringify(...))` its result.
    Returns the parsed JSON object printed by the script.
    """
    script = "\n".join(
        [
            marked_source,
            # The UMD wrapper assigns the marked module to `module.exports`
            # when run under Node/CommonJS (which `node -e` provides). Make
            # it available as a bare/global `marked`, mirroring how the
            # browser exposes `window.marked` from the same inlined script.
            "if (typeof module !== 'undefined' && module.exports && module.exports.marked) {"
            " global.marked = module.exports.marked; }",
            markdown_to_html_source,
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


class TestMarkdownToHtmlSyntaxTypes:
    """The AC's named syntax types: headings, lists, bold/italic, inline
    code, and fenced code blocks -- each producing the corresponding raw
    HTML element."""

    def test_renders_heading(self, marked_source, markdown_to_html_source):
        harness = """
console.log(JSON.stringify({ result: markdownToHtml('# Heading') }));
"""
        output = _run_node_harness(marked_source, markdown_to_html_source, harness)
        assert output["result"] == "<h1>Heading</h1>\n"

    def test_renders_unordered_list(self, marked_source, markdown_to_html_source):
        harness = r"""
console.log(JSON.stringify({ result: markdownToHtml('- one\n- two\n- three') }));
"""
        output = _run_node_harness(marked_source, markdown_to_html_source, harness)
        assert output["result"] == (
            "<ul>\n<li>one</li>\n<li>two</li>\n<li>three</li>\n</ul>\n"
        )

    def test_renders_bold_text(self, marked_source, markdown_to_html_source):
        harness = """
console.log(JSON.stringify({ result: markdownToHtml('**bold text**') }));
"""
        output = _run_node_harness(marked_source, markdown_to_html_source, harness)
        assert output["result"] == "<p><strong>bold text</strong></p>\n"

    def test_renders_italic_text(self, marked_source, markdown_to_html_source):
        harness = """
console.log(JSON.stringify({ result: markdownToHtml('*italic text*') }));
"""
        output = _run_node_harness(marked_source, markdown_to_html_source, harness)
        assert output["result"] == "<p><em>italic text</em></p>\n"

    def test_renders_inline_code(self, marked_source, markdown_to_html_source):
        harness = """
console.log(JSON.stringify({ result: markdownToHtml('Use `code` here') }));
"""
        output = _run_node_harness(marked_source, markdown_to_html_source, harness)
        assert output["result"] == "<p>Use <code>code</code> here</p>\n"

    def test_renders_fenced_code_block(self, marked_source, markdown_to_html_source):
        harness = r"""
console.log(JSON.stringify({ result: markdownToHtml('```\nconst x = 1;\n```') }));
"""
        output = _run_node_harness(marked_source, markdown_to_html_source, harness)
        assert output["result"] == "<pre><code>const x = 1;\n</code></pre>\n"

    def test_renders_mixed_markdown(self, marked_source, markdown_to_html_source):
        # A realistic combination of a heading, bold text, and a list.
        harness = r"""
console.log(JSON.stringify({ result: markdownToHtml('# Title\n\nHere is **important** info:\n\n- alpha\n- beta') }));
"""
        output = _run_node_harness(marked_source, markdown_to_html_source, harness)
        assert output["result"] == (
            "<h1>Title</h1>\n"
            "<p>Here is <strong>important</strong> info:</p>\n"
            "<ul>\n<li>alpha</li>\n<li>beta</li>\n</ul>\n"
        )


class TestMarkdownToHtmlIsUnsanitized:
    """`markdownToHtml` returns the *raw* HTML structure from marked.js,
    with no sanitization pass -- in contrast to `renderMarkdown`."""

    def test_passes_through_raw_script_tags_unmodified(self, marked_source, markdown_to_html_source):
        harness = """
console.log(JSON.stringify({ result: markdownToHtml('Hello <script>alert(1)</script> world') }));
"""
        output = _run_node_harness(marked_source, markdown_to_html_source, harness)
        assert "<script>alert(1)</script>" in output["result"]

    def test_passes_through_event_handler_attributes_unmodified(self, marked_source, markdown_to_html_source):
        harness = """
console.log(JSON.stringify({ result: markdownToHtml('<img src="x.png" onerror="alert(1)">') }));
"""
        output = _run_node_harness(marked_source, markdown_to_html_source, harness)
        assert "onerror" in output["result"]


class TestMarkdownToHtmlEdgeCases:
    def test_renders_empty_string_for_null_input(self, marked_source, markdown_to_html_source):
        harness = """
console.log(JSON.stringify({ result: markdownToHtml(null) }));
"""
        output = _run_node_harness(marked_source, markdown_to_html_source, harness)
        assert output["result"] == ""

    def test_renders_empty_string_for_undefined_input(self, marked_source, markdown_to_html_source):
        harness = """
console.log(JSON.stringify({ result: markdownToHtml(undefined) }));
"""
        output = _run_node_harness(marked_source, markdown_to_html_source, harness)
        assert output["result"] == ""

    def test_renders_plain_text_paragraph(self, marked_source, markdown_to_html_source):
        harness = """
console.log(JSON.stringify({ result: markdownToHtml('just plain text') }));
"""
        output = _run_node_harness(marked_source, markdown_to_html_source, harness)
        assert output["result"] == "<p>just plain text</p>\n"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
