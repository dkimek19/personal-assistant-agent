"""Behavioral tests for the ``renderMarkdown(text)`` JS function.

Sub-AC 2 (governed dispatch)
-----------------------------
``assistant/static/index.html`` must define a top-level ``renderMarkdown(text)``
function (or module) that converts a markdown-formatted reply string -- the
ontology's ``chat_response.reply`` -- into sanitized HTML suitable for
insertion into an assistant ``chat_message`` bubble via ``innerHTML``.

This is verified end-to-end by extracting the function's source from
``index.html`` (along with the inlined ``marked.js`` library it depends on)
and executing it under Node.js, asserting the exact HTML produced for sample
markdown inputs -- bold text, lists, and fenced code blocks, per the AC's
examples -- as well as sanitization of dangerous constructs (``<script>``
tags, ``on*`` event-handler attributes, and ``javascript:`` URLs) so the
output is safe to assign to ``innerHTML``.

``renderMarkdown`` is defined at the top level of the second ``<script>``
block (outside the app's DOM-setup IIFE) specifically so it has no
``document``/``window`` dependencies beyond the inlined ``marked`` library and
can be evaluated standalone like this.
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
    NODE_BIN is None, reason="Node.js is required to execute renderMarkdown() for this test"
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
def render_markdown_source(script_blocks: list[str]) -> str:
    candidates = [block for block in script_blocks if "function renderMarkdown" in block]
    assert candidates, "expected a <script> block defining `renderMarkdown`"
    return _extract_function_source(candidates[0], "renderMarkdown")


@pytest.fixture(scope="module")
def markdown_to_html_source(script_blocks: list[str]) -> str:
    candidates = [block for block in script_blocks if "function markdownToHtml" in block]
    assert candidates, "expected a <script> block defining `markdownToHtml`"
    return _extract_function_source(candidates[0], "markdownToHtml")


@pytest.fixture(scope="module")
def sanitize_html_source(script_blocks: list[str]) -> str:
    candidates = [block for block in script_blocks if "function sanitizeHtml" in block]
    assert candidates, "expected a <script> block defining `sanitizeHtml`"
    return _extract_function_source(candidates[0], "sanitizeHtml")


def _run_node_harness(marked_source: str, render_markdown_source: str, harness_body: str) -> dict:
    """Run a small Node.js script: inlined marked.js + renderMarkdown + harness_body.

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
            render_markdown_source,
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


# The UMD wrapper assigns the marked module to `module.exports` when run
# under Node/CommonJS (which `node -e` provides). Make it available as a
# bare/global `marked`, mirroring how the browser exposes `window.marked`
# from the same inlined script.
_MARKED_GLOBAL_SHIM = (
    "if (typeof module !== 'undefined' && module.exports && module.exports.marked) {"
    " global.marked = module.exports.marked; }"
)


def _run_node_harness_multi(js_sources: list[str], harness_body: str) -> dict:
    """Run a Node.js script composed of several JS source fragments plus a harness.

    `harness_body` should `console.log(JSON.stringify(...))` its result.
    Returns the parsed JSON object printed by the script.
    """
    script = "\n".join([*js_sources, harness_body])

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


class TestRenderMarkdownBasicFormatting:
    """The AC's named examples: bold text, links, code, line breaks, lists,
    and fenced code blocks."""

    def test_renders_bold_text(self, marked_source, render_markdown_source):
        harness = """
console.log(JSON.stringify({ result: renderMarkdown('**bold text**') }));
"""
        output = _run_node_harness(marked_source, render_markdown_source, harness)
        assert output["result"] == "<p><strong>bold text</strong></p>\n"

    def test_renders_link(self, marked_source, render_markdown_source):
        # A standard (safe) markdown link -- one of the AC's named sample
        # markdown strings ("links") -- renders to an anchor tag with its
        # `href` attribute preserved.
        harness = """
console.log(JSON.stringify({ result: renderMarkdown('[OpenAI](https://example.com)') }));
"""
        output = _run_node_harness(marked_source, render_markdown_source, harness)
        assert output["result"] == '<p><a href="https://example.com">OpenAI</a></p>\n'

    def test_renders_inline_code(self, marked_source, render_markdown_source):
        # Inline code span -- one of the AC's named sample markdown strings
        # ("code").
        harness = """
console.log(JSON.stringify({ result: renderMarkdown('Use `code` here') }));
"""
        output = _run_node_harness(marked_source, render_markdown_source, harness)
        assert output["result"] == "<p>Use <code>code</code> here</p>\n"

    def test_renders_line_break(self, marked_source, render_markdown_source):
        # A hard line break (trailing double-space + newline) -- one of the
        # AC's named sample markdown strings ("line breaks") -- renders to a
        # `<br>` within the same paragraph.
        harness = r"""
console.log(JSON.stringify({ result: renderMarkdown('line one  \nline two') }));
"""
        output = _run_node_harness(marked_source, render_markdown_source, harness)
        assert output["result"] == "<p>line one<br>line two</p>\n"

    def test_renders_unordered_list(self, marked_source, render_markdown_source):
        harness = r"""
console.log(JSON.stringify({ result: renderMarkdown('- one\n- two\n- three') }));
"""
        output = _run_node_harness(marked_source, render_markdown_source, harness)
        assert output["result"] == (
            "<ul>\n<li>one</li>\n<li>two</li>\n<li>three</li>\n</ul>\n"
        )

    def test_renders_fenced_code_block(self, marked_source, render_markdown_source):
        harness = r"""
console.log(JSON.stringify({ result: renderMarkdown('```\nconst x = 1;\n```') }));
"""
        output = _run_node_harness(marked_source, render_markdown_source, harness)
        assert output["result"] == "<pre><code>const x = 1;\n</code></pre>\n"

    def test_renders_mixed_markdown_reply(self, marked_source, render_markdown_source):
        # A realistic `chat_response.reply` combining a heading, bold text,
        # and a list -- representative of an assistant reply rendered into a
        # chat_message bubble.
        harness = r"""
console.log(JSON.stringify({ result: renderMarkdown('# Title\n\nHere is **important** info:\n\n- alpha\n- beta') }));
"""
        output = _run_node_harness(marked_source, render_markdown_source, harness)
        assert output["result"] == (
            "<h1>Title</h1>\n"
            "<p>Here is <strong>important</strong> info:</p>\n"
            "<ul>\n<li>alpha</li>\n<li>beta</li>\n</ul>\n"
        )


class TestRenderMarkdownSanitization:
    """`renderMarkdown` returns *sanitized* HTML safe for `innerHTML`."""

    def test_strips_script_tags_and_contents(self, marked_source, render_markdown_source):
        harness = """
console.log(JSON.stringify({ result: renderMarkdown('Hello <script>alert(1)</script> world') }));
"""
        output = _run_node_harness(marked_source, render_markdown_source, harness)
        assert "<script" not in output["result"]
        assert "alert(1)" not in output["result"]
        assert output["result"] == "<p>Hello  world</p>\n"

    def test_strips_event_handler_attributes(self, marked_source, render_markdown_source):
        harness = """
console.log(JSON.stringify({ result: renderMarkdown('<img src="x.png" onerror="alert(1)">') }));
"""
        output = _run_node_harness(marked_source, render_markdown_source, harness)
        assert "onerror" not in output["result"]
        assert output["result"] == '<img src="x.png">'

    def test_neutralizes_javascript_protocol_links(self, marked_source, render_markdown_source):
        harness = """
console.log(JSON.stringify({ result: renderMarkdown('[click me](javascript:alert(1))') }));
"""
        output = _run_node_harness(marked_source, render_markdown_source, harness)
        assert "javascript:" not in output["result"]
        assert output["result"] == "<p><a>click me</a></p>\n"

    def test_strips_embedded_script_while_preserving_formatting(self, marked_source, render_markdown_source):
        # A single markdown input combining formatting (a heading and bold
        # emphasis) with an embedded <script> tag, exercising
        # `renderMarkdown`'s full markdown -> HTML -> sanitized-HTML pipeline
        # end-to-end on one realistic `chat_response.reply`-style input.
        harness = r"""
console.log(JSON.stringify({ result: renderMarkdown('# Report\n\n**Important**: <script>alert(1)</script> please review.') }));
"""
        output = _run_node_harness(marked_source, render_markdown_source, harness)
        assert "<script" not in output["result"]
        assert "alert(1)" not in output["result"]
        assert output["result"] == (
            "<h1>Report</h1>\n"
            "<p><strong>Important</strong>:  please review.</p>\n"
        )


class TestRenderMarkdownPipelineComposition:
    """`renderMarkdown` composes `markdownToHtml` and `sanitizeHtml` into a
    single pipeline: markdown source -> raw HTML (the same conversion as
    `markdownToHtml`) -> sanitized HTML (the same sanitization as
    `sanitizeHtml`).

    Verified end-to-end by running all three functions together under Node
    on a markdown input that combines formatting (a heading and bold
    emphasis) with an embedded `<script>` tag, and asserting that calling
    `renderMarkdown` directly produces exactly the same sanitized HTML as
    explicitly piping the input through `markdownToHtml` and then
    `sanitizeHtml`.
    """

    def test_renderMarkdown_equals_sanitizeHtml_of_markdownToHtml(
        self,
        marked_source,
        markdown_to_html_source,
        sanitize_html_source,
        render_markdown_source,
    ):
        harness = r"""
var input = '# Report\n\n**Important**: <script>alert(1)</script> please review.';
var direct = renderMarkdown(input);
var composed = sanitizeHtml(markdownToHtml(input));
console.log(JSON.stringify({ direct: direct, composed: composed }));
"""
        output = _run_node_harness_multi(
            [
                marked_source,
                _MARKED_GLOBAL_SHIM,
                markdown_to_html_source,
                sanitize_html_source,
                render_markdown_source,
            ],
            harness,
        )

        # renderMarkdown's pipeline produces exactly the same result as
        # explicitly composing markdownToHtml -> sanitizeHtml.
        assert output["direct"] == output["composed"]

        # The composed/sanitized output preserves the markdown formatting
        # (heading + bold emphasis) while the <script> tag and its contents
        # are removed.
        assert "<script" not in output["direct"]
        assert "alert(1)" not in output["direct"]
        assert output["direct"] == (
            "<h1>Report</h1>\n"
            "<p><strong>Important</strong>:  please review.</p>\n"
        )


class TestRenderMarkdownEdgeCases:
    def test_renders_empty_string_for_null_input(self, marked_source, render_markdown_source):
        harness = """
console.log(JSON.stringify({ result: renderMarkdown(null) }));
"""
        output = _run_node_harness(marked_source, render_markdown_source, harness)
        assert output["result"] == ""

    def test_renders_empty_string_for_undefined_input(self, marked_source, render_markdown_source):
        harness = """
console.log(JSON.stringify({ result: renderMarkdown(undefined) }));
"""
        output = _run_node_harness(marked_source, render_markdown_source, harness)
        assert output["result"] == ""

    def test_renders_plain_text_paragraph(self, marked_source, render_markdown_source):
        harness = """
console.log(JSON.stringify({ result: renderMarkdown('just plain text') }));
"""
        output = _run_node_harness(marked_source, render_markdown_source, harness)
        assert output["result"] == "<p>just plain text</p>\n"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
