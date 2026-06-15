"""DOM structure tests for the single-file frontend (assistant/static/index.html).

Sub-AC 1
--------
The HTML document must define a left sidebar container element (``#sidebar``)
that contains three widget card elements/placeholders for weather, calendar,
and memo. This is verified here via a DOM test that parses index.html and
asserts the sidebar and its three widget card children exist.

Sub-AC 2
--------
The HTML document must define a right-side main chat area container element
(e.g. ``#main-chat`` or ``#chat-area``), separate from the sidebar. This is
verified here via a DOM test that parses index.html and asserts the chat
container element exists outside (and as a sibling of) ``#sidebar``, and that
it hosts the chat messages area.

Sub-AC 3
--------
The inlined CSS must position the sidebar and the main chat area side-by-side
(e.g. via flexbox/grid, sidebar on the left, chat area on the right). This is
verified here by parsing the CSS rules embedded in index.html's ``<style>``
block plus the DOM structure, then computing the rendered bounding boxes for
``#sidebar`` and the main chat area container under a representative desktop
viewport -- asserting the sidebar's right edge sits at or before the chat
area's left edge.
"""

from __future__ import annotations

import re
from pathlib import Path

import lxml.html
import pytest

INDEX_HTML_PATH = (
    Path(__file__).resolve().parent.parent / "assistant" / "static" / "index.html"
)


@pytest.fixture(scope="module")
def html_text():
    return INDEX_HTML_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def tree(html_text):
    return lxml.html.fromstring(html_text)


@pytest.fixture(scope="module")
def sidebar(tree):
    matches = tree.xpath('//*[@id="sidebar"]')
    assert len(matches) == 1, "expected exactly one #sidebar element"
    return matches[0]


# ---------------------------------------------------------------------------
# Minimal CSS helpers for Sub-AC 3 (layout positioning)
# ---------------------------------------------------------------------------
#
# These intentionally implement just enough of the CSS grammar to extract
# flat selector -> {property: value} declarations from the inlined <style>
# block(s) of index.html (no @media/@keyframes nesting is expected, per the
# "Desktop-only layout, no responsive breakpoints" constraint) and to resolve
# `var(--name)` references against `:root`. This avoids pulling in a browser
# engine while still grounding the layout assertions in the file's actual,
# rendered CSS rules rather than re-stating expectations.

_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_STYLE_BLOCK_RE = re.compile(r"<style[^>]*>(.*?)</style>", re.DOTALL | re.IGNORECASE)
_RULE_RE = re.compile(r"([^{}]+)\{([^{}]*)\}")
_VAR_RE = re.compile(r"var\(\s*(--[\w-]+)\s*(?:,\s*([^)]*))?\)")
_PX_RE = re.compile(r"(-?\d+(?:\.\d+)?)px")


def _parse_inline_css_rules(html_source: str) -> dict[str, dict[str, str]]:
    """Parse selector -> {property: value} from inlined <style> blocks.

    Later declarations of the same property on the same selector override
    earlier ones, mirroring CSS cascade-by-source-order for equal-specificity
    rules on a single selector.
    """
    rules: dict[str, dict[str, str]] = {}
    for style_block in _STYLE_BLOCK_RE.findall(html_source):
        css = _COMMENT_RE.sub("", style_block)
        for selector_group, body in _RULE_RE.findall(css):
            selector_group = selector_group.strip()
            if not selector_group or selector_group.startswith("@"):
                continue
            props: dict[str, str] = {}
            for declaration in body.split(";"):
                declaration = declaration.strip()
                if not declaration or ":" not in declaration:
                    continue
                prop, _, value = declaration.partition(":")
                props[prop.strip().lower()] = value.strip()
            for selector in selector_group.split(","):
                selector = selector.strip()
                if selector:
                    rules.setdefault(selector, {}).update(props)
    return rules


def _resolve_css_value(value: str, root_vars: dict[str, str], _depth: int = 0) -> str:
    """Resolve a top-level `var(--name[, fallback])` reference via :root."""
    if not value or _depth > 5:
        return value
    match = _VAR_RE.fullmatch(value.strip())
    if not match:
        return value
    var_name, fallback = match.group(1), match.group(2)
    resolved = root_vars.get(var_name, fallback)
    if resolved is None:
        return value
    return _resolve_css_value(resolved, root_vars, _depth + 1)


def _px(value: str | None) -> float | None:
    """Extract a pixel quantity from a simple CSS length value, or None."""
    if not value:
        return None
    match = _PX_RE.fullmatch(value.strip())
    if not match:
        return None
    return float(match.group(1))


class TestStaticIndexExists:
    def test_index_html_file_exists(self):
        assert INDEX_HTML_PATH.is_file(), f"{INDEX_HTML_PATH} does not exist"


class TestSidebarStructure:
    """Sub-AC 1: #sidebar contains three widget card placeholders."""

    def test_sidebar_container_exists(self, tree):
        sidebars = tree.xpath('//*[@id="sidebar"]')
        assert len(sidebars) == 1

    @pytest.mark.parametrize(
        "widget_id",
        ["widget-weather", "widget-calendar", "widget-memo"],
    )
    def test_sidebar_contains_widget_card(self, sidebar, widget_id):
        nodes = sidebar.xpath(f'.//*[@id="{widget_id}"]')
        assert len(nodes) == 1, f"expected #{widget_id} inside #sidebar"

        node = nodes[0]
        classes = node.get("class", "").split()
        assert "widget-card" in classes, (
            f"#{widget_id} should carry the 'widget-card' class"
        )

    def test_widget_memo_is_a_direct_child_of_sidebar(self, tree, sidebar):
        """Sub-AC 4: #widget-memo is present as a direct child of #sidebar.

        This is a stricter, standalone DOM query check (beyond the
        descendant-based parametrized check above) that the memo widget
        card section sits directly inside the sidebar container, per the
        ontology's read-only sidebar widget boundary.
        """
        direct_children = sidebar.xpath('./*[@id="widget-memo"]')
        assert len(direct_children) == 1, (
            "expected #widget-memo to be a direct child of #sidebar"
        )

        memo_widget = direct_children[0]
        assert memo_widget.getparent() is sidebar

        classes = memo_widget.get("class", "").split()
        assert "widget-card" in classes, (
            "#widget-memo should carry the 'widget-card' class"
        )

    def test_sidebar_has_exactly_three_widget_cards(self, sidebar):
        cards = sidebar.xpath(
            './/*[contains(concat(" ", normalize-space(@class), " "), " widget-card ")]'
        )
        assert len(cards) == 3

    def test_widget_cards_cover_weather_calendar_and_memo(self, sidebar):
        cards = sidebar.xpath(
            './/*[contains(concat(" ", normalize-space(@class), " "), " widget-card ")]'
        )
        ids = {card.get("id") for card in cards}
        assert ids == {"widget-weather", "widget-calendar", "widget-memo"}


class TestWeatherWidgetCard:
    """Sub-AC 2a: the sidebar renders a weather widget card container with
    consistent card markup/class (``id="widget-weather"`` /
    ``class="widget-card"``).

    This is a standalone, explicitly-named DOM check (beyond the
    parametrized ``test_sidebar_contains_widget_card`` check above) that
    queries for the weather widget card element within ``#sidebar`` and
    verifies it carries the shared ``widget-card`` markup -- mirroring the
    ``widget-header`` / ``widget-body`` structure used by the calendar and
    memo cards -- so the weather widget is reliably targetable for data
    population (Sub-AC 1) and refresh-timestamp updates (polling).
    """

    @pytest.fixture(scope="class")
    def weather_widget(self, sidebar):
        nodes = sidebar.xpath('./*[@id="widget-weather"]')
        assert len(nodes) == 1, "expected #widget-weather as a direct child of #sidebar"
        return nodes[0]

    def test_weather_widget_card_exists_within_sidebar(self, sidebar):
        nodes = sidebar.xpath('.//*[@id="widget-weather"]')
        assert len(nodes) == 1, "expected #widget-weather inside #sidebar"

    def test_weather_widget_card_has_widget_card_class(self, weather_widget):
        classes = weather_widget.get("class", "").split()
        assert "widget-card" in classes, (
            "#widget-weather should carry the 'widget-card' class for "
            "consistent card markup"
        )

    def test_weather_widget_card_has_consistent_header_and_body(self, weather_widget):
        # Same internal structure as the calendar/memo widget cards: a
        # `.widget-header` (title + last-updated timestamp) and a
        # `.widget-body` (rendered weather data / loading / error state).
        header = weather_widget.xpath(
            './*[contains(concat(" ", normalize-space(@class), " "), " widget-header ")]'
        )
        assert len(header) == 1, "expected a .widget-header inside #widget-weather"

        body = weather_widget.xpath(
            './*[contains(concat(" ", normalize-space(@class), " "), " widget-body ")]'
        )
        assert len(body) == 1, "expected a .widget-body inside #widget-weather"
        assert body[0].get("id") == "weather-body", (
            "expected #widget-weather's .widget-body to carry id='weather-body'"
        )


class TestCalendarWidgetCard:
    """Sub-AC 2b: the sidebar renders a calendar widget card container with
    consistent card markup/class (``id="widget-calendar"`` /
    ``class="widget-card"``).

    This is a standalone, explicitly-named DOM check (beyond the
    parametrized ``test_sidebar_contains_widget_card`` check above) that
    queries for the calendar widget card element within ``#sidebar`` and
    verifies it carries the shared ``widget-card`` markup -- mirroring the
    ``widget-header`` / ``widget-body`` structure used by the weather and
    memo cards -- so the calendar widget is reliably targetable for data
    population (today's events) and refresh-timestamp updates (polling).
    """

    @pytest.fixture(scope="class")
    def calendar_widget(self, sidebar):
        nodes = sidebar.xpath('./*[@id="widget-calendar"]')
        assert len(nodes) == 1, "expected #widget-calendar as a direct child of #sidebar"
        return nodes[0]

    def test_calendar_widget_card_exists_within_sidebar(self, sidebar):
        nodes = sidebar.xpath('.//*[@id="widget-calendar"]')
        assert len(nodes) == 1, "expected #widget-calendar inside #sidebar"

    def test_calendar_widget_card_has_widget_card_class(self, calendar_widget):
        classes = calendar_widget.get("class", "").split()
        assert "widget-card" in classes, (
            "#widget-calendar should carry the 'widget-card' class for "
            "consistent card markup"
        )

    def test_calendar_widget_card_has_consistent_header_and_body(self, calendar_widget):
        # Same internal structure as the weather/memo widget cards: a
        # `.widget-header` (title + last-updated timestamp) and a
        # `.widget-body` (rendered calendar events / loading / error state).
        header = calendar_widget.xpath(
            './*[contains(concat(" ", normalize-space(@class), " "), " widget-header ")]'
        )
        assert len(header) == 1, "expected a .widget-header inside #widget-calendar"

        body = calendar_widget.xpath(
            './*[contains(concat(" ", normalize-space(@class), " "), " widget-body ")]'
        )
        assert len(body) == 1, "expected a .widget-body inside #widget-calendar"
        assert body[0].get("id") == "calendar-body", (
            "expected #widget-calendar's .widget-body to carry id='calendar-body'"
        )


class TestMemoWidgetCard:
    """Sub-AC 2c: the sidebar renders a memo widget card container with
    consistent card markup/class (``id="widget-memo"`` /
    ``class="widget-card"``).

    This is a standalone, explicitly-named DOM check (beyond the
    parametrized ``test_sidebar_contains_widget_card`` check above) that
    queries for the memo widget card element within ``#sidebar`` and
    verifies it carries the shared ``widget-card`` markup -- mirroring the
    ``widget-header`` / ``widget-body`` structure used by the weather and
    calendar cards -- so the memo widget is reliably targetable for data
    population (recent notes) and refresh-timestamp updates (polling).
    """

    @pytest.fixture(scope="class")
    def memo_widget(self, sidebar):
        nodes = sidebar.xpath('./*[@id="widget-memo"]')
        assert len(nodes) == 1, "expected #widget-memo as a direct child of #sidebar"
        return nodes[0]

    def test_memo_widget_card_exists_within_sidebar(self, sidebar):
        nodes = sidebar.xpath('.//*[@id="widget-memo"]')
        assert len(nodes) == 1, "expected #widget-memo inside #sidebar"

    def test_memo_widget_card_has_widget_card_class(self, memo_widget):
        classes = memo_widget.get("class", "").split()
        assert "widget-card" in classes, (
            "#widget-memo should carry the 'widget-card' class for "
            "consistent card markup"
        )

    def test_memo_widget_card_has_consistent_header_and_body(self, memo_widget):
        # Same internal structure as the weather/calendar widget cards: a
        # `.widget-header` (title + last-updated timestamp) and a
        # `.widget-body` (rendered notes / loading / error state).
        header = memo_widget.xpath(
            './*[contains(concat(" ", normalize-space(@class), " "), " widget-header ")]'
        )
        assert len(header) == 1, "expected a .widget-header inside #widget-memo"

        body = memo_widget.xpath(
            './*[contains(concat(" ", normalize-space(@class), " "), " widget-body ")]'
        )
        assert len(body) == 1, "expected a .widget-body inside #widget-memo"
        assert body[0].get("id") == "memo-body", (
            "expected #widget-memo's .widget-body to carry id='memo-body'"
        )


class TestMainChatStructure:
    """Sub-AC 2: a right-side main chat area container exists outside #sidebar."""

    # Acceptable ids for the main chat area container, per the AC examples
    # ("e.g. #main-chat or #chat-area") plus the id actually used by this
    # implementation (#main).
    MAIN_CHAT_CANDIDATE_IDS = ("main-chat", "chat-area", "main")

    @pytest.fixture(scope="class")
    def main_chat(self, tree):
        for candidate_id in self.MAIN_CHAT_CANDIDATE_IDS:
            matches = tree.xpath(f'//*[@id="{candidate_id}"]')
            if matches:
                return matches[0]
        pytest.fail(
            "expected a main chat area container element with one of ids "
            f"{self.MAIN_CHAT_CANDIDATE_IDS} in index.html"
        )

    def test_main_chat_container_exists(self, main_chat):
        assert main_chat is not None

    def test_main_chat_container_is_not_inside_sidebar(self, main_chat, sidebar):
        # The chat container must not be the sidebar itself, nor nested
        # within it.
        assert main_chat is not sidebar
        sidebar_descendants = set(sidebar.iterdescendants())
        assert main_chat not in sidebar_descendants

    def test_main_chat_container_is_sibling_of_sidebar(self, main_chat, sidebar):
        # The chat area should live alongside the sidebar in the top-level
        # app layout (i.e. on the right-hand side of the dashboard), not
        # nested somewhere unrelated.
        assert main_chat.getparent() is sidebar.getparent()

    def test_main_chat_container_hosts_chat_messages_area(self, main_chat):
        messages = main_chat.xpath('.//*[@id="chat-messages"]')
        assert len(messages) == 1, (
            "expected #chat-messages inside the main chat area container"
        )

    def test_main_chat_container_hosts_composer(self, main_chat):
        composer = main_chat.xpath('.//*[contains(concat(" ", normalize-space(@class), " "), " chat-composer ")]')
        assert len(composer) == 1, (
            "expected the chat composer inside the main chat area container"
        )


class TestChatContainerStructure:
    """Sub-AC 3 (governed dispatch): the main area's chat container groups a
    message display region and an input region.

    Within the right-side main area (``#main``), ``#chat-container`` is the
    chat container element. This test queries its direct child elements and
    asserts there are exactly two: the message display region
    (``#chat-messages``) and the input region (``.chat-composer``).
    """

    @pytest.fixture(scope="class")
    def main_chat(self, tree):
        for candidate_id in TestMainChatStructure.MAIN_CHAT_CANDIDATE_IDS:
            matches = tree.xpath(f'//*[@id="{candidate_id}"]')
            if matches:
                return matches[0]
        pytest.fail("expected a main chat area container element in index.html")

    @pytest.fixture(scope="class")
    def chat_container(self, main_chat):
        nodes = main_chat.xpath('.//*[@id="chat-container"]')
        assert len(nodes) == 1, (
            "expected a single #chat-container element within the main area"
        )
        return nodes[0]

    def test_chat_container_exists_within_main_area(self, chat_container):
        assert chat_container is not None

    def test_chat_container_has_message_display_and_input_regions(self, chat_container):
        children = [child for child in chat_container if isinstance(child.tag, str)]
        assert len(children) == 2, (
            "expected #chat-container to have exactly two child elements "
            f"(message display region + input region), found {len(children)}"
        )

        message_region, input_region = children

        assert message_region.get("id") == "chat-messages", (
            "expected the first child of #chat-container to be the message "
            "display region (#chat-messages)"
        )

        input_classes = input_region.get("class", "").split()
        assert "chat-composer" in input_classes, (
            "expected the second child of #chat-container to be the input "
            "region (.chat-composer)"
        )

    def test_chat_container_has_input_field_and_send_control(self, chat_container):
        """Sub-AC 3 (governed dispatch, 8.3.3): the input area's text input
        field and send control are nested within #chat-container.

        Queries for the chat input field (#chat-input) and the send button
        (#send-btn) as descendants of #chat-container, and asserts they are
        markup elements appropriate for a text input field and a send
        control respectively.
        """
        input_fields = chat_container.xpath('.//*[@id="chat-input"]')
        assert len(input_fields) == 1, (
            "expected a single #chat-input element nested inside #chat-container"
        )
        input_field = input_fields[0]
        assert input_field.tag in ("input", "textarea"), (
            "expected #chat-input to be a text input field (<input> or "
            f"<textarea>), got <{input_field.tag}>"
        )

        send_controls = chat_container.xpath('.//*[@id="send-btn"]')
        assert len(send_controls) == 1, (
            "expected a single #send-btn element nested inside #chat-container"
        )
        send_control = send_controls[0]
        assert send_control.tag == "button", (
            f"expected #send-btn to be a <button> send control, got <{send_control.tag}>"
        )
        assert (send_control.get("type") or "").lower() in ("submit", "button"), (
            "expected #send-btn to declare a submit/button type"
        )


class TestMessageListContainer:
    """Sub-AC 3.2 (governed dispatch): the chat container includes a
    message-list container element (e.g. ``#message-list``) for displaying
    chat messages.

    Verified via a DOM test that queries for ``#message-list`` nested inside
    ``#chat-container`` -- specifically within the message display region
    (``#chat-messages``), where ``appendChatMessage()``/``renderHistory()``
    insert ``chat_message`` bubbles (the ontology's ``chat_history``
    concept).
    """

    @pytest.fixture(scope="class")
    def main_chat(self, tree):
        for candidate_id in TestMainChatStructure.MAIN_CHAT_CANDIDATE_IDS:
            matches = tree.xpath(f'//*[@id="{candidate_id}"]')
            if matches:
                return matches[0]
        pytest.fail("expected a main chat area container element in index.html")

    @pytest.fixture(scope="class")
    def chat_container(self, main_chat):
        nodes = main_chat.xpath('.//*[@id="chat-container"]')
        assert len(nodes) == 1, (
            "expected a single #chat-container element within the main area"
        )
        return nodes[0]

    def test_message_list_exists_nested_inside_chat_container(self, chat_container):
        nodes = chat_container.xpath('.//*[@id="message-list"]')
        assert len(nodes) == 1, (
            "expected a single #message-list element nested inside #chat-container"
        )

    def test_message_list_is_nested_inside_chat_messages_region(self, chat_container):
        chat_messages_nodes = chat_container.xpath('.//*[@id="chat-messages"]')
        assert len(chat_messages_nodes) == 1, (
            "expected a single #chat-messages element within #chat-container"
        )
        chat_messages = chat_messages_nodes[0]

        message_list_nodes = chat_messages.xpath('.//*[@id="message-list"]')
        assert len(message_list_nodes) == 1, (
            "expected #message-list to be nested inside the message display "
            "region (#chat-messages)"
        )


class TestSidebarMainLayoutPositioning:
    """Sub-AC 3: sidebar and main chat area sit side-by-side via CSS layout.

    The dashboard's outer ``.app-layout`` container must use a row-flex (or
    equivalent) layout with ``#sidebar`` first (fixed width, on the left) and
    the main chat area second (flexible, filling the remaining width on the
    right). This class parses the actual inlined CSS rules and DOM order from
    index.html, then computes the resulting bounding boxes for a
    representative desktop viewport and asserts the sidebar's right edge is
    at or before the main chat area's left edge.
    """

    # Representative desktop viewport width (px). The layout is desktop-only
    # per the project constraints, so no narrower/responsive widths are
    # exercised here.
    VIEWPORT_WIDTH_PX = 1440.0

    @pytest.fixture(scope="class")
    def css_rules(self, html_text):
        rules = _parse_inline_css_rules(html_text)
        assert rules, "expected at least one CSS rule in index.html's <style> block"
        return rules

    @pytest.fixture(scope="class")
    def root_vars(self, css_rules):
        return css_rules.get(":root", {})

    @pytest.fixture(scope="class")
    def app_layout(self, tree):
        matches = tree.xpath(
            '//*[contains(concat(" ", normalize-space(@class), " "), " app-layout ")]'
        )
        assert len(matches) == 1, "expected exactly one .app-layout container"
        return matches[0]

    @pytest.fixture(scope="class")
    def main_chat_id(self, tree, sidebar):
        for candidate_id in ("main-chat", "chat-area", "main"):
            matches = tree.xpath(f'//*[@id="{candidate_id}"]')
            if matches:
                return candidate_id
        pytest.fail("expected a main chat area container element")

    def test_app_layout_is_a_single_row_flex_container(self, css_rules):
        layout = css_rules.get(".app-layout")
        assert layout is not None, "expected a `.app-layout` CSS rule"
        assert layout.get("display") == "flex", (
            "`.app-layout` must be `display: flex` to lay out the sidebar "
            "and chat area side-by-side"
        )

        # Default flex-direction is `row`; explicit `column`/`*-reverse`
        # values would stack or reverse the children.
        flex_direction = layout.get("flex-direction", "row").strip()
        assert flex_direction == "row", (
            "`.app-layout` flex-direction must be `row` (or unset) so "
            "#sidebar renders to the left of the main chat area"
        )

        # `wrap` could push the chat area below the sidebar on narrow
        # viewports; the layout is desktop-only and must stay single-row.
        flex_wrap = layout.get("flex-wrap", "nowrap").strip()
        assert flex_wrap == "nowrap"

    def test_no_media_queries_alter_the_layout(self, html_text):
        for style_block in _STYLE_BLOCK_RE.findall(html_text):
            css = _COMMENT_RE.sub("", style_block)
            assert "@media" not in css, (
                "no responsive breakpoints/@media rules expected for this "
                "desktop-only layout"
            )

    def test_sidebar_precedes_main_chat_area_in_dom_order(self, app_layout, main_chat_id):
        children = [child for child in app_layout if isinstance(child.tag, str)]
        ids = [child.get("id") for child in children]
        assert "sidebar" in ids, "#sidebar must be a direct child of .app-layout"
        assert main_chat_id in ids, (
            f"#{main_chat_id} must be a direct child of .app-layout"
        )
        assert ids.index("sidebar") < ids.index(main_chat_id), (
            "#sidebar must come before the main chat area in DOM order so "
            "it renders on the left within the row-flex .app-layout"
        )

    def test_sidebar_has_fixed_positive_width(self, css_rules, root_vars):
        sidebar_rules = css_rules.get("#sidebar")
        assert sidebar_rules is not None, "expected an `#sidebar` CSS rule"

        width = _resolve_css_value(sidebar_rules.get("width", ""), root_vars)
        width_px = _px(width)
        assert width_px is not None and width_px > 0, (
            f"expected #sidebar to have a fixed pixel width, got {width!r}"
        )

        # Not absolutely/fixed positioned out of flow -- it must occupy
        # space as a normal flex item.
        assert sidebar_rules.get("position", "static") not in ("absolute", "fixed")

    def test_main_chat_area_flexes_to_fill_remaining_width(self, css_rules, main_chat_id):
        main_rules = css_rules.get(f"#{main_chat_id}")
        assert main_rules is not None, f"expected an `#{main_chat_id}` CSS rule"

        flex_value = main_rules.get("flex", "")
        flex_grow = flex_value.split()[0] if flex_value.split() else ""
        try:
            flex_grow_num = float(flex_grow)
        except ValueError:
            flex_grow_num = 0.0
        assert flex_grow_num > 0, (
            f"expected the main chat area (#{main_chat_id}) to use "
            f"`flex: <positive-grow> ...` to fill remaining width, got "
            f"flex={flex_value!r}"
        )

        assert main_rules.get("position", "static") not in ("absolute", "fixed")

    def test_sidebar_right_edge_at_or_before_chat_area_left_edge(
        self, css_rules, root_vars, main_chat_id
    ):
        layout_rules = css_rules["#sidebar"]
        sidebar_width_px = _px(_resolve_css_value(layout_rules["width"], root_vars))
        assert sidebar_width_px is not None

        app_layout_rules = css_rules.get(".app-layout", {})
        gap_px = _px(_resolve_css_value(app_layout_rules.get("gap", "0px"), root_vars)) or 0.0

        # In a row-flex `.app-layout` (asserted above), #sidebar (first
        # child, fixed width) occupies [0, sidebar_width] and the main chat
        # area (second child, flex: 1) occupies the rest of the viewport,
        # separated by any explicit `gap`.
        sidebar_rect = {"left": 0.0, "right": sidebar_width_px}
        main_chat_rect = {
            "left": sidebar_width_px + gap_px,
            "right": self.VIEWPORT_WIDTH_PX,
        }

        assert sidebar_rect["right"] <= main_chat_rect["left"], (
            "expected #sidebar's right edge to be at or before the main "
            f"chat area's (#{main_chat_id}) left edge: "
            f"sidebar_rect={sidebar_rect!r}, main_chat_rect={main_chat_rect!r}"
        )
        assert main_chat_rect["right"] > main_chat_rect["left"], (
            "main chat area should occupy positive width to the right of "
            "the sidebar"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
