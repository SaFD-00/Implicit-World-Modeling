"""Tests for monkey_collector.parser.structured_parser — 5-stage XML pipeline."""

import xml.etree.ElementTree as ET

from monkey_collector.xml.structured_parser import (
    StructuredXmlParser,
    encode_to_html_xml,
    hierarchy_parse,
    indent_xml,
    parse_to_html_xml,
)
from tests.fixtures.xml_samples import COMPLEX_XML, SIMPLE_XML

# ── Helpers ──


def _make_node(class_name, bounds="[10,10][200,200]", **attrs):
    defaults = {
        "text": "",
        "resource-id": "",
        "content-desc": "",
        "checkable": "false",
        "checked": "false",
        "clickable": "false",
        "enabled": "true",
        "scrollable": "false",
        "long-clickable": "false",
        "password": "false",
        "selected": "false",
        "important": "false",
        "index": "0",
    }
    defaults.update(attrs)
    defaults["class"] = class_name
    defaults["bounds"] = bounds
    attr_str = " ".join(f'{k}="{v}"' for k, v in defaults.items())
    return f"<node {attr_str} />"


def _wrap(*nodes):
    inner = "\n".join(nodes)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<hierarchy rotation="0">\n{inner}\n</hierarchy>'
    )


# ── Stage 1: Reformat ──


class TestReformat:
    def _child(self, xml_str):
        p = StructuredXmlParser()
        result = p._reformat(xml_str)
        root = ET.fromstring(result)
        children = list(root)
        return children[0] if children else root

    def test_edittext_to_input(self):
        xml = _wrap(_make_node("android.widget.EditText", text="hello"))
        child = self._child(xml)
        assert child.tag == "input"
        assert child.text == "hello"

    def test_checkable_to_checker(self):
        xml = _wrap(
            _make_node(
                "android.widget.Switch", checkable="true", checked="true"
            )
        )
        child = self._child(xml)
        assert child.tag == "Checker"
        assert child.attrib.get("checked") == "true"

    def test_clickable_to_button(self):
        xml = _wrap(
            _make_node(
                "android.widget.ImageButton",
                clickable="true",
                **{"content-desc": "Go"},
            )
        )
        child = self._child(xml)
        assert child.tag == "Button"
        assert "clickable" not in child.attrib

    def test_layout_to_div(self):
        xml = _wrap(
            f'<node class="android.widget.FrameLayout" bounds="[0,0][1080,1920]" '
            f'index="0" text="" resource-id="" content-desc="" checkable="false" '
            f'checked="false" clickable="false" enabled="true" scrollable="false" '
            f'long-clickable="false" password="false" selected="false" important="false">'
            f'{_make_node("android.widget.TextView", text="child")}'
            f"</node>"
        )
        child = self._child(xml)
        assert child.tag == "div"

    def test_imageview_to_image(self):
        xml = _wrap(
            _make_node("android.widget.ImageView", **{"content-desc": "photo"})
        )
        child = self._child(xml)
        assert child.tag == "Image"

    def test_textview_to_textfield(self):
        xml = _wrap(_make_node("android.widget.TextView", text="Hello"))
        child = self._child(xml)
        assert child.tag == "TextField"
        assert child.text == "Hello"

    def test_scrollable_to_scroll(self):
        xml = _wrap(
            f'<node class="android.widget.ScrollView" bounds="[0,0][1080,1920]" '
            f'index="0" text="" resource-id="" content-desc="" checkable="false" '
            f'checked="false" clickable="false" enabled="true" scrollable="true" '
            f'long-clickable="false" password="false" selected="false" important="false">'
            f'{_make_node("android.widget.TextView", text="item")}'
            f"</node>"
        )
        child = self._child(xml)
        assert child.tag == "Scroll"

    def test_resource_id_stripped(self):
        xml = _wrap(
            _make_node(
                "android.widget.ImageButton",
                clickable="true",
                **{"resource-id": "com.app:id/my_fab", "content-desc": "fab"},
            )
        )
        child = self._child(xml)
        assert child.attrib.get("id") == "my_fab"

    def test_invalid_xml(self):
        p = StructuredXmlParser()
        assert p._reformat("<not valid!!!") == ""

    def test_prunes_meaningless_leaf(self):
        """Leaf with no text, no description, not Button/Checker → pruned."""
        xml = _wrap(_make_node("android.view.View"))
        p = StructuredXmlParser()
        result = p._reformat(xml)
        root = ET.fromstring(result)
        assert len(list(root)) == 0

    def test_keeps_button_leaf(self):
        """Button leaf without text should be kept."""
        xml = _wrap(
            _make_node(
                "android.widget.ImageButton",
                clickable="true",
                **{"content-desc": ""},
            )
        )
        child = self._child(xml)
        assert child.tag == "Button"

    def test_linear_layout_compat_to_div(self):
        """LinearLayoutCompat → div."""
        xml = _wrap(
            f'<node class="androidx.appcompat.widget.LinearLayoutCompat" '
            f'bounds="[0,0][1080,200]" index="0" text="" resource-id="" '
            f'content-desc="" checkable="false" checked="false" clickable="false" '
            f'enabled="true" scrollable="false" long-clickable="false" '
            f'password="false" selected="false" important="false">'
            f'{_make_node("android.widget.TextView", text="item")}'
            f"</node>"
        )
        child = self._child(xml)
        assert child.tag == "div"

    def test_recycler_view_scrollable_to_scroll(self):
        """RecyclerView with scrollable=true → Scroll."""
        xml = _wrap(
            f'<node class="androidx.recyclerview.widget.RecyclerView" '
            f'bounds="[0,0][1080,1920]" index="0" text="" resource-id="" '
            f'content-desc="" checkable="false" checked="false" clickable="false" '
            f'enabled="true" scrollable="true" long-clickable="false" '
            f'password="false" selected="false" important="false">'
            f'{_make_node("android.widget.TextView", text="item")}'
            f"</node>"
        )
        child = self._child(xml)
        assert child.tag == "Scroll"

    def test_recycler_view_not_scrollable_to_div(self):
        """RecyclerView with scrollable=false → div."""
        xml = _wrap(
            f'<node class="androidx.recyclerview.widget.RecyclerView" '
            f'bounds="[0,0][1080,1920]" index="0" text="" resource-id="" '
            f'content-desc="" checkable="false" checked="false" clickable="false" '
            f'enabled="true" scrollable="false" long-clickable="false" '
            f'password="false" selected="false" important="false">'
            f'{_make_node("android.widget.TextView", text="item")}'
            f"</node>"
        )
        child = self._child(xml)
        assert child.tag == "div"

    def test_floating_action_button_to_button(self):
        """FloatingActionButton → Button (even without clickable=true)."""
        xml = _wrap(
            _make_node(
                "com.google.android.material.floatingactionbutton.FloatingActionButton",
                clickable="false",
                **{"content-desc": "Add"},
            )
        )
        child = self._child(xml)
        assert child.tag == "Button"

    def test_unknown_class_fallback_to_div(self):
        """Unrecognized class → div (not raw class name)."""
        xml = _wrap(
            f'<node class="com.custom.widget.FancyView" '
            f'bounds="[0,0][100,100]" index="0" text="" resource-id="" '
            f'content-desc="" checkable="false" checked="false" clickable="false" '
            f'enabled="true" scrollable="false" long-clickable="false" '
            f'password="false" selected="false" important="false">'
            f'{_make_node("android.widget.TextView", text="child")}'
            f"</node>"
        )
        child = self._child(xml)
        assert child.tag == "div"


# ── Stage 2: Simplify ──


class TestSimplify:
    def test_collapses_single_child(self):
        xml = '<div bounds="[0,0][1080,1920]"><div bounds="[0,0][1080,1920]"><TextField>Hello</TextField></div></div>'
        p = StructuredXmlParser()
        result = p._simplify(xml)
        root = ET.fromstring(result)
        assert root.tag == "TextField"
        assert root.text == "Hello"

    def test_preserves_button(self):
        xml = "<Button><TextField>Click</TextField></Button>"
        p = StructuredXmlParser()
        result = p._simplify(xml)
        root = ET.fromstring(result)
        assert root.tag == "Button"

    def test_preserves_text_attr(self):
        xml = '<div text="keep"><TextField>child</TextField></div>'
        p = StructuredXmlParser()
        result = p._simplify(xml)
        root = ET.fromstring(result)
        assert root.tag == "div"
        assert root.attrib["text"] == "keep"

    def test_iterative_convergence(self):
        """Multiple rounds of simplification needed."""
        xml = "<div><div><div><TextField>deep</TextField></div></div></div>"
        p = StructuredXmlParser()
        result = p._simplify(xml)
        root = ET.fromstring(result)
        assert root.tag == "TextField"
        assert root.text == "deep"


# ── Stage 3: Clean ──


class TestClean:
    def test_tag_normalization(self):
        xml = '<TextField index="0" bounds="[0,0][100,100]">Hi</TextField>'
        p = StructuredXmlParser()
        result = p._clean(xml)
        root = ET.fromstring(result)
        assert root.tag == "p"

    def test_scroll_to_div_data_scroll(self):
        xml = '<Scroll index="0" bounds="[0,0][100,100]"><TextField index="1" bounds="[0,0][50,50]">a</TextField></Scroll>'
        p = StructuredXmlParser()
        result = p._clean(xml)
        root = ET.fromstring(result)
        assert root.tag == "div"
        assert root.attrib.get("data-scroll") == "true"

    def test_image_to_img(self):
        xml = '<Image index="0" bounds="[0,0][100,100]" description="photo" />'
        p = StructuredXmlParser()
        result = p._clean(xml)
        root = ET.fromstring(result)
        assert root.tag == "img"
        assert root.attrib.get("alt") == "photo"
        assert "description" not in root.attrib

    def test_checker_to_input_checkbox(self):
        xml = '<Checker index="0" bounds="[0,0][100,100]" checked="true" />'
        p = StructuredXmlParser()
        result = p._clean(xml)
        root = ET.fromstring(result)
        assert root.tag == "input"
        assert root.attrib.get("type") == "checkbox"
        assert root.attrib.get("checked") == "checked"

    def test_removes_empty_bounds(self):
        xml = '<div index="0" bounds="[0,0][100,100]"><p index="1" bounds="[0,0][0,0]">x</p><p index="2" bounds="[10,10][100,100]">y</p></div>'
        p = StructuredXmlParser()
        result = p._clean(xml)
        root = ET.fromstring(result)
        children = list(root)
        assert len(children) == 1

    def test_attribute_whitelist(self):
        xml = '<Button index="0" bounds="[0,0][100,100]" class="foo" important="true" description="ok">Go</Button>'
        p = StructuredXmlParser()
        result = p._clean(xml)
        root = ET.fromstring(result)
        assert "class" not in root.attrib
        assert "important" not in root.attrib
        assert root.attrib.get("aria-label") == "ok"

    def test_scroll_dedup(self):
        xml = (
            '<div index="0" bounds="[0,0][100,100]">'
            '<div data-scroll="true" index="1" bounds="[0,0][100,100]">'
            '<p index="2" bounds="[0,0][50,50]">a</p>'
            '<p index="3" bounds="[0,50][50,100]">a</p>'
            '<p index="4" bounds="[50,0][100,50]">b</p>'
            "</div></div>"
        )
        p = StructuredXmlParser()
        result = p._clean(xml)
        root = ET.fromstring(result)
        scroll_div = root.find(".//*[@data-scroll]")
        # Two <p> with same text "a" should be deduped to 1, plus "b" = 2
        children = list(scroll_div)
        assert len(children) == 2


# ── Stage 4: Renumber ──


class TestRenumber:
    def test_sequential_indices(self):
        xml = '<div index="5"><p index="10">Hello</p><p index="20">World</p></div>'
        p = StructuredXmlParser()
        result = p._renumber(xml)
        root = ET.fromstring(result)
        indices = [int(el.attrib["index"]) for el in root.iter()]
        assert indices == [0, 1, 2]


# ── Bounds management ──


class TestClearBounds:
    def test_caches_and_strips(self):
        xml = '<div index="0" bounds="[0,0][100,100]"><p index="1" bounds="[10,10][50,50]">Hi</p></div>'
        p = StructuredXmlParser()
        result = p._clear_bounds(xml)
        root = ET.fromstring(result)

        for el in root.iter():
            assert "bounds" not in el.attrib

        assert p.bounds_cache[0] == "[0,0][100,100]"
        assert p.bounds_cache[1] == "[10,10][50,50]"

    def test_get_bounds(self):
        xml = '<div index="0" bounds="[0,0][100,100]" />'
        p = StructuredXmlParser()
        p._clear_bounds(xml)
        assert p.get_bounds(0) == "[0,0][100,100]"
        assert p.get_bounds(99) is None


# ── Full pipeline ──


class TestFullPipeline:
    def test_parse_simple_xml(self):
        p = StructuredXmlParser()
        result = p.parse(SIMPLE_XML)
        assert result != ""
        root = ET.fromstring(result)
        assert root is not None

    def test_parse_complex_xml(self):
        p = StructuredXmlParser()
        result = p.parse(COMPLEX_XML)
        assert result != ""
        root = ET.fromstring(result)
        assert root is not None

    def test_parse_invalid(self):
        p = StructuredXmlParser()
        result = p.parse("<bad xml!!!")
        assert result == ""

    def test_views_stored(self):
        p = StructuredXmlParser()
        p.parse(SIMPLE_XML)
        assert p.views is not None


# ── Convenience functions ──


class TestConvenienceFunctions:
    def test_parse_to_html_xml(self):
        result = parse_to_html_xml(SIMPLE_XML)
        assert result != ""
        root = ET.fromstring(result)
        assert root is not None

    def test_encode_strips_bounds(self):
        result = encode_to_html_xml(SIMPLE_XML)
        assert result != ""
        root = ET.fromstring(result)
        for el in root.iter():
            assert "bounds" not in el.attrib

    def test_encode_empty(self):
        assert encode_to_html_xml("") == ""

    def test_hierarchy_parse_structure_only(self):
        result = hierarchy_parse(SIMPLE_XML)
        assert result != ""
        root = ET.fromstring(result)
        for el in root.iter():
            assert "bounds" not in el.attrib
            assert "index" not in el.attrib
            assert el.text is None

    def test_hierarchy_parse_empty(self):
        assert hierarchy_parse("") == ""

    def test_indent_xml(self):
        result = indent_xml("<div><p>Hello</p></div>")
        assert "\n" in result
        assert "  " in result

    def test_indent_invalid(self):
        bad = "<not valid!!!"
        assert indent_xml(bad) == bad
