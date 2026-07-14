"""HTML-like structured XML parser for uiautomator dumps.

5-stage pipeline:
    _reformat  -> Android classes to semantic tags (Button, TextField, etc.)
    _simplify  -> Remove meaningless leaves, collapse single-child wrappers
    _clean     -> Semantic tags to HTML, attribute whitelist, scroll dedup
    _renumber  -> Sequential index reassignment
    pretty_xml -> Indentation
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from loguru import logger

from monkey_collector.xml.parser_base import Parser

_BUTTON_CLASSES = frozenset(
    {
        "FloatingActionButton",
        "ImageButton",
        "MaterialButton",
        "ExtendedFloatingActionButton",
        "Chip",
    }
)

_LAYOUT_CLASSES = frozenset(
    {
        # Core layouts
        "FrameLayout",
        "LinearLayout",
        "RelativeLayout",
        "ViewGroup",
        "ConstraintLayout",
        "unknown",
        # Compat / extended layouts
        "LinearLayoutCompat",
        "GridLayout",
        "GridView",
        "RadioGroup",
        "TableLayout",
        "TableRow",
        "CoordinatorLayout",
        "CardView",
        "AppBarLayout",
        "CollapsingToolbarLayout",
        "Toolbar",
        "NavigationView",
        "DrawerLayout",
        "ViewFlipper",
        "ViewSwitcher",
        "BottomNavigationView",
        "TabLayout",
        "ChipGroup",
        # Scrollable containers (when scrollable=false → div)
        "RecyclerView",
        "ListView",
        "ScrollView",
        "HorizontalScrollView",
        "ViewPager",
        "ViewPager2",
        "NestedScrollView",
        # Misc
        "node",
        "View",
    }
)


class StructuredXmlParser(Parser):
    """Transforms raw uiautomator XML into a simplified HTML-like structure."""

    def __init__(self) -> None:
        super().__init__("structured_xml")
        self.bounds_cache: dict[int, str] = {}
        self.views: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, raw_xml: str) -> str:
        """Run the full 5-stage pipeline.  Returns pretty-printed XML."""
        reformatted = self._reformat(raw_xml)
        simplified = self._simplify(reformatted)
        cleaned = self._clean(simplified)
        renumbered = self._renumber(cleaned)
        self.views = renumbered
        return self.pretty_xml(renumbered)

    # ------------------------------------------------------------------
    # Stage 1 – Reformat
    # ------------------------------------------------------------------

    def _reformat(self, xml_string: str) -> str:
        """Convert Android class names to semantic tags.

        Keeps intermediate tag names (Button, TextField, Image, Scroll,
        Checker) that ``_simplify`` relies on for pruning decisions.
        """
        xml_string = xml_string.replace("$", "_")
        try:
            tree = ET.fromstring(xml_string)
        except ET.ParseError as exc:
            logger.error(f"XML parse error in _reformat: {exc}")
            return ""

        new_tree = self._process_element(tree)
        if new_tree is None:
            return ""
        return ET.tostring(new_tree, encoding="unicode")

    def _process_element(self, element: ET.Element) -> ET.Element | None:
        # --- attribute extraction ---
        attrib_text = {
            "text": "text",
            "description": "content-desc",
            "important": "important",
            "class": "class",
        }
        attrib_bool = {
            "checkable": "checkable",
            "clickable": "clickable",
            "scrollable": "scrollable",
            "long-clickable": "long-clickable",
        }
        attrib_int = {
            "bounds": "bounds",
            "index": "index",
        }

        new_attrs: dict[str, str] = {}
        for key, src in attrib_text.items():
            val = element.attrib.get(src, "")
            if val:
                new_attrs[key] = val
        for key, src in attrib_bool.items():
            val = element.attrib.get(src, "false")
            if val != "false":
                new_attrs[key] = val
        for key, src in attrib_int.items():
            val = element.attrib.get(src)
            if val is not None:
                new_attrs[key] = val

        # Strip package prefix from resource-id → store as "id"
        rid = element.attrib.get("resource-id", "")
        if rid:
            new_attrs["id"] = rid.split("/")[-1]

        # --- tag selection ---
        # Priority order:
        #   1. EditText → input
        #   2. checkable → Checker
        #   3. clickable OR _BUTTON_CLASSES → Button
        #   4. scrollable → Scroll  (before layout so scrollable RecyclerView → Scroll)
        #   5. _LAYOUT_CLASSES → div
        #   6. ImageView → Image
        #   7. TextView → TextField
        #   8. has text → TextField (leaf) / div (parent)
        #   9. else → div  (fallback — no raw class names leak through)
        class_name = element.attrib.get("class", "node") or "node"
        class_short = class_name.split(".")[-1]

        new_element: ET.Element

        if class_short == "EditText":
            new_element = ET.Element("input", new_attrs)
            if len(element) == 0 and "text" in new_element.attrib:
                new_element.text = new_element.attrib.pop("text")

        elif new_attrs.get("checkable") == "true":
            new_attrs["checked"] = element.attrib.get("checked", "false")
            new_attrs.pop("checkable", None)
            new_element = ET.Element("Checker", new_attrs)

        elif new_attrs.get("clickable") == "true" or class_short in _BUTTON_CLASSES:
            new_attrs.pop("clickable", None)
            new_element = ET.Element("Button", new_attrs)

        elif new_attrs.get("scrollable") == "true":
            new_element = ET.Element("Scroll", new_attrs)

        elif class_short in _LAYOUT_CLASSES:
            new_element = ET.Element("div", new_attrs)

        elif class_short == "ImageView":
            new_element = ET.Element("Image", new_attrs)

        elif class_short == "TextView":
            new_element = ET.Element("TextField", new_attrs)
            if len(element) == 0 and "text" in new_element.attrib:
                new_element.text = new_element.attrib.pop("text")

        elif "text" in new_attrs:
            if len(element) == 0:
                new_element = ET.Element("TextField", new_attrs)
                new_element.text = new_element.attrib.pop("text")
            else:
                new_element = ET.Element("div", new_attrs)

        else:
            # Fallback: map unknown classes to div (prevents raw class names in output)
            new_element = ET.Element("div", new_attrs)

        # --- recurse into children ---
        for child in element:
            new_child = self._process_element(child)
            if new_child is not None:
                new_element.append(new_child)

        # --- prune meaningless leaf ---
        if (
            new_element.tag not in ("Button", "Checker")
            and len(element) == 0
            and "description" not in new_element.attrib
            and not new_element.text
        ):
            return None

        return new_element

    # ------------------------------------------------------------------
    # Stage 2 – Simplify
    # ------------------------------------------------------------------

    def _simplify(self, xml_string: str) -> str:
        """Remove meaningless leaves and collapse single-child wrappers.

        Iterates until convergence.
        """
        try:
            root = ET.fromstring(xml_string)
        except ET.ParseError as exc:
            logger.error(f"XML parse error in _simplify: {exc}")
            return xml_string

        def is_meaningless_leaf(elem: ET.Element) -> bool:
            return (
                len(list(elem)) == 0
                and elem.tag not in ("Button", "Checker")
                and "description" not in elem.attrib
                and not elem.text
            )

        def remove_meaningless_leaves(elem: ET.Element) -> None:
            for child in list(elem):
                remove_meaningless_leaves(child)
            for child in list(elem):
                if is_meaningless_leaf(child):
                    elem.remove(child)

        def simplify_wrappers(elem: ET.Element) -> bool:
            changed = False
            while (
                len(list(elem)) == 1
                and all(x not in elem.attrib for x in ("text", "description"))
                and elem.tag not in ("Button", "Checker")
            ):
                child = elem[0]
                elem.tag = child.tag
                elem.attrib = child.attrib
                elem.text = child.text
                elem[:] = child[:]
                changed = True

            for child in list(elem):
                if simplify_wrappers(child):
                    changed = True
            return changed

        while True:
            remove_meaningless_leaves(root)
            if not simplify_wrappers(root):
                break

        return ET.tostring(root, encoding="unicode")

    # ------------------------------------------------------------------
    # Stage 3 – Clean
    # ------------------------------------------------------------------

    def _clean(self, xml_string: str) -> str:
        """Normalize tags to HTML, apply attribute whitelist, deduplicate scrolls."""
        try:
            root = ET.fromstring(xml_string)
        except ET.ParseError as exc:
            logger.error(f"XML parse error in _clean: {exc}")
            return xml_string

        # Remove nodes with empty bounds
        self._remove_nodes_with_empty_bounds(root)

        # 1) Tag rename to standard HTML
        for el in root.iter():
            if el.tag == "TextField":
                el.tag = "p"
            elif el.tag == "Scroll":
                el.tag = "div"
                el.attrib["data-scroll"] = "true"
            elif el.tag == "Image":
                el.tag = "img"
            elif el.tag == "Button":
                el.tag = "button"
            elif el.tag == "Checker":
                el.tag = "input"
                el.attrib["type"] = "checkbox"
            elif el.tag == "View":
                el.tag = "div"

        # 2) Attribute normalization
        for el in root.iter():
            # Remove legacy attributes
            for k in (
                "important",
                "class",
                "scrollable",
                "long-clickable",
                "clickable",
                "checkable",
                "focusable",
                "focused",
                "visible",
            ):
                el.attrib.pop(k, None)

            # description → alt (img) / aria-label (others)
            desc = el.attrib.pop("description", None)
            if desc:
                if el.tag == "img":
                    el.attrib.setdefault("alt", desc)
                else:
                    el.attrib.setdefault("aria-label", desc)

            # Normalize checkbox checked
            if el.tag == "input" and el.attrib.get("type") == "checkbox":
                checked = el.attrib.get("checked")
                if checked is not None:
                    if str(checked).lower() == "true":
                        el.attrib["checked"] = "checked"
                    else:
                        del el.attrib["checked"]

            # Normalize input text/value
            if el.tag == "input" and el.attrib.get("type") != "checkbox":
                el.attrib.setdefault("type", "text")
                if el.text and el.text.strip():
                    el.attrib.setdefault("value", el.text.strip())
                    el.text = None
                text_attr = el.attrib.pop("text", None)
                if text_attr:
                    el.attrib.setdefault("value", text_attr)

            # Normalize button label
            if el.tag == "button":
                text_attr = el.attrib.pop("text", None)
                if text_attr and (el.text is None or not el.text.strip()):
                    el.text = text_attr

        # 3) Attribute whitelist
        allowed_common = {"index", "bounds"}
        allowed_by_tag = {
            "button": {"aria-label"},
            "p": {"aria-label"},
            "div": {"data-scroll", "aria-label"},
            "img": {"alt"},
            "input": {"type", "value", "checked", "aria-label", "role"},
        }

        for el in root.iter():
            allowed = set(allowed_common) | allowed_by_tag.get(el.tag, set())
            for k in list(el.attrib):
                if k not in allowed:
                    del el.attrib[k]

        # 4) Remove duplicate children in scroll containers
        self._remove_scroll_redundancies(root)

        return ET.tostring(root, encoding="unicode")

    @staticmethod
    def _remove_nodes_with_empty_bounds(element: ET.Element) -> None:
        for node in list(element):
            if node.get("bounds") == "[0,0][0,0]":
                element.remove(node)
            else:
                StructuredXmlParser._remove_nodes_with_empty_bounds(node)

    @staticmethod
    def _remove_scroll_redundancies(root: ET.Element) -> None:
        """Deduplicate identical children inside ``div[data-scroll]``."""
        _dynamic_attrs = {"bounds", "index"}

        def _elem_key(elem: ET.Element) -> tuple:
            attrs = tuple(
                sorted((k, v) for k, v in elem.attrib.items() if k not in _dynamic_attrs)
            )
            return (
                elem.tag,
                attrs,
                elem.text or "",
                tuple(
                    (
                        c.tag,
                        tuple(sorted((k, v) for k, v in c.attrib.items() if k not in _dynamic_attrs)),
                        c.text or "",
                    )
                    for c in list(elem)
                ),
            )

        for scroll in root.iter():
            if scroll.attrib.get("data-scroll") != "true":
                continue
            seen: dict[tuple, ET.Element] = {}
            to_remove: list[ET.Element] = []
            for child in list(scroll):
                key = _elem_key(child)
                if key in seen:
                    to_remove.append(child)
                else:
                    seen[key] = child
            for item in to_remove:
                scroll.remove(item)

    # ------------------------------------------------------------------
    # Stage 4 – Renumber
    # ------------------------------------------------------------------

    def _renumber(self, xml_string: str) -> str:
        """Reassign sequential index numbers (pre-order traversal)."""
        try:
            root = ET.fromstring(xml_string)
        except ET.ParseError as exc:
            logger.error(f"XML parse error in _renumber: {exc}")
            return xml_string

        for idx, element in enumerate(root.iter()):
            element.attrib["index"] = str(idx)
        return ET.tostring(root, encoding="unicode")

    # ------------------------------------------------------------------
    # Stage 5 – Pretty print
    # ------------------------------------------------------------------

    @staticmethod
    def pretty_xml(xml_str: str) -> str:
        """Return indented XML string."""
        try:
            root = ET.fromstring(xml_str)
            ET.indent(root, space="  ")
            return ET.tostring(root, encoding="unicode")
        except ET.ParseError:
            return xml_str

    # ------------------------------------------------------------------
    # Bounds management
    # ------------------------------------------------------------------

    def _clear_bounds(self, xml_string: str) -> str:
        """Cache bounds in ``self.bounds_cache`` and strip them from the XML."""
        self.bounds_cache.clear()
        try:
            root = ET.fromstring(xml_string)
        except ET.ParseError as exc:
            logger.error(f"XML parse error in _clear_bounds: {exc}")
            return xml_string

        for element in root.iter():
            bounds = element.get("bounds")
            index = element.get("index")
            if bounds and index is not None:
                self.bounds_cache[int(index)] = bounds
                del element.attrib["bounds"]

        return ET.tostring(root, encoding="unicode")

    # ------------------------------------------------------------------
    # Element lookup (from base)
    # ------------------------------------------------------------------

    def find_element_by_index(self, index: int) -> ET.Element | None:
        if self.views is None:
            return None
        root = ET.fromstring(self.views)
        for element in root.iter():
            if element.get("index") == str(index):
                return element
        return None

    def get_bounds(self, index: int) -> str | None:
        return self.bounds_cache.get(index)


# ------------------------------------------------------------------
# Module-level convenience functions (replaces xml_encoder.py API)
# ------------------------------------------------------------------


def parse_to_html_xml(raw_xml: str) -> str:
    """Full parse pipeline — bounds kept."""
    parser = StructuredXmlParser()
    result = parser.parse(raw_xml)
    return result


def encode_to_html_xml(raw_xml: str) -> str:
    """Parse and strip bounds.  Returns encoded XML for LLM consumption."""
    parser = StructuredXmlParser()
    parsed = parser.parse(raw_xml)
    if not parsed:
        return ""
    encoded = parser._clear_bounds(parser.views)
    # Re-deduplicate after bounds removal (keys change)
    return encoded


def encode_with_bounds(raw_xml: str) -> tuple[str, dict[int, str]]:
    """Encode for LLM consumption *and* return an ``index -> bounds`` map.

    Same encoded XML as :func:`encode_to_html_xml` (bounds stripped, ``index``
    kept), plus the bounds that were stripped — keyed by the encoded ``index`` —
    so callers can recover on-screen coordinates that line up 1:1 with the
    encoded ``index`` values.
    """
    parser = StructuredXmlParser()
    if not parser.parse(raw_xml) or parser.views is None:
        return "", {}
    encoded = parser._clear_bounds(parser.views)
    return encoded, dict(parser.bounds_cache)


def hierarchy_parse(raw_xml: str) -> str:
    """Parse, then strip text/bounds/index — structure only."""
    parser = StructuredXmlParser()
    parsed = parser.parse(raw_xml)
    if not parsed:
        return ""

    try:
        root = ET.fromstring(parser.views)
    except ET.ParseError:
        return ""

    for el in root.iter():
        el.attrib.pop("bounds", None)
        el.attrib.pop("index", None)
        el.text = None
        # Keep: tag, id/alt/aria-label, data-scroll, type, checked
        for k in list(el.attrib):
            if k in ("bounds", "index"):
                del el.attrib[k]

    return ET.tostring(root, encoding="unicode")


def indent_xml(xml_string: str, indent: str = "  ") -> str:
    """Pretty-print XML with indentation."""
    try:
        root = ET.fromstring(xml_string)
        ET.indent(root, space=indent)
        return ET.tostring(root, encoding="unicode")
    except ET.ParseError:
        return xml_string
