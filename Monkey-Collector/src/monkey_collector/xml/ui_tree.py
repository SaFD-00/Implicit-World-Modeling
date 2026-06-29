"""Parse uiautomator XML dump into structured UIElement objects."""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from loguru import logger


@dataclass
class UIElement:
    """Represents a single UI element from accessibility tree."""

    index: int
    resource_id: str
    class_name: str
    text: str
    content_desc: str
    bounds: tuple[int, int, int, int]  # left, top, right, bottom
    clickable: bool
    scrollable: bool
    enabled: bool
    checkable: bool
    checked: bool
    long_clickable: bool
    password: bool
    selected: bool
    package: str
    visible: bool
    important: bool

    @property
    def area(self) -> int:
        w = self.bounds[2] - self.bounds[0]
        h = self.bounds[3] - self.bounds[1]
        return max(0, w * h)

    @property
    def center(self) -> tuple[int, int]:
        return (
            (self.bounds[0] + self.bounds[2]) // 2,
            (self.bounds[1] + self.bounds[3]) // 2,
        )

    @property
    def short_class(self) -> str:
        return self.class_name.split(".")[-1] if self.class_name else ""

    @property
    def display_name(self) -> str:
        """Best human-readable name for this element."""
        if self.content_desc:
            return self.content_desc
        if self.text:
            return self.text
        if self.resource_id:
            return self.resource_id.split("/")[-1].replace("_", " ")
        return self.short_class


BOUNDS_PATTERN = re.compile(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]")


def parse_bounds(bounds_str: str) -> tuple[int, int, int, int]:
    """Parse bounds string '[left,top][right,bottom]' to tuple."""
    m = BOUNDS_PATTERN.match(bounds_str)
    if not m:
        return (0, 0, 0, 0)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))


def _parse_bool(val: str) -> bool:
    return val.lower() == "true"


EDITABLE_CLASSES = {"EditText", "AutoCompleteTextView", "MultiAutoCompleteTextView",
                     "ExtractEditText", "SearchAutoComplete"}


class UITree:
    """Wrapper around a list of UIElements providing query methods for the explorer."""

    def __init__(self, elements: list[UIElement]):
        self.elements = elements

    def __len__(self) -> int:
        return len(self.elements)

    def __iter__(self):
        return iter(self.elements)

    def get_clickable_elements(self) -> list[UIElement]:
        """Return all clickable, visible, enabled elements."""
        return [e for e in self.elements if e.clickable and e.enabled and e.area > 0]

    def get_editable_elements(self) -> list[UIElement]:
        """Return all editable (EditText-like) elements."""
        return [
            e for e in self.elements
            if e.enabled and any(ec in e.class_name for ec in EDITABLE_CLASSES)
        ]

    def get_scrollable_elements(self) -> list[UIElement]:
        """Return all scrollable elements."""
        return [e for e in self.elements if e.scrollable]

    def get_interactable_elements(self) -> list[UIElement]:
        """Return elements that can be interacted with."""
        return [
            e for e in self.elements
            if e.enabled and e.area > 0
            and (e.clickable or e.scrollable or e.long_clickable or e.checkable
                 or any(ec in e.class_name for ec in EDITABLE_CLASSES))
        ]

    @staticmethod
    def from_xml_string(xml_str: str) -> "UITree":
        """Parse a uiautomator XML string and return a UITree."""
        elements = parse_uiautomator_xml(xml_str)
        return UITree(elements)


def parse_uiautomator_xml(xml_content: str) -> list[UIElement]:
    """Parse a uiautomator XML dump into a flat list of UIElements."""
    elements: list[UIElement] = []
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        logger.error(f"Failed to parse uiautomator XML: {e}")
        return elements

    idx_counter = [0]

    def walk(node: ET.Element) -> None:
        if node.tag == "node":
            bounds = parse_bounds(node.attrib.get("bounds", "[0,0][0,0]"))
            elem = UIElement(
                index=idx_counter[0],
                resource_id=node.attrib.get("resource-id", ""),
                class_name=node.attrib.get("class", ""),
                text=node.attrib.get("text", ""),
                content_desc=node.attrib.get("content-desc", ""),
                bounds=bounds,
                clickable=_parse_bool(node.attrib.get("clickable", "false")),
                scrollable=_parse_bool(node.attrib.get("scrollable", "false")),
                enabled=_parse_bool(node.attrib.get("enabled", "true")),
                checkable=_parse_bool(node.attrib.get("checkable", "false")),
                checked=_parse_bool(node.attrib.get("checked", "false")),
                long_clickable=_parse_bool(
                    node.attrib.get("long-clickable", "false")
                ),
                password=_parse_bool(node.attrib.get("password", "false")),
                selected=_parse_bool(node.attrib.get("selected", "false")),
                package=node.attrib.get("package", ""),
                visible=_parse_bool(
                    node.attrib.get("visible-to-user", "true")
                ),
                important=_parse_bool(node.attrib.get("important", "false")),
            )
            idx_counter[0] += 1
            if bounds != (0, 0, 0, 0) and elem.visible:
                elements.append(elem)

        for child in node:
            walk(child)

    walk(root)
    logger.debug(f"Parsed {len(elements)} visible UI elements from XML")
    return elements
