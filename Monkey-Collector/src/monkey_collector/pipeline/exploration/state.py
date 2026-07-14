"""Semantic screen state: a device-independent abstraction of one screen.

Ports LLM-Explorer's ``DeviceState`` representation onto Monkey-Collector's
existing XML utilities instead of re-deriving a view hierarchy:

- ``state_str``     content-aware identity (text included) — like the reference's
                    ``text_representation``; two screens differing only in text
                    get different ``state_str``.
- ``structure_str`` content-free identity (layout only) — like the reference's
                    ``text_representation_frame``; the abstract-page key.
- ``elements``      interactable elements keyed by **encoded XML index**.

Coordinates for each element are recovered from the ``index -> bounds`` map
returned alongside the encoded XML, so an element's ``center`` is always
consistent with its encoded ``index``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from xml.etree import ElementTree as ET

from monkey_collector.domain.page_graph import compute_xml_fingerprint
from monkey_collector.xml.structured_parser import encode_with_bounds
from monkey_collector.xml.ui_tree import UITree, parse_bounds

# Action types in the engine's vocabulary (decoupled from domain.Action; the
# ActionMapper translates these into concrete Tap/Swipe/InputText/LongPress).
TOUCH = "touch"
LONG_TOUCH = "long_touch"
SET_TEXT = "set_text"
SELECT = "select"
SCROLL = "scroll"


@dataclass(frozen=True)
class SemanticElement:
    """One interactable element of a screen, identified independently of layout.

    ``index`` is the encoded-XML index.
    ``signature`` is a coordinate-free identity used for coverage tracking and
    navigation re-matching, so it survives scrolling and re-rendering.
    """

    index: int
    signature: str
    allowed_actions: tuple[str, ...]
    center: tuple[int, int]
    desc: str


@dataclass(frozen=True)
class SemanticState:
    """An abstract screen: identity hashes plus its interactable elements.

    ``page_key`` is the abstract-page identity used for coverage / navigation.
    With a live :class:`~monkey_collector.pipeline.screen_matching.screen_matcher.ScreenMatcher`
    it is the element-set page key; without one it falls back to
    ``structure_str`` (the text-free structural digest), preserving the legacy
    behaviour byte-for-byte.
    """

    state_str: str
    structure_str: str
    activity: str
    package: str
    elements: tuple[SemanticElement, ...]
    page_key: str = ""

    @classmethod
    def from_screen(
        cls, raw_xml: str, activity: str, package: str, page_key: str = ""
    ) -> SemanticState:
        """Build a SemanticState from raw uiautomator XML and screen context."""
        activity = activity or ""
        package = package or ""
        encoded_xml, bounds_by_index = encode_with_bounds(raw_xml)
        elements = (
            *_extract_elements(encoded_xml, bounds_by_index),
            *_extract_scrollables(raw_xml),
        )
        state_str = _digest(activity, encoded_xml)
        structure_str = _digest(activity, compute_xml_fingerprint(raw_xml))
        return cls(
            state_str=state_str,
            structure_str=structure_str,
            activity=activity,
            package=package,
            elements=elements,
            page_key=page_key or structure_str,
        )

    def is_in_app(self) -> bool:
        """True when the foreground activity belongs to the target package."""
        return bool(self.package) and self.activity.startswith(self.package)

    def actionable_elements(self) -> list[SemanticElement]:
        """Elements that support at least one action."""
        return [e for e in self.elements if e.allowed_actions]

    def find_by_signature(self, signature: str) -> SemanticElement | None:
        """Locate an element by its coordinate-free signature, if present."""
        for element in self.elements:
            if element.signature == signature:
                return element
        return None


# ---------------------------------------------------------------------------
# Element extraction (encoded XML → SemanticElement)
# ---------------------------------------------------------------------------


def _extract_elements(
    encoded_xml: str,
    bounds_by_index: dict[int, str],
) -> tuple[SemanticElement, ...]:
    if not encoded_xml:
        return ()
    try:
        root = ET.fromstring(encoded_xml)
    except ET.ParseError:
        return ()

    elements: list[SemanticElement] = []
    for node in root.iter():
        index_attr = node.attrib.get("index")
        if index_attr is None:
            continue
        actions = _allowed_actions(node)
        if not actions:
            continue
        index = int(index_attr)
        elements.append(
            SemanticElement(
                index=index,
                signature=_signature(node, index),
                allowed_actions=actions,
                center=_center(bounds_by_index.get(index, "")),
                desc=_describe(node),
            )
        )
    return tuple(elements)


def _extract_scrollables(raw_xml: str) -> tuple[SemanticElement, ...]:
    """Recover scroll containers directly from the UI tree.

    The encoder collapses single-child scroll containers, dropping the
    ``data-scroll`` marker, so scrollables are taken from :class:`UITree`
    instead. A screen has only a handful, so they use **negative indices**
    to stay clear of the encoded index space.
    """
    tree = UITree.from_xml_string(raw_xml)
    scrollables: list[SemanticElement] = []
    for offset, element in enumerate(tree.get_scrollable_elements()):
        scrollables.append(
            SemanticElement(
                index=-(offset + 1),
                signature=f"scroll:{element.resource_id}:{element.short_class}",
                allowed_actions=(SCROLL,),
                center=element.center,
                desc=f"<scroll>{element.display_name}</scroll>",
            )
        )
    return tuple(scrollables)


def _allowed_actions(node: ET.Element) -> tuple[str, ...]:
    """Map an encoded HTML-like tag to the actions it supports.

    The encoder collapses Android classes to semantic tags, so action support is
    inferred from the tag: ``button`` is tappable (and long-pressable), an
    ``input`` is a checkbox toggle or a text field, a scroll container swipes.
    Text/image/plain nodes carry no action.
    """
    tag = node.tag
    if tag == "button":
        return (TOUCH, LONG_TOUCH)
    if tag == "input":
        if node.attrib.get("type") == "checkbox":
            return (SELECT,)
        return (SET_TEXT,)
    if tag == "div" and node.attrib.get("data-scroll") == "true":
        return (SCROLL,)
    return ()


def _signature(node: ET.Element, index: int) -> str:
    """Coordinate-free identity: tag + type + best available label.

    Prefers human-stable content (button/text label, aria-label, alt, input
    value) so the same logical element re-matches across scrolls and re-renders.
    Falls back to the encoded index only when no label is available.
    """
    tag = node.tag
    type_ = node.attrib.get("type", "")
    label = (node.text or "").strip()
    key = (
        label
        or node.attrib.get("aria-label", "")
        or node.attrib.get("alt", "")
        or node.attrib.get("value", "")
    )
    if key:
        return f"{tag}:{type_}:{key}"
    return f"{tag}:{type_}:@{index}"


def _describe(node: ET.Element) -> str:
    """A short single-line HTML-ish description for logs and event records."""
    label = (node.text or "").strip()
    aria = node.attrib.get("aria-label") or node.attrib.get("alt") or ""
    value = node.attrib.get("value", "")
    inner = label or aria or value or ""
    return f"<{node.tag}>{inner}</{node.tag}>"


def _center(bounds_str: str) -> tuple[int, int]:
    left, top, right, bottom = parse_bounds(bounds_str)
    return ((left + right) // 2, (top + bottom) // 2)


def _digest(activity: str, payload: str) -> str:
    return hashlib.md5(f"{activity}|{payload}".encode()).hexdigest()
