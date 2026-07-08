"""UI element fingerprinting + structural matching primitives.

Adapted port of MobileGPT-V2 Node-Clustering ``utils/xml_parser.py`` onto
Monkey-Collector's *encoded* XML schema. The reference matcher fingerprints on
``tag / id / class / description / text``; Monkey-Collector's encoder
(:func:`~monkey_collector.xml.structured_parser.encode_to_html_xml`) keeps only
``index, bounds`` plus per-tag ``aria-label / alt / type / value / checked /
data-scroll / role`` — it drops resource-id and class. So the fingerprint here
is re-expressed on ``tag / text / aria-label / alt / type / value`` and the
"distinctive self" heuristic (which lets a strong self identity skip the parent
requirement) is re-pointed from ``id/description`` to ``aria-label/alt``.

Interactable tags in encoded XML are ``button`` and ``input`` (a text field is
``input`` with no checkbox type; a checkbox is ``input type="checkbox"``).
Scroll containers are ``div[data-scroll="true"]`` and are handled by the
exploration engine via negative indices, so they are intentionally excluded
from element-set matching here.

These functions operate on the SAME index space as ``SemanticElement.index``
and ``{step}_encoded.xml`` (both flow through ``StructuredXmlParser.parse`` →
``_renumber`` → ``_clear_bounds``), so the indices an :class:`ElementExtractor`
returns align 1:1 with exploration grounding.
"""

from __future__ import annotations

import contextlib
import copy
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

# Encoded interactable tags. Scroll (``div[data-scroll]``) is deliberately
# excluded — the exploration engine owns scrollables via negative indices.
INTERACTABLE_TAGS = ("button", "input")

# Self-fingerprint keys that, when non-NONE, identify a node strongly enough to
# drop the parent requirement (V2's id/description role on this schema).
_DISTINCTIVE_KEYS = ("aria-label", "alt")

# Self attributes captured for an element fingerprint (besides tag/text).
_SELF_ATTR_KEYS = ("aria-label", "alt", "type", "value")
# Parent attributes captured (besides tag).
_PARENT_ATTR_KEYS = ("aria-label",)

_NONE = "NONE"


@dataclass
class UIAttributes:
    """Coordinate-free structural fingerprint of one UI element.

    ``self_attrs`` / ``parent`` are attribute dicts; ``children`` is a list of
    ``[attr_dict, depth, rank]`` triples. Element indices drift between dumps,
    so this fingerprint — not the index — is the matching key. Mirrors
    MobileGPT-V2 ``models.UIAttributes`` (which uses a pydantic ``alias="self"``;
    here :meth:`from_attrib_dict` adapts the ``{"self", "parent", "children"}``
    dict that :func:`get_ui_key_attrib` returns).
    """

    self_attrs: dict = field(default_factory=dict)
    parent: dict = field(default_factory=dict)
    children: list = field(default_factory=list)

    def to_dict(self) -> dict:
        # Hands out the live ``self``/``children`` references; callers that
        # mutate (text-blinding) must deepcopy first (see text_blind_requirements).
        return {"self": self.self_attrs, "parent": self.parent, "children": self.children}

    @classmethod
    def from_attrib_dict(cls, d: dict) -> UIAttributes:
        return cls(
            self_attrs=d.get("self", {}),
            parent=d.get("parent", {}),
            children=d.get("children", []),
        )

# ---------------------------------------------------------------------------
# Tree traversal helpers
# ---------------------------------------------------------------------------


def find_parent_node(root: ET.Element, child_index: int) -> tuple[int, ET.Element | None]:
    """Find the parent of the node carrying ``index == child_index``.

    Returns ``(rank, parent)`` or ``(0, None)`` when not found.
    """
    if isinstance(child_index, str):
        child_index = int(child_index)
    for parent in root.iter():
        for rank, child in enumerate(parent):
            child_idx = child.get("index")
            if child_idx is not None and int(child_idx) == child_index:
                return rank, parent
    return 0, None


def find_children_with_attributes(element: ET.Element, depth: int = 1) -> list[tuple[ET.Element, int, int]]:
    """Recursively collect descendants carrying text or an aria-label/alt, up to depth 3.

    V2 gates on ``text`` or ``description``; on the encoded schema the content
    carriers are ``text`` (node text) and ``aria-label`` / ``alt``.
    """
    valid_children: list[tuple[ET.Element, int, int]] = []
    if depth > 3:
        return valid_children
    for rank, child in enumerate(element, start=0):
        has_text = child.text is not None and child.text.strip() != ""
        has_label = "aria-label" in child.attrib or "alt" in child.attrib
        if has_text or has_label:
            valid_children.append((child, depth, rank))
        valid_children.extend(find_children_with_attributes(child, depth + 1))
    return valid_children


def find_children_by_depth_and_rank(
    element: ET.Element, target_depth: int, target_rank: int, current_depth: int = 1
) -> list[ET.Element]:
    """Return the child at ``(target_depth, target_rank)`` under *element*, if any."""
    matched: list[ET.Element] = []
    if current_depth == target_depth:
        with contextlib.suppress(IndexError):
            matched.append(element[target_rank])
    else:
        for child in element:
            matched.extend(
                find_children_by_depth_and_rank(child, target_depth, target_rank, current_depth + 1)
            )
    return matched


# ---------------------------------------------------------------------------
# Condition matching
# ---------------------------------------------------------------------------


def match_conditions(node: ET.Element, condition: dict) -> bool:
    """True when *node* satisfies every non-NONE field of *condition*.

    Fields whose required value is ``"NONE"`` are skipped (wildcards). ``tag``
    and ``text`` get dedicated handling; all other keys (aria-label/alt/type/
    value) compare against ``node.attrib.get(key, "NONE")``.
    """
    for key, value in condition.items():
        if value == _NONE:
            continue
        if key == "tag":
            if node.tag != value:
                return False
        elif key == "text":
            text = (node.text or "").strip() or _NONE
            if text != value:
                return False
        else:
            if node.attrib.get(key, _NONE) != value:
                return False
    return True


def find_matching_node(tree: ET.Element, requirements: dict) -> list[ET.Element]:
    """Find indexed nodes matching ``{self, parent, children}`` *requirements*.

    The ``parent`` requirement is matched against the whole ancestor chain (not
    only the direct parent) so intermediate wrapper nodes inserted/removed
    between captures of the same screen don't break the match (V2 parity). A
    *distinctive* self (non-NONE aria-label or alt) identifies the node on its
    own and drops the parent requirement; a generic self (tag only) still
    requires an ancestor match.
    """
    matched_nodes: list[ET.Element] = []

    # ElementTree has no parent pointers — build child→parent once.
    parent_of: dict = {}
    for parent in tree.iter():
        for child in parent:
            parent_of[child] = parent

    def ancestor_matches(node: ET.Element, parent_cond: dict) -> bool:
        cur = parent_of.get(node)
        while cur is not None:
            if match_conditions(cur, parent_cond):
                return True
            cur = parent_of.get(cur)
        return False

    def check_node(node: ET.Element) -> ET.Element | None:
        self_cond = requirements.get("self", {})
        if not match_conditions(node, self_cond):
            return None

        distinctive = any(
            self_cond.get(k, _NONE) not in (None, _NONE, "") for k in _DISTINCTIVE_KEYS
        )

        parent_cond = requirements.get("parent", {})
        if not distinctive and parent_of.get(node) is not None and not ancestor_matches(node, parent_cond):
            return None

        children_requirements = requirements.get("children", [])
        if children_requirements:
            matched_children: list[ET.Element] = []
            for child_cond, child_depth, child_rank in children_requirements:
                children = find_children_by_depth_and_rank(node, child_depth, child_rank)
                for child in children:
                    if match_conditions(child, child_cond) and child not in matched_children:
                        matched_children.append(child)
                        break
            if len(matched_children) != len(children_requirements):
                return None

        return node

    for node in tree.iter():
        if node.get("index") is not None:
            result = check_node(node)
            if result is not None:
                matched_nodes.append(result)

    return matched_nodes


def text_blind_requirements(requirements: dict) -> dict:
    """Deep-copy *requirements* with ``self.text`` and every ``children[*].text``
    neutralized to ``"NONE"``.

    Used ONLY at page-classification (step-1) match time. ``match_conditions``
    skips fields whose value == ``"NONE"``, so blinding ``text`` stops dynamic
    text (clocks, counters, "N items") from gating page identity — the exact-
    match brittleness that over-segments pages on revisit. The stored
    fingerprint is left intact; only this per-call copy is blinded. deepcopy is
    mandatory: ``UIAttributes.to_dict()`` hands out the live ``children``
    reference, so in-place mutation would corrupt stored knowledge. V2 parity.
    """
    blinded = copy.deepcopy(requirements)
    self_cond = blinded.get("self")
    if isinstance(self_cond, dict) and "text" in self_cond:
        self_cond["text"] = _NONE
    for child in blinded.get("children", []):
        # child == [attr_dict, depth, rank]
        if child and isinstance(child[0], dict) and "text" in child[0]:
            child[0]["text"] = _NONE
    return blinded


# ---------------------------------------------------------------------------
# Fingerprint extraction
# ---------------------------------------------------------------------------


def get_ui_key_attrib(ui_index: int, encoded_xml: str) -> dict:
    """Build the ``{self, parent, children}`` fingerprint of element ``ui_index``.

    Returns ``{"self": {}, "parent": {}, "children": []}`` when the index is
    absent. ``self`` carries ``tag/text`` plus aria-label/alt/type/value;
    ``parent`` carries ``tag`` + aria-label; ``children`` carries text/label
    bearing descendants as ``[attr_dict, depth, rank]`` triples.
    """
    try:
        tree = ET.fromstring(encoded_xml)
    except ET.ParseError:
        return {"self": {}, "parent": {}, "children": []}

    node = tree.find(f".//*[@index='{ui_index}']")
    if node is None:
        return {"self": {}, "parent": {}, "children": []}

    self_attrs = {
        "tag": node.tag,
        "text": (node.text or "").strip() or _NONE,
    }
    for key in _SELF_ATTR_KEYS:
        self_attrs[key] = node.attrib.get(key, _NONE)

    _, parent_node = find_parent_node(tree, ui_index)
    parent_attrs: dict = {}
    if parent_node is not None:
        parent_attrs = {"tag": parent_node.tag}
        for key in _PARENT_ATTR_KEYS:
            parent_attrs[key] = parent_node.attrib.get(key, _NONE)

    children_attrs_str: list[str] = []
    for child_node, depth, rank in find_children_with_attributes(node):
        child_attr = {
            "tag": child_node.tag,
            "text": (child_node.text or "").strip() or _NONE,
        }
        for key in _SELF_ATTR_KEYS:
            child_attr[key] = child_node.attrib.get(key, _NONE)
        child_attr_str = json.dumps((child_attr, depth, rank))
        if child_attr_str not in children_attrs_str:
            children_attrs_str.append(child_attr_str)
    children_attrs = [json.loads(s) for s in children_attrs_str]

    return {"self": self_attrs, "parent": parent_attrs, "children": children_attrs}


def extract_interactable_indexes(encoded_xml: str) -> list[int]:
    """Sorted indices of interactable elements (``button`` / ``input``) in encoded XML.

    Root-inclusive: ``tree.iter()`` (not ``.//tag``) so a screen that collapses
    to a single interactable at the root is still counted — a real splash/loading
    frame has a ``<div>`` root, but a one-control screen can become a bare
    ``<button>`` root, and missing it would mis-flag the screen as empty.
    """
    try:
        tree = ET.fromstring(encoded_xml)
    except ET.ParseError:
        return []
    indexes: list[int] = []
    for node in tree.iter():
        if node.tag in INTERACTABLE_TAGS:
            index = node.attrib.get("index")
            if index is not None:
                indexes.append(int(index))
    return sorted(set(indexes))

