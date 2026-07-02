"""Encoded-XML → element-line document for BM25 page matching (Mobile3M port).

Ports the reference ``xml_to_html.any_tree_to_html`` "one HTML line per element"
serialization (Mobile3M ``corpus/``) onto Monkey-Collector's encoded schema.
Each screen becomes a flat list of normalized element-lines; the SET of those
lines is what the page matcher compares (BM25 retrieval + element-diff /
Jaccard), mirroring the reference ``compare_actions`` which does
``set(html.strip().split("\\n"))`` on both sides.

Normalization is coordinate-free on purpose: ``index`` and ``bounds`` are
dropped because they drift between two dumps of the same screen (re-numbering,
sub-pixel layout), which would make every line unique per dump and defeat set
overlap. The remaining identity attributes are emitted in a FIXED order so a
line is a stable, comparable string.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

# Identity attributes kept on an element-line, in a FIXED emission order so the
# serialized string is deterministic. The canonical minimal set is
# tag + text + aria-label + alt + type + value; checked/role/data-scroll are
# stable, already-whitelisted (``structured_parser._clean``) identity attributes
# that add discriminativeness. Drop the last three for the strict paper set.
_ATTR_ORDER = ("aria-label", "alt", "type", "value", "checked", "role", "data-scroll")


def serialize_element_lines(encoded_xml: str) -> list[str]:
    """Encoded XML → normalized element-line document (document order, dups kept).

    Emits one line for every node that is a LEAF (no element children) OR a
    scroll container (``data-scroll="true"``) — mirroring ``any_tree_to_html``'s
    "leaf-visible + scrollable" emission rule. ``index``/``bounds`` are dropped;
    the attributes in :data:`_ATTR_ORDER` (when non-empty) plus the node text
    form the line. Duplicates are preserved so BM25 term frequency is faithful.

    Returns ``[]`` on parse failure (the matcher must never break on a bad dump).
    """
    try:
        root = ET.fromstring(encoded_xml)
    except ET.ParseError:
        return []

    lines: list[str] = []
    for node in root.iter():
        is_leaf = len(node) == 0
        is_scroll = node.attrib.get("data-scroll") == "true"
        if not (is_leaf or is_scroll):
            continue
        attrs = "".join(
            f' {key}="{node.attrib[key]}"'
            for key in _ATTR_ORDER
            if node.attrib.get(key)
        )
        text = (node.text or "").replace("\n", "").strip()
        lines.append(f"<{node.tag}{attrs}>{text}</{node.tag}>")
    return lines


def element_diff_count(a: set[str], b: set[str]) -> int:
    """Number of differing element-lines: ``|A △ B|`` (symmetric difference).

    The paper's "element diff < 5" criterion. Two screens are the "same page"
    (element-wise) when this count is below ``element_diff_max``.
    """
    return len(a ^ b)


def element_jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity of two element-line sets, in [0.0, 1.0].

    The reference ``compare_actions`` criterion (``compare_actions`` returns
    ``jaccard > diff_max``, default 0.5). Empty-union → 0.0.
    """
    union = a | b
    return len(a & b) / len(union) if union else 0.0
