"""Interactable-element detection on Monkey-Collector's *encoded* XML schema.

The encoder (:func:`~monkey_collector.xml.structured_parser.encode_to_html_xml`)
keeps only ``index, bounds`` plus per-tag ``aria-label / alt / type / value /
checked / data-scroll / role`` — it drops resource-id and class. Interactable
tags in encoded XML are ``button`` and ``input`` (a text field is ``input`` with
no checkbox type; a checkbox is ``input type="checkbox"``). Scroll containers
are ``div[data-scroll="true"]`` and are handled by the exploration engine via
negative indices, so they are intentionally excluded here.

These functions operate on the SAME index space as ``SemanticElement.index``
and ``{step}_encoded.xml`` (both flow through ``StructuredXmlParser.parse`` →
``_renumber`` → ``_clear_bounds``), so the indices align 1:1 with exploration
grounding.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

# Encoded interactable tags. Scroll (``div[data-scroll]``) is deliberately
# excluded — the exploration engine owns scrollables via negative indices.
INTERACTABLE_TAGS = ("button", "input")


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
