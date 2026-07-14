"""Canvas-screen detection on the RAW uiautomator XML (S-9 map fragmentation).

A "canvas" screen is one whose content is a single interactive drawing surface
filling the viewport — a map, a photo viewer, a game board. Its accessibility
tree carries almost no structure, but the widgets floating ON it (osmand's
scale bar, distance readout, address, and state-dependent aria-labels) rewrite
themselves on every pan/zoom, so the element-line document changes on every
frame even though the page is the same. That is what fragmented osmand's
``MapActivity`` into 46/26 pages (95.6% / 84.6% of mints were element-blocked).

Detection is BOUNDS-BASED, not class-based: the osmand map surface appears in
the a11y tree as a LEAF ``android.view.View`` with ``clickable="true"`` and
bounds covering 0.97 of the screen — there is no SurfaceView/TextureView/
GLSurfaceView node to key off (zero occurrences across the archived map dumps).
So: a leaf, clickable (or long-clickable), covering at least *min_area_frac* of
the screen. The screen area is taken as the largest node bounds in the dump
(the root frame), which keeps the test resolution-independent.

Scroll containers do NOT hit this test: a full-screen RecyclerView is
``scrollable``, not ``clickable``, and is not a leaf (it holds its rows).
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

# uiautomator bounds: "[x1,y1][x2,y2]".
_BOUNDS_RE = re.compile(r"\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]")


def _bounds_area(bounds: str) -> int:
    """Area of a uiautomator ``bounds`` string; 0 when absent/malformed/empty."""
    m = _BOUNDS_RE.fullmatch(bounds.strip()) if bounds else None
    if m is None:
        return 0
    x1, y1, x2, y2 = (int(g) for g in m.groups())
    return max(0, x2 - x1) * max(0, y2 - y1)


def is_canvas_screen(raw_xml: str, min_area_frac: float = 0.7) -> bool:
    """Does *raw_xml* show a full-screen interactive drawing surface?

    True when some node is (a) a LEAF (no element children), (b) ``clickable``
    or ``long-clickable``, and (c) at least *min_area_frac* of the screen area
    (the largest node bounds in this dump). Parse failure, no bounds, or a
    zero-area screen → False (the matcher must never break on a bad dump; a
    False here just routes the screen down the unchanged non-canvas path).
    """
    try:
        root = ET.fromstring(raw_xml)
    except ET.ParseError:
        return False

    nodes = [(node, _bounds_area(node.attrib.get("bounds", ""))) for node in root.iter("node")]
    screen_area = max((area for _, area in nodes), default=0)
    if screen_area <= 0:
        return False

    return any(
        len(node) == 0
        and (node.attrib.get("clickable") == "true" or node.attrib.get("long-clickable") == "true")
        and area >= min_area_frac * screen_area
        for node, area in nodes
    )
