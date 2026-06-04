"""Page map: identify unique pages and build a transition graph."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from xml.etree import ElementTree as ET

from loguru import logger

from monkey_collector.xml.structured_parser import StructuredXmlParser

# ---------------------------------------------------------------------------
# XML structural fingerprinting (with parser preprocessing)
# ---------------------------------------------------------------------------

# Max children to include for scrollable containers (handles varying list lengths)
_MAX_SCROLL_CHILDREN = 3

# resource-id substrings of transient overlays (snackbar/toast/tooltip). Nodes
# whose stripped id contains one of these — and their subtrees — are excluded
# from the structural fingerprint so a momentary overlay does not fork a page.
_TRANSIENT_ID_MARKERS = ("snackbar", "toast", "tooltip")

# Window/activity labels that are not stable page identities. Kept in sync with
# pipeline.screen_guard (duplicated here to avoid a domain→pipeline import cycle).
_NON_PAGE_ACTIVITY_MARKERS = ("SoftInputWindow",)


def _canonical_activity(activity: str) -> str:
    """Normalize an activity label for page matching.

    Empty, keyboard, or other non-page window labels collapse to ``""`` so a
    flapping/missing activity name does not split one logical page into many.
    """
    activity = (activity or "").strip()
    if not activity:
        return ""
    if any(marker in activity for marker in _NON_PAGE_ACTIVITY_MARKERS):
        return ""
    return activity


def _is_transient_id(rid: str) -> bool:
    return any(marker in rid.lower() for marker in _TRANSIENT_ID_MARKERS)


def _preprocess_xml_for_fingerprint(xml_str: str) -> str:
    """Run XML through _reformat + _simplify for stable fingerprinting.

    Uses semantic tags (Button, TextField, etc.) and collapsed wrappers
    so that fingerprints are robust to wrapper depth changes and custom
    class name variations.  ``_clean`` is NOT applied to keep the
    ``Scroll`` tag for scrollable detection.
    """
    parser = StructuredXmlParser()
    try:
        reformatted = parser._reformat(xml_str)
        if not reformatted:
            return xml_str
        return parser._simplify(reformatted)
    except Exception:
        return xml_str


def _extract_structural_tuples(xml_str: str) -> frozenset[tuple[str, str, int]]:
    """Extract (tag, id, depth) tuples from preprocessed XML.

    After preprocessing, tags are semantic (Button, TextField, Image,
    Scroll, div, etc.) and resource-id is stored as ``id`` with the
    package prefix stripped.
    """
    preprocessed = _preprocess_xml_for_fingerprint(xml_str)
    try:
        root = ET.fromstring(preprocessed)
    except ET.ParseError:
        return frozenset()

    tuples: list[tuple[str, str, int]] = []
    stack: list[tuple[ET.Element, int]] = [(root, 0)]

    while stack:
        node, depth = stack.pop()
        tag = node.tag
        rid = node.attrib.get("id", "")

        # Skip transient overlays (snackbar/toast/tooltip) and their subtrees so
        # a momentary popup does not change the page fingerprint.
        if rid and _is_transient_id(rid):
            continue

        if tag:
            tuples.append((tag, rid, depth))

        is_scrollable = node.tag == "Scroll"
        children = list(node)
        if is_scrollable and len(children) > _MAX_SCROLL_CHILDREN:
            children = children[:_MAX_SCROLL_CHILDREN]

        for child in reversed(children):
            stack.append((child, depth + 1))

    return frozenset(tuples)


def compute_xml_fingerprint(xml_str: str) -> str:
    """Compute a structural fingerprint (MD5 hex) of uiautomator XML."""
    tuples = _extract_structural_tuples(xml_str)
    normalized = str(sorted(tuples))
    return hashlib.md5(normalized.encode()).hexdigest()


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class PageNode:
    id: int
    activity: str
    xml_fingerprint: str
    first_seen_step: int
    screenshot_step: int
    visit_count: int = 1


@dataclass
class TransitionEdge:
    from_page: int
    to_page: int
    action_type: str
    element_info: str
    first_seen_step: int
    count: int = 1


# ---------------------------------------------------------------------------
# PageGraph
# ---------------------------------------------------------------------------


class PageGraph:
    """Builds and manages a page-level transition graph."""

    def __init__(self, threshold: float = 0.85):
        self.nodes: list[PageNode] = []
        self.edges: list[TransitionEdge] = []
        self.threshold = threshold

        # Lookup caches
        self._page_lookup: dict[tuple[str, str], int] = {}
        self._page_tuples: dict[int, frozenset[tuple[str, str, int]]] = {}
        self._edge_lookup: set[tuple[int, int, str]] = set()

    # -- Page identification --------------------------------------------------

    def get_or_create_page(self, activity: str, xml_str: str, step: int) -> int:
        """Identify or create a page from activity name and XML content.

        Returns the page id.
        """
        activity = _canonical_activity(activity)
        fp = compute_xml_fingerprint(xml_str)
        tuples = _extract_structural_tuples(xml_str)

        # 1. Exact match on (activity, fingerprint)
        key = (activity, fp)
        if key in self._page_lookup:
            page_id = self._page_lookup[key]
            self.nodes[page_id].visit_count += 1
            return page_id

        # 2. Fuzzy match: Jaccard >= threshold. The activity gate only rejects
        #    when *both* labels are known and differ — an unknown ("") activity
        #    on either side falls back to pure structural matching so a missing
        #    or noisy label does not spawn a duplicate page.
        for node in self.nodes:
            if activity and node.activity and node.activity != activity:
                continue
            existing_tuples = self._page_tuples.get(node.id, frozenset())
            if _jaccard(tuples, existing_tuples) >= self.threshold:
                # Cache the new fingerprint for this page too
                self._page_lookup[key] = node.id
                # Upgrade an unknown stored label once a real one is observed.
                if not node.activity and activity:
                    node.activity = activity
                node.visit_count += 1
                return node.id

        # 3. New page
        page_id = len(self.nodes)
        node = PageNode(
            id=page_id,
            activity=activity,
            xml_fingerprint=fp,
            first_seen_step=step,
            screenshot_step=step,
        )
        self.nodes.append(node)
        self._page_lookup[key] = page_id
        self._page_tuples[page_id] = tuples
        return page_id

    # -- Transition management ------------------------------------------------

    def add_transition(
        self,
        from_page: int,
        to_page: int,
        action_type: str,
        element_info: str,
        step: int,
    ) -> bool:
        """Add a transition edge. Returns True if new, False if duplicate.

        Self-loops (from_page == to_page) are skipped.
        Dedup key: (from_page, to_page, action_type).
        """
        if from_page == to_page:
            return False

        edge_key = (from_page, to_page, action_type)
        if edge_key in self._edge_lookup:
            for edge in self.edges:
                if (edge.from_page, edge.to_page, edge.action_type) == edge_key:
                    edge.count += 1
                    break
            return False

        self._edge_lookup.add(edge_key)
        self.edges.append(
            TransitionEdge(
                from_page=from_page,
                to_page=to_page,
                action_type=action_type,
                element_info=element_info,
                first_seen_step=step,
            )
        )
        return True

    # -- Serialization --------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "nodes": [asdict(n) for n in self.nodes],
            "edges": [asdict(e) for e in self.edges],
            "metadata": {
                "total_pages": len(self.nodes),
                "total_transitions": len(self.edges),
                "threshold": self.threshold,
            },
        }

    def save(self, path: str) -> None:
        """Save graph to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(
            f"Page graph saved: {len(self.nodes)} pages, "
            f"{len(self.edges)} transitions → {path}"
        )

    @classmethod
    def load(cls, path: str) -> PageGraph:
        """Load graph from a JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        threshold = data.get("metadata", {}).get("threshold", 0.85)
        graph = cls(threshold=threshold)

        for nd in data.get("nodes", []):
            node = PageNode(
                id=nd["id"],
                activity=nd["activity"],
                xml_fingerprint=nd["xml_fingerprint"],
                first_seen_step=nd["first_seen_step"],
                screenshot_step=nd["screenshot_step"],
                visit_count=nd.get("visit_count", 1),
            )
            graph.nodes.append(node)
            graph._page_lookup[(node.activity, node.xml_fingerprint)] = node.id

        for ed in data.get("edges", []):
            edge = TransitionEdge(
                from_page=ed["from_page"],
                to_page=ed["to_page"],
                action_type=ed["action_type"],
                element_info=ed.get("element_info", ""),
                first_seen_step=ed["first_seen_step"],
                count=ed.get("count", 1),
            )
            graph.edges.append(edge)
            graph._edge_lookup.add(
                (edge.from_page, edge.to_page, edge.action_type)
            )

        return graph


# ---------------------------------------------------------------------------
# Post-hoc graph building from saved session
# ---------------------------------------------------------------------------


def _load_activity_map(session_dir: str) -> dict[int, str]:
    """Load step → activity_name mapping from activity_coverage.csv."""
    csv_path = os.path.join(session_dir, "activity_coverage.csv")
    mapping: dict[int, str] = {}
    if not os.path.exists(csv_path):
        return mapping
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                step = int(row["step"])
                activity = row.get("activity", "")
                mapping[step] = activity
            except (KeyError, ValueError):
                continue
    return mapping


def _load_events(session_dir: str) -> dict[int, dict]:
    """Load step → event dict mapping from events.jsonl."""
    events_path = os.path.join(session_dir, "events.jsonl")
    mapping: dict[int, dict] = {}
    if not os.path.exists(events_path):
        return mapping
    with open(events_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                step = event.get("step")
                if step is not None and not event.get("no_change_retry"):
                    mapping[int(step)] = event
            except (json.JSONDecodeError, ValueError):
                continue
    return mapping


def build_graph_from_session(
    session_dir: str,
    threshold: float = 0.85,
) -> PageGraph:
    """Reconstruct a page graph from a saved session directory.

    Reads xml/, events.jsonl, and activity_coverage.csv to build the graph.
    """
    graph = PageGraph(threshold=threshold)
    xml_dir = os.path.join(session_dir, "xml")

    if not os.path.isdir(xml_dir):
        logger.warning(f"No xml/ directory in {session_dir}")
        return graph

    # Load supporting data
    activity_map = _load_activity_map(session_dir)
    events = _load_events(session_dir)

    # Sorted XML files
    xml_files = sorted(
        f for f in os.listdir(xml_dir) if f.endswith(".xml")
    )

    previous_page_id: int | None = None

    for xml_file in xml_files:
        step_str = os.path.splitext(xml_file)[0]
        try:
            step = int(step_str)
        except ValueError:
            continue

        xml_path = os.path.join(xml_dir, xml_file)
        with open(xml_path, encoding="utf-8") as f:
            xml_str = f.read()

        # Get activity name (prefer events.jsonl, fall back to coverage CSV)
        event = events.get(step, {})
        activity = event.get("activity_name", "")
        if not activity:
            activity = activity_map.get(step, "")

        # Identify page
        current_page_id = graph.get_or_create_page(activity, xml_str, step)

        # Add transition from previous step
        if previous_page_id is not None and step > 0:
            prev_step = step - 1
            prev_event = events.get(prev_step, {})
            action_type = prev_event.get("action_type", "unknown")
            element_info = _element_info_from_event(prev_event)
            graph.add_transition(
                from_page=previous_page_id,
                to_page=current_page_id,
                action_type=action_type,
                element_info=element_info,
                step=step,
            )

        previous_page_id = current_page_id

    logger.info(
        f"Built graph from {session_dir}: "
        f"{len(graph.nodes)} pages, {len(graph.edges)} transitions"
    )
    return graph


def _element_info_from_event(event: dict) -> str:
    """Extract a brief element description from an event dict."""
    action_type = event.get("action_type", "unknown")
    idx = event.get("element_index", -1)
    if idx >= 0:
        return f"element_{idx}"
    return action_type
