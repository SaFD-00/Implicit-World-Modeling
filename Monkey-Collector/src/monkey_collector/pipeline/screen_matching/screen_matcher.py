"""Live element-set screen matcher (page identity via extracted elements).

Replaces the structural-fingerprint page identity with MobileGPT-V2's
element-set cluster assignment, run live in the collection loop. Per screen:

  0. Pre-filter   — a structural fingerprint short-circuits exact revisits with
                    NO LLM call (the cheap analogue of V2's luminance prefilter).
  1. Step-1 match — for each stored page, text-blind ALL-match its anchor
                    fingerprints against the current screen → supported element
                    names + remaining (unaccounted) interactable indices.
  2. Expand       — mask the screen to the remaining UIs and re-extract until
                    dry (or a cap), discovering elements new to the best page.
  3. Classify     — set-assign A=(supported ∪ extracted) vs B=(stored names) →
                    EQSET / *_MERGE / *_NEW / DISJOINT.
  4. Dispatch     — MERGE reuses the stored page_key (frozen); NEW registers a
                    fresh page_key whose anchors are fingerprinted on the
                    current screen.

The emitted ``page_key`` drives BOTH the ``page_graph.json`` node identity and
the exploration abstract page. ``families`` (freshly-extracted, current-index
element families) feed the explorer's same-function compression; on a merge /
prefilter revisit it is empty (the page's families were recorded on first
sighting).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger

from monkey_collector.domain.page_graph import compute_xml_fingerprint
from monkey_collector.pipeline.screen_matching.page_knowledge import (
    KnowledgeRegistry,
    PageKnowledge,
)
from monkey_collector.pipeline.screen_matching.set_classifier import classify
from monkey_collector.pipeline.screen_matching.ui_attributes import (
    UIAttributes,
    extract_interactable_indexes,
    find_matching_node,
    get_ui_key_attrib,
    mask_xml_to_indexes,
    text_blind_requirements,
)

if TYPE_CHECKING:
    from monkey_collector.llm.element_extractor import ElementExtractor, ExtractedElement


@dataclass(frozen=True)
class ElementFamily:
    """A same-function element family with current-screen indices (for Memory)."""

    name: str
    element_index: list[int]
    key_element_index: list[int]


@dataclass(frozen=True)
class ScreenMatch:
    """Outcome of matching one screen: its page identity and (on a new page) families."""

    page_key: str
    is_new_page: bool
    match_type: str
    families: list[ElementFamily] = field(default_factory=list)
    page_description: str = ""


class ScreenMatcher:
    """Element-set page identifier with on-the-fly single-call extraction."""

    def __init__(
        self,
        extractor: ElementExtractor,
        cluster_merge_tolerance: float = 0.2,
        max_expand_iters: int = 3,
    ):
        self._extractor = extractor
        self._tolerance = cluster_merge_tolerance
        self._max_expand_iters = max_expand_iters
        self._registry = KnowledgeRegistry()
        self._fp_to_key: dict[tuple[str, str], str] = {}
        self._counter = 0

    def reset(self) -> None:
        """Drop all per-session knowledge (called between app sessions)."""
        self._registry = KnowledgeRegistry()
        self._fp_to_key = {}
        self._counter = 0

    # -- public ---------------------------------------------------------------

    def match(
        self,
        raw_xml: str,
        encoded_xml: str,
        activity: str,
        screenshot_path: str | None = None,
    ) -> ScreenMatch:
        """Identify the page of the current screen (see module docstring)."""
        fp = compute_xml_fingerprint(raw_xml)
        fp_key = (activity or "", fp)

        # 0. Structural pre-filter: exact revisit short-circuits, no LLM.
        cached = self._fp_to_key.get(fp_key)
        if cached is not None:
            logger.debug(f"screen_match: structural prefilter hit page={cached}")
            return ScreenMatch(cached, is_new_page=False, match_type="STRUCTURAL_IDENTICAL")

        try:
            tree = ET.fromstring(encoded_xml)
        except ET.ParseError:
            tree = None

        # No stored pages yet → straight to a new page (full extract).
        if len(self._registry) == 0 or tree is None:
            return self._new_page(
                fp_key, encoded_xml, tree, supported_names=[], best=None, additional=None,
                screenshot_path=screenshot_path,
            )

        # 1. Step-1 trigger match per stored page.
        candidates = []
        for page_key in self._registry.all_page_keys():
            page = self._registry.get(page_key)
            if page is None:
                continue
            supported, remaining = self._match_step1(tree, page)
            candidates.append((page_key, page, supported, remaining))

        # 2. Pick best (largest |supported|, ties → smaller stored page).
        best_key, best, supported, remaining = max(
            candidates, key=lambda c: (len(c[2]), -len(c[1].elements))
        )

        # 2b. Expand: re-extract on the leftover UIs until dry.
        additional = self._expand(encoded_xml, best, remaining)

        # 3. Classify A vs B.
        a_names = set(supported) | {e.name for e in additional}
        b_names = best.element_names
        cls = classify(a_names, b_names, self._tolerance)
        logger.info(
            f"screen_match: type={cls.match_type} page={best_key if cls.is_merge else 'NEW'} "
            f"|A|={len(a_names)} |B|={len(b_names)} |A∩B|={len(a_names & b_names)} "
            f"tol={self._tolerance:.2f}"
        )

        if cls.is_merge:
            self._fp_to_key[fp_key] = best_key
            return ScreenMatch(best_key, is_new_page=False, match_type=cls.match_type)

        return self._new_page(
            fp_key, encoded_xml, tree, supported_names=list(supported), best=best,
            additional=additional, match_type=cls.match_type, screenshot_path=screenshot_path,
        )

    # -- step 1 ---------------------------------------------------------------

    def _match_step1(self, tree: ET.Element, page: PageKnowledge) -> tuple[list[str], set[int]]:
        """Text-blind ALL-match a stored page's anchors. Returns (supported, remaining)."""
        remaining: set[int] = set()
        for tag in ("button", "input"):
            for node in tree.findall(f".//{tag}"):
                idx = node.attrib.get("index")
                if idx is not None:
                    remaining.add(int(idx))

        supported: list[str] = []
        for name, ui_list in page.key_elements.items():
            found = 0
            for ui in ui_list:
                matched = self._matched_indexes(tree, ui)
                for mi in matched:
                    remaining.discard(mi)
                found += len(matched)
            # ALL-match: every stored anchor must be present.
            if found >= len(ui_list) and ui_list:
                supported.append(name)

        for ui in page.extra_uis:
            for mi in self._matched_indexes(tree, ui):
                remaining.discard(mi)

        return supported, remaining

    @staticmethod
    def _matched_indexes(tree: ET.Element, ui: UIAttributes) -> list[int]:
        """Indices of nodes matching *ui* on the current tree (text-blind)."""
        req = text_blind_requirements(ui.to_dict())
        out: list[int] = []
        for node in find_matching_node(tree, req):
            idx = node.attrib.get("index")
            if idx is not None:
                out.append(int(idx))
        return out

    # -- step 2 (expand) ------------------------------------------------------

    def _expand(
        self, encoded_xml: str, best: PageKnowledge, remaining: set[int]
    ) -> list[ExtractedElement]:
        """Re-extract on the leftover UIs until no new element / remaining dry / cap."""
        additional: list[ExtractedElement] = []
        if not remaining:
            return additional
        known: list[ExtractedElement] = list(best.elements)
        known_names = {e.name for e in known}
        current = set(remaining)
        iters = 0
        while current and iters < self._max_expand_iters:
            iters += 1
            masked = mask_xml_to_indexes(encoded_xml, current)
            new = self._extractor.extract(masked, known_elements=known)
            new = [e for e in new if e.name not in known_names]
            if not new:
                break
            additional.extend(new)
            for e in new:
                known.append(e)
                known_names.add(e.name)
            claimed: set[int] = set()
            for e in new:
                claimed.update(e.element_index)
            shrunk = current - claimed
            if shrunk == current:
                break  # no progress — avoid non-termination
            current = shrunk
        return additional

    # -- new page -------------------------------------------------------------

    def _new_page(
        self,
        fp_key: tuple[str, str],
        encoded_xml: str,
        tree: ET.Element | None,
        supported_names: list[str],
        best: PageKnowledge | None,
        additional: list[ExtractedElement] | None,
        match_type: str = "NEW",
        screenshot_path: str | None = None,
    ) -> ScreenMatch:
        """Register a fresh page; fingerprint its anchors on the current screen."""
        # Freshly-extracted elements (current-screen indices). For an empty
        # registry this is a full extract; otherwise it is the expand output.
        if additional is None:
            additional = self._extractor.extract(encoded_xml, known_elements=[])

        page_key = f"page_{self._counter}"
        self._counter += 1

        key_elements: dict[str, list[UIAttributes]] = {}
        accounted: set[int] = set()

        # Supported elements: reuse stored anchor fingerprints (they matched here),
        # so the new page re-identifies on future visits the same way.
        if best is not None and tree is not None:
            supported_set = set(supported_names)
            for name in supported_set:
                ui_list = best.key_elements.get(name)
                if not ui_list:
                    continue
                key_elements[name] = ui_list
                for ui in ui_list:
                    accounted.update(self._matched_indexes(tree, ui))

        # Additional elements: fingerprint their anchors on the current screen.
        for el in additional:
            attrs = [
                UIAttributes.from_attrib_dict(get_ui_key_attrib(i, encoded_xml))
                for i in el.key_element_index
            ]
            if attrs:
                key_elements[el.name] = attrs
            accounted.update(el.key_element_index)

        # Extras: leftover interactables not owned by any anchor.
        all_inter = set(extract_interactable_indexes(encoded_xml))
        leftover = sorted(all_inter - accounted)
        extra_uis = [
            UIAttributes.from_attrib_dict(get_ui_key_attrib(i, encoded_xml)) for i in leftover
        ]

        # Stored elements for the page = supported (from B) + additional.
        supported_objs: list[ExtractedElement] = []
        if best is not None:
            supported_set = set(supported_names)
            supported_objs = [e for e in best.elements if e.name in supported_set]
        page_elements = supported_objs + list(additional)

        self._registry.add(
            PageKnowledge(
                page_key=page_key,
                elements=page_elements,
                key_elements=key_elements,
                extra_uis=extra_uis,
            )
        )
        self._fp_to_key[fp_key] = page_key
        logger.debug(
            f"screen_match: new page={page_key} type={match_type} "
            f"elements={[e.name for e in page_elements]} extras={len(extra_uis)}"
        )

        # Families for Memory: ONLY the freshly-extracted (current-index)
        # elements. Supported-from-B elements anchor page identity but carry
        # stored indices, so they are not handed to same-function compression.
        families = [
            ElementFamily(e.name, list(e.element_index), list(e.key_element_index))
            for e in additional
        ]
        return ScreenMatch(
            page_key, is_new_page=True, match_type=match_type, families=families
        )
