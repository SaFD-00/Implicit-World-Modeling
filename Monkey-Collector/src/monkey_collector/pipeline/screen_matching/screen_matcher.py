"""Live element-set screen matcher (page identity via extracted elements).

Replaces the structural-fingerprint page identity with MobileGPT-V2's
element-set cluster assignment, run live in the collection loop. Per screen:

  0. Pre-filter   — a structural fingerprint short-circuits exact revisits with
                    NO LLM call (the cheap analogue of V2's luminance prefilter);
                    a screen with no interactable (button/input) is declined
                    outright (``pending``) so a loading/splash frame never
                    registers as a page — the first VALID screen becomes page_0.
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
the exploration abstract page. On a NEW page ``families`` carries the
freshly-extracted, current-index element families that feed the explorer's
same-function compression. On a merge / prefilter revisit ``families`` is also
populated — the matched page's stored elements re-grounded on the current
screen (their anchors re-matched here, so ``{step}_elements.json`` records
current-screen indices) — but the explorer ignores them (the page's groups were
computed on first sighting); since only anchors are stored the merge-path
``element_index`` approximates to the anchor indices and elements not rendered
on the current screen are dropped.
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
    """A same-function element family with current-screen indices (for Memory).

    ``description`` / ``parameters`` carry the LLM's extracted semantics through
    to ``{step}_elements.json`` (they originate on :class:`ExtractedElement` and
    are filled at family-build time; both are appended at the END to preserve the
    positional construction and the dataclass default-ordering rule).
    """

    name: str
    element_index: list[int]
    key_element_index: list[int]
    description: str = ""
    parameters: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ScreenMatch:
    """Outcome of matching one screen: its page identity and its element families.

    ``families`` is populated both on a new page (freshly-extracted, current-index
    families) and on a merge / prefilter revisit (the matched page's stored
    elements re-grounded on the current screen — see the module docstring). The
    explorer consumes it only on the first sighting of a ``page_key``; on a
    revisit it reaches ``{step}_elements.json`` but not same-function compression.

    ``pending`` flags a screen the matcher declined to register (a loading /
    splash frame with no interactable, or an otherwise empty extract): it carries
    no ``page_key`` and the collection loop must NOT create a page node or persist
    elements for it — the first VALID screen of a session becomes ``page_0``.
    """

    page_key: str
    is_new_page: bool
    match_type: str
    families: list[ElementFamily] = field(default_factory=list)
    page_description: str = ""
    pending: bool = False


def _families_from_elements(elements: list[ExtractedElement]) -> list[ElementFamily]:
    """Convert ExtractedElements to an ElementFamily list, copying indices verbatim.

    ``name`` / ``description`` / ``parameters`` carry the LLM semantics; the
    indices are copied as-is. Use this ONLY for elements whose indices already
    live in the CURRENT screen's index space (a fresh extract / the expand
    output). A stored page's elements on a revisit carry first-sighting indices
    and must be re-grounded via :meth:`ScreenMatcher._remap_families` instead.
    """
    return [
        ElementFamily(
            name=e.name,
            element_index=list(e.element_index),
            key_element_index=list(e.key_element_index),
            description=e.description,
            parameters=dict(e.parameters),
        )
        for e in elements
    ]


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
            # Fill families from the cached page, re-grounded on the current
            # screen (still no LLM call — anchors come from the registry).
            cached_page = self._registry.get(cached)
            try:
                c_tree = ET.fromstring(encoded_xml)
            except ET.ParseError:
                c_tree = None
            fams = (
                self._remap_families(c_tree, cached_page)
                if (c_tree is not None and cached_page is not None)
                else []
            )
            return ScreenMatch(
                cached, is_new_page=False, match_type="STRUCTURAL_IDENTICAL", families=fams
            )

        # No interactable (button/input) on this screen → a loading/splash frame.
        # Decline to register it (no LLM call, no page): the first VALID screen of
        # the session must become page_0, not an empty blackhole that later
        # screens merge into. The loop's empty-UI guard then waits / relaunches.
        if not extract_interactable_indexes(encoded_xml):
            logger.debug("screen_match: no interactable, declining (pending)")
            return ScreenMatch("", is_new_page=False, match_type="PENDING_EMPTY", pending=True)

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
            # Fill families from the matched page, re-grounded on the current
            # screen (stored indices are first-sighting; remap to this step).
            # Then append any expand-discovered elements (e.g. SUPERSET
            # scroll-reveal), which already carry current-screen indices.
            fams = self._remap_families(tree, best) if tree is not None else []
            seen = {f.name for f in fams}
            fams += [f for f in _families_from_elements(additional) if f.name not in seen]
            return ScreenMatch(
                best_key, is_new_page=False, match_type=cls.match_type, families=fams
            )

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

    def _remap_families(self, tree: ET.Element, page: PageKnowledge) -> list[ElementFamily]:
        """Re-ground a stored page's elements on the CURRENT screen.

        On a merge / structural revisit the page_key is reused, but the stored
        ``element_index`` / ``key_element_index`` live in the page's
        FIRST-SIGHTING index space, which need not match the current step's
        encoded XML. Re-match each element's anchor fingerprints
        (``page.key_elements[name]``) against the current ``tree`` to recover
        current-screen indices. Only anchors are stored, so ``element_index`` is
        approximated by the anchor indices (the full same-function family is not
        recoverable); an element with no current match (not rendered on this
        screen) is dropped.
        """
        fams: list[ElementFamily] = []
        for e in page.elements:
            anchors = page.key_elements.get(e.name, [])
            cur = sorted({mi for ui in anchors for mi in self._matched_indexes(tree, ui)})
            if not cur:
                continue
            fams.append(
                ElementFamily(
                    name=e.name,
                    element_index=cur,
                    key_element_index=cur,
                    description=e.description,
                    parameters=dict(e.parameters),
                )
            )
        return fams

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
        # description/parameters ride along so they reach {step}_elements.json.
        families = _families_from_elements(additional)
        return ScreenMatch(
            page_key, is_new_page=True, match_type=match_type, families=families
        )
