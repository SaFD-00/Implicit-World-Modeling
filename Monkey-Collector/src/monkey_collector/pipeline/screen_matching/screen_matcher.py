"""Live BM25 page matcher (Mobile3M "Unique Page" mechanism).

Replaces the LLM element-set cluster assignment with the Mobile3M unique-page
mechanism, run live in the collection loop. Per screen:

  0. Pre-filter   — a structural fingerprint short-circuits exact XML revisits
                    with NO work; a screen with no interactable (button/input)
                    is declined outright (``pending``) so a loading/splash frame
                    never registers as a page — the first VALID screen becomes
                    page_0.
  1. Serialize    — the encoded XML is flattened to a normalized element-line
                    document (``element_lines.serialize_element_lines``): one
                    line per leaf / scroll node, ``index``/``bounds`` dropped.
  2. Retrieve     — BM25 (``bm25.Bm25Index``) over every stored page's
                    element-line document returns the top-K candidates.
  3. Verify       — each candidate, in score order, is confirmed by a
                    CONJUNCTIVE gate: an element criterion (symmetric-difference
                    count ``|A △ B| < element_diff_max``, or Jaccard
                    ``> element_jaccard_min``) AND a pixel gate (luminance
                    difference fraction ``< page_pixel_diff_threshold``, default
                    0.3). The first candidate passing BOTH merges; if none pass,
                    a fresh page is minted and added to the BM25 corpus.

This engine is LLM-FREE. LLM element extraction is now OPTIONAL enrichment: when
an *extractor* is supplied (``llm.element_extraction`` on), a NEW page runs ONE
extract to populate ``ScreenMatch.families`` for the explorer's same-function
compression; without one (the default) families are empty and matching relies
solely on the element-line document. Matching itself never calls the LLM.

Beyond PAGE identity (above), a MERGE also resolves a second-level OBSERVATION
identity: given the page is already fixed, which of its stored visual states (if
any) does this screenshot pixel-match? ``_record_observation`` answers this via a
page-SCOPED luminance lookup (``_page_luminance_lookup`` at the tighter
``screenshot_diff_threshold``, distinct from the page-level pixel gate above) —
a hit reuses that ``observation_num`` (no new write); a miss (or no comparator
available: prefilter disabled / no screenshot) allocates a new one. This is what
``ScreenMatch.observation_num``/``is_new_observation`` carry to the caller, which
persists a new observation's files only when ``is_new_observation`` is true.

The ``persist_filtered`` flag (config: ``screen_matching.persist_filtered``,
default ON) inverts the reuse decision for STORAGE: when set, every deduped
revisit — a structural-prefilter hit or a BM25 merge — allocates a FRESH
``observation_num`` under its (reused) ``page_key`` and reports
``is_new_observation=True``, so the caller persists it as its own observation (a
per-visit chain ``0,1,2,...`` per page). Page identity is untouched. A structural
prefilter HIT allocates via ``_allocate_observation(..., append_luma=False)`` so
the near-dup fingerprint is not re-appended to the capped luminance ring. With
the flag off the historical no-write-on-reuse behaviour is preserved.

The emitted ``page_key`` drives BOTH the ``page_graph.json`` node identity and
the exploration abstract page. On a NEW page ``families`` carries the
freshly-extracted, current-index element families (when an extractor is present)
that feed the explorer's same-function compression. On a merge / structural
revisit ``families`` is also populated — the matched page's stored elements
re-grounded on the current screen (their anchors re-matched here) — but the
explorer ignores them (the page's groups were computed on first sighting); it is
empty when the page was minted without an extractor.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger

from monkey_collector.domain.page_graph import compute_xml_fingerprint
from monkey_collector.pipeline.screen_matching.bm25 import Bm25Index
from monkey_collector.pipeline.screen_matching.element_lines import (
    element_diff_count,
    element_jaccard,
    serialize_element_lines,
)
from monkey_collector.pipeline.screen_matching.luminance import (
    extract_luminance_features,
    luminance_diff,
)
from monkey_collector.pipeline.screen_matching.page_knowledge import (
    KnowledgeRegistry,
    PageKnowledge,
)
from monkey_collector.pipeline.screen_matching.ui_attributes import (
    UIAttributes,
    extract_interactable_indexes,
    find_matching_node,
    get_ui_key_attrib,
    text_blind_requirements,
)

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

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

    ``families`` is populated on a new page (freshly-extracted, current-index
    families when an extractor is present, else empty) and on a merge / structural
    revisit (the matched page's stored elements re-grounded on the current
    screen). The explorer consumes it only on the first sighting of a
    ``page_key``; on a revisit it reaches ``{step}_elements.json`` but not
    same-function compression.

    ``pending`` flags a screen the matcher declined to register (a loading /
    splash frame with no interactable): it carries no ``page_key`` and the
    collection loop must NOT create a page node or persist elements for it — the
    first VALID screen of a session becomes ``page_0``.

    ``observation_num``/``is_new_observation`` carry the second-level decision:
    which of the page's stored visual states this screenshot corresponds to, and
    whether the caller must persist a new observation's files (true) or reuse an
    existing one (false, no new write). Under ``persist_filtered`` a deduped
    revisit reports true with a fresh ``observation_num`` so it is persisted.
    """

    page_key: str
    is_new_page: bool
    match_type: str
    families: list[ElementFamily] = field(default_factory=list)
    page_description: str = ""
    pending: bool = False
    observation_num: int = 0
    is_new_observation: bool = True


def _families_from_elements(elements: list[ExtractedElement]) -> list[ElementFamily]:
    """Convert ExtractedElements to an ElementFamily list, copying indices verbatim.

    ``name`` / ``description`` / ``parameters`` carry the LLM semantics; the
    indices are copied as-is. Use this ONLY for elements whose indices already
    live in the CURRENT screen's index space (a fresh extract). A stored page's
    elements on a revisit carry first-sighting indices and must be re-grounded
    via :meth:`ScreenMatcher._remap_families` instead.
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
    """BM25 page identifier with conjunctive element + pixel verification.

    Page identity is decided WITHOUT the LLM: serialize the encoded XML to an
    element-line document, retrieve BM25 top-K candidates, and confirm the first
    candidate that passes both the element criterion and the pixel gate. When an
    *extractor* is supplied (``llm.element_extraction`` on) a NEW page also runs
    one element extraction to populate ``families`` for exploration; without one
    (the default) families are empty and identity relies solely on element-lines.
    """

    # Per-page luminance-fingerprint observation cap. Bounds the observation
    # compare loop over a long session.
    _MAX_LUMINANCE_OBS = 10

    def __init__(
        self,
        extractor: ElementExtractor | None,
        cluster_merge_tolerance: float = 0.2,
        max_expand_iters: int = 3,
        luminance_prefilter: bool = False,
        luminance_threshold: int = 10,
        screenshot_diff_threshold: float = 0.02,
        luminance_low_res_width: int = 100,
        persist_filtered: bool = False,
        bm25_top_k: int = 5,
        element_criterion: str = "diff",
        element_diff_max: int = 5,
        element_jaccard_min: float = 0.5,
        page_pixel_diff_threshold: float = 0.3,
    ):
        self._extractor = extractor
        # cluster_merge_tolerance / max_expand_iters are retained for signature
        # stability (config/CLI threading) but are unused by the BM25 match path.
        self._tolerance = cluster_merge_tolerance
        self._max_expand_iters = max_expand_iters
        # Luminance knobs. ``screenshot_diff_threshold`` governs OBSERVATION
        # identity (the tighter per-page dedup); the PAGE-level pixel gate uses
        # ``page_pixel_diff_threshold``. All image work is skipped when
        # ``luminance_prefilter`` is off (current_feat stays None).
        self._luma_enabled = luminance_prefilter
        self._luma_threshold = luminance_threshold
        self._luma_diff_threshold = screenshot_diff_threshold
        self._luma_width = luminance_low_res_width
        self._persist_filtered = persist_filtered
        # BM25 / verification knobs (Mobile3M unique-page mechanism).
        self._bm25_top_k = bm25_top_k
        self._element_criterion = element_criterion
        self._element_diff_max = element_diff_max
        self._element_jaccard_min = element_jaccard_min
        self._page_pixel_diff_threshold = page_pixel_diff_threshold
        self._bm25 = Bm25Index()
        self._registry = KnowledgeRegistry()
        self._fp_to_key: dict[tuple[str, str], tuple[str, int]] = {}
        self._counter = 0

    def reset(self) -> None:
        """Drop all per-session knowledge (called between app sessions)."""
        self._registry = KnowledgeRegistry()
        self._fp_to_key = {}
        self._counter = 0
        self._bm25.clear()

    def rehydrate(
        self,
        pages: dict[str, PageKnowledge],
        fp_to_key: dict[tuple[str, str], tuple[str, int]],
        counter: int,
    ) -> None:
        """Replace in-memory state wholesale from disk (resume only).

        Call right after :meth:`reset`, before the collection loop resumes.
        *pages* becomes the registry's contents, *fp_to_key* the structural
        exact-match cache, *counter* the next ``page_N`` index to allocate — all
        rebuilt from the durable ``data/{package}/pages/`` tree by
        ``pipeline.screen_matching.rehydrate.rehydrate_screen_matcher``. The BM25
        corpus is rebuilt from each page's ``element_lines`` (in page-key order).
        """
        for page in pages.values():
            self._registry.add(page)
        self._fp_to_key = dict(fp_to_key)
        self._counter = counter
        self._bm25.clear()
        for page_key in sorted(pages, key=lambda k: int(k) if k.isdigit() else 0):
            self._bm25.add_document(page_key, pages[page_key].element_lines)

    # -- public ---------------------------------------------------------------

    def get_page_knowledge(self, page_key: str) -> PageKnowledge | None:
        """Look up a page's stored knowledge by key (for persisting ``page.json``)."""
        return self._registry.get(page_key)

    def match(
        self,
        raw_xml: str,
        encoded_xml: str,
        activity: str,
        screenshot: bytes | None = None,
    ) -> ScreenMatch:
        """Identify the page of the current screen (see module docstring)."""
        fp = compute_xml_fingerprint(raw_xml)
        fp_key = (activity or "", fp)

        # Luminance fingerprint of the current frame, computed once. Only when the
        # prefilter is enabled AND a screenshot arrived — otherwise zero image work.
        current_feat = (
            extract_luminance_features(screenshot, self._luma_width)
            if (self._luma_enabled and screenshot)
            else None
        )

        # 0. Structural pre-filter: exact revisit short-circuits, no work. When
        # persist_filtered is off the cached (page_key, observation_num) IS the
        # reused observation (no new files); when on, this revisit is persisted
        # as a fresh observation under the same page.
        cached = self._fp_to_key.get(fp_key)
        if cached is not None:
            cached_key, cached_obs = cached
            logger.debug(
                f"screen_match: structural prefilter hit page={cached_key} obs={cached_obs}"
            )
            cached_page = self._registry.get(cached_key)
            try:
                c_tree = ET.fromstring(encoded_xml)
            except ET.ParseError:
                c_tree = None
            fams = (
                self._remap_families(c_tree, cached_page)
                if (c_tree is not None and cached_page is not None)
                else []
            )
            if self._persist_filtered and cached_page is not None:
                new_obs = self._allocate_observation(cached_page, current_feat, append_luma=False)
                self._fp_to_key[fp_key] = (cached_key, new_obs)
                return ScreenMatch(
                    cached_key, is_new_page=False, match_type="STRUCTURAL_IDENTICAL", families=fams,
                    observation_num=new_obs, is_new_observation=True,
                )
            return ScreenMatch(
                cached_key, is_new_page=False, match_type="STRUCTURAL_IDENTICAL", families=fams,
                observation_num=cached_obs, is_new_observation=False,
            )

        # No interactable (button/input) on this screen → a loading/splash frame.
        # Decline to register it (no page): the first VALID screen of the session
        # must become page_0, not an empty blackhole that later screens merge into.
        if not extract_interactable_indexes(encoded_xml):
            logger.debug("screen_match: no interactable, declining (pending)")
            return ScreenMatch("", is_new_page=False, match_type="PENDING_EMPTY", pending=True)

        try:
            tree = ET.fromstring(encoded_xml)
        except ET.ParseError:
            tree = None

        # 1. Serialize the encoded XML to the element-line document (BM25 doc).
        lines = serialize_element_lines(encoded_xml)

        # An empty document must never merge (a size-0 query would let a tiny
        # candidate spuriously pass the element criterion). First screen of a
        # session (empty registry) is likewise a new page.
        if not lines or len(self._registry) == 0:
            return self._new_page(fp_key, encoded_xml, tree, lines, current_feat)

        # 2-3. BM25 top-K retrieval → conjunctive (element AND pixel) verification.
        cur_set = set(lines)
        for page_key, _score in self._bm25.top_k(lines, self._bm25_top_k):
            page = self._registry.get(page_key)
            if page is None:
                continue
            if not self._element_ok(cur_set, set(page.element_lines)):
                continue
            if not self._pixel_ok(page, current_feat):
                continue
            # MATCH → merge. Resolve OBSERVATION identity within the page.
            obs_num, is_new_obs = self._record_observation(
                page_key, current_feat, allow_reuse=True,
            )
            self._fp_to_key[fp_key] = (page_key, obs_num)
            fams = self._remap_families(tree, page) if tree is not None else []
            logger.info(
                f"screen_match: BM25_MERGE page={page_key} "
                f"|cur|={len(cur_set)} |cand|={len(page.element_lines)} "
                f"crit={self._element_criterion}"
            )
            return ScreenMatch(
                page_key, is_new_page=False, match_type="BM25_MERGE", families=fams,
                observation_num=obs_num, is_new_observation=is_new_obs,
            )

        # No candidate confirmed → a genuinely new page.
        return self._new_page(fp_key, encoded_xml, tree, lines, current_feat)

    # -- verification ---------------------------------------------------------

    def _element_ok(self, cur: set[str], cand: set[str]) -> bool:
        """Element criterion between the current and a candidate element-line set."""
        if self._element_criterion == "jaccard":
            return element_jaccard(cur, cand) > self._element_jaccard_min
        return element_diff_count(cur, cand) < self._element_diff_max

    def _pixel_ok(self, page: PageKnowledge, feat: PILImage | None) -> bool:
        """Pixel gate: is *feat* within ``page_pixel_diff_threshold`` of *page*?

        No comparator (no screenshot / luminance disabled) OR the candidate has
        no stored fingerprint → the gate ABSTAINS (returns True) and the element
        criterion alone decides. Otherwise the minimum luminance-difference
        fraction across the page's stored fingerprints must be below the
        PAGE-level threshold. A size mismatch makes ``luminance_diff`` return 1.0
        → the gate fails = different page (Mobile3M dimension-mismatch=different).
        """
        if feat is None or not page.luminance_features:
            return True
        return min(
            luminance_diff(feat, stored, self._luma_threshold)
            for _, stored in page.luminance_features
        ) < self._page_pixel_diff_threshold

    # -- family regrounding ---------------------------------------------------

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
        approximated by the anchor indices; an element with no current match (not
        rendered on this screen) is dropped. Returns ``[]`` for a page minted
        without an extractor (no stored elements).
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

    # -- observation identity -------------------------------------------------

    def _page_luminance_lookup(self, page: PageKnowledge, feat: PILImage) -> int | None:
        """First observation_num of *page* whose luminance fingerprint matches
        *feat* (within ``screenshot_diff_threshold``), else None. Scoped to one
        page's own observations — decides OBSERVATION identity given the page is
        already fixed (distinct from the PAGE-level :meth:`_pixel_ok` gate).
        """
        for obs_num, stored in page.luminance_features:
            if luminance_diff(feat, stored, self._luma_threshold) < self._luma_diff_threshold:
                return int(obs_num)
        return None

    def _allocate_observation(
        self, page: PageKnowledge, feat: PILImage | None, append_luma: bool = True,
    ) -> int:
        """Allocate *page*'s next observation_num (monotonic ++), the single
        place per-page observation numbers are minted.

        When *append_luma* and the luminance prefilter is on and a fingerprint
        was computed, records ``(obs_num, feat)`` so future screens can dedup
        against this exact observation (capped at ``_MAX_LUMINANCE_OBS``, oldest
        evicted). Callers persisting a prefilter HIT pass ``append_luma=False``:
        that frame already matches a stored fingerprint, so re-adding a near-dup
        would only churn the cap.
        """
        obs_num = page.next_observation_num
        page.next_observation_num += 1
        if append_luma and self._luma_enabled and feat is not None:
            page.luminance_features.append((obs_num, feat))
            if len(page.luminance_features) > self._MAX_LUMINANCE_OBS:
                del page.luminance_features[0]
        return obs_num

    def _record_observation(
        self, page_key: str, feat: PILImage | None, allow_reuse: bool,
    ) -> tuple[int, bool]:
        """Resolve (observation_num, is_new_observation) for *page_key*.

        ``allow_reuse=False`` (a brand-new page): always allocates observation 0.
        ``allow_reuse=True`` (a BM25 merge, page identity already resolved): tries
        the page-scoped luminance lookup first when the prefilter is enabled and a
        fingerprint was computed; a hit reuses that observation (no new write). A
        miss — or no pixel comparator available (prefilter disabled / no
        screenshot) — always allocates a new observation.

        When ``persist_filtered`` is on the reuse short-circuit is skipped
        entirely: a merge revisit is persisted as its own fresh observation, so
        every visit lands on disk (per-visit observation chain).
        """
        page = self._registry.get(page_key)
        if page is None:
            # Caller already resolved the page identity, so this should not
            # happen; fail safe to a fresh observation rather than raise.
            return 0, True

        if (
            allow_reuse
            and not self._persist_filtered
            and self._luma_enabled
            and feat is not None
        ):
            hit = self._page_luminance_lookup(page, feat)
            if hit is not None:
                return hit, False

        return self._allocate_observation(page, feat, append_luma=True), True

    # -- new page -------------------------------------------------------------

    def _new_page(
        self,
        fp_key: tuple[str, str],
        encoded_xml: str,
        tree: ET.Element | None,
        lines: list[str],
        current_feat: PILImage | None = None,
    ) -> ScreenMatch:
        """Register a fresh page; add its element-line document to the BM25 corpus.

        With an extractor (``llm.element_extraction`` on), run ONE extract to
        populate the page's elements + anchor fingerprints and the returned
        families (the explorer's same-function grouping). Without one (default),
        the page carries no elements/anchors and families are empty — matching
        re-identifies it via BM25 + element/pixel only, which needs neither.
        """
        page_key = str(self._counter)
        self._counter += 1

        elements: list[ExtractedElement] = []
        key_elements: dict[str, list[UIAttributes]] = {}
        extra_uis: list[UIAttributes] = []

        if self._extractor is not None:
            elements = self._extractor.extract(encoded_xml, known_elements=[])
            accounted: set[int] = set()
            for el in elements:
                attrs = [
                    UIAttributes.from_attrib_dict(get_ui_key_attrib(i, encoded_xml))
                    for i in el.key_element_index
                ]
                if attrs:
                    key_elements[el.name] = attrs
                accounted.update(el.key_element_index)
            leftover = sorted(set(extract_interactable_indexes(encoded_xml)) - accounted)
            extra_uis = [
                UIAttributes.from_attrib_dict(get_ui_key_attrib(i, encoded_xml)) for i in leftover
            ]

        self._registry.add(
            PageKnowledge(
                page_key=page_key,
                elements=elements,
                key_elements=key_elements,
                extra_uis=extra_uis,
                element_lines=lines,
            )
        )
        self._bm25.add_document(page_key, lines)

        # Observation 0: a brand-new page has nothing to reuse. Also records the
        # first-sighting luminance fingerprint (no-op when the prefilter is off).
        obs_num, is_new_obs = self._record_observation(page_key, current_feat, allow_reuse=False)
        self._fp_to_key[fp_key] = (page_key, obs_num)
        logger.debug(
            f"screen_match: new page={page_key} elements={[e.name for e in elements]} "
            f"lines={len(lines)}"
        )

        families = _families_from_elements(elements)
        return ScreenMatch(
            page_key, is_new_page=True, match_type="NEW", families=families,
            observation_num=obs_num, is_new_observation=is_new_obs,
        )
