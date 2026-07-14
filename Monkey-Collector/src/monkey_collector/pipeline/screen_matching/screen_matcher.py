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

THE SAME-PACKAGE GUARD (``package_guard``, default on) is a veto that runs ahead
of the whole of step 3: a candidate whose page was minted under a DIFFERENT
package than the current screen can never merge, whatever the gates say. BM25
verification never looked at which APP a screen belongs to (only the structural
pre-filter keys on the activity), so a launcher/home frame sharing a few generic
element-lines with an app screen could — and did — merge into its page (measured
in the pre-fix live corpora: osmand armB pages 0/34, broccoli page 0). Text was
the de-facto defence, which is why the guard covers the WHOLE merge path and not
just the canvas pairs whose blinding removes it. The comparison is at PACKAGE
level, not activity: the baseline deliberately merges across window labels of one
app (DrawerLayout ↔ MapActivity), and ``page_graph._canonical_activity`` already
holds that a window label is not an identity. Either side's package missing → the
guard ABSTAINS (fail-open): an unparsable activity name must not fragment a page.

CANVAS PAIRS (``canvas_merge``, default on) are the one exception to step 3.
When the current screen AND the candidate page are BOTH canvas screens (a
full-screen interactive drawing surface — ``canvas.is_canvas_screen``), the
element criterion is evaluated on the TEXT-BLIND element-line sets and the pixel
gate abstains. On a map, the floating scale bar / distance / address readouts
rewrite themselves on every pan and the rendered tiles change wholesale, so both
gates are measuring viewport state rather than page identity (measured: 95.6% /
84.6% of osmand's map-page mints were element-blocked, at diffs far above any
sane threshold). The two relaxations are one mechanism, not two knobs: blinding
the text alone still leaves the pixel gate rejecting most map re-visits. Every
OTHER pair — and the whole knob-off path — takes the unchanged route above; the
thresholds are the same ones, reused. BM25 retrieval, the structural pre-filter,
and observation identity are untouched by the canvas path.

This engine is LLM-FREE: page identity relies solely on the element-line
document plus the gates above. Matching never calls the LLM.

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
the exploration abstract page.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from monkey_collector.domain.page_graph import compute_xml_fingerprint
from monkey_collector.pipeline.screen_matching.bm25 import Bm25Index
from monkey_collector.pipeline.screen_matching.canvas import is_canvas_screen
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
    extract_interactable_indexes,
)

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage


@dataclass(frozen=True)
class ScreenMatch:
    """Outcome of matching one screen: its page identity.

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
    pending: bool = False
    observation_num: int = 0
    is_new_observation: bool = True


def package_of(activity: str) -> str:
    """Package part of an activity label (``package/window.Class`` → ``package``).

    Returns ``""`` for an empty/degenerate label — the caller treats that as
    "unknown", which makes the merge guard abstain (fail-open) rather than
    fragment a page over a missing activity name.
    """
    return (activity or "").strip().split("/", 1)[0].strip()


class ScreenMatcher:
    """BM25 page identifier with conjunctive element + pixel verification.

    Page identity is decided WITHOUT the LLM: serialize the encoded XML to an
    element-line document, retrieve BM25 top-K candidates, and confirm the first
    candidate that passes both the element criterion and the pixel gate.
    """

    # Per-page luminance-fingerprint observation cap. Bounds the observation
    # compare loop over a long session.
    _MAX_LUMINANCE_OBS = 10

    def __init__(
        self,
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
        canvas_merge: bool = True,
        canvas_min_area_frac: float = 0.7,
        package_guard: bool = True,
    ):
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
        # Canvas-gated text-blind verification. The knob gates the MATCH path
        # only: a page always stores its is_canvas / blind element-lines, so the
        # durable page.json cannot depend on how the knob stood when it was
        # written (and a resume with the knob flipped is coherent either way).
        self._canvas_merge = canvas_merge
        self._canvas_min_area_frac = canvas_min_area_frac
        # Same-package merge guard. BM25 verification never looked at WHICH APP a
        # screen belongs to — only the structural pre-filter keys on the activity —
        # so a launcher/home frame that happens to share a few generic element-lines
        # with an app screen could merge into its page (measured in the pre-fix live
        # corpora: armB osmand pages 0/34, broccoli page 0). Text was the de-facto
        # defence and the canvas path's blinding removes it, so the guard covers the
        # WHOLE BM25 merge path, not just canvas pairs.
        self._package_guard = package_guard
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
            if self._persist_filtered and cached_page is not None:
                new_obs = self._allocate_observation(cached_page, current_feat, append_luma=False)
                self._fp_to_key[fp_key] = (cached_key, new_obs)
                return ScreenMatch(
                    cached_key, is_new_page=False, match_type="STRUCTURAL_IDENTICAL",
                    observation_num=new_obs, is_new_observation=True,
                )
            return ScreenMatch(
                cached_key, is_new_page=False, match_type="STRUCTURAL_IDENTICAL",
                observation_num=cached_obs, is_new_observation=False,
            )

        # No interactable (button/input) on this screen → a loading/splash frame.
        # Decline to register it (no page): the first VALID screen of the session
        # must become page_0, not an empty blackhole that later screens merge into.
        if not extract_interactable_indexes(encoded_xml):
            logger.debug("screen_match: no interactable, declining (pending)")
            return ScreenMatch("", is_new_page=False, match_type="PENDING_EMPTY", pending=True)

        # 1. Serialize the encoded XML to the element-line document (BM25 doc),
        # plus the text-blind projection + canvas flag the canvas path needs.
        # Both are computed regardless of the knob so a minted page's stored
        # identity never depends on it (see __init__).
        lines = serialize_element_lines(encoded_xml)
        blind_lines = serialize_element_lines(encoded_xml, blind_text=True)
        cur_canvas = is_canvas_screen(raw_xml, self._canvas_min_area_frac)

        # An empty document must never merge (a size-0 query would let a tiny
        # candidate spuriously pass the element criterion). First screen of a
        # session (empty registry) is likewise a new page.
        if not lines or len(self._registry) == 0:
            return self._new_page(
                fp_key, lines, current_feat, cur_canvas, blind_lines, activity,
            )

        # 2-3. BM25 top-K retrieval → same-package guard, then the conjunctive
        # (element AND pixel) verification, with a canvas pair taking the
        # text-blind / pixel-abstaining route.
        cur_set = set(lines)
        cur_blind_set = set(blind_lines)
        for page_key, _score in self._bm25.top_k(lines, self._bm25_top_k):
            page = self._registry.get(page_key)
            if page is None:
                continue
            canvas_pair, package_ok, element_ok, pixel_ok = self._verify_candidate(
                cur_set, cur_blind_set, page, current_feat, cur_canvas, activity,
            )
            if not package_ok:
                logger.debug(
                    f"screen_match: package guard blocked merge into page={page_key} "
                    f"(cur={package_of(activity)!r} page={package_of(page.first_activity)!r})"
                )
                continue
            if not (element_ok and pixel_ok):
                continue
            # MATCH → merge. Resolve OBSERVATION identity within the page.
            obs_num, is_new_obs = self._record_observation(
                page_key, current_feat, allow_reuse=True,
            )
            self._fp_to_key[fp_key] = (page_key, obs_num)
            logger.info(
                f"screen_match: BM25_MERGE page={page_key} "
                f"|cur|={len(cur_set)} |cand|={len(page.element_lines)} "
                f"crit={self._element_criterion} canvas_pair={canvas_pair}"
            )
            return ScreenMatch(
                page_key, is_new_page=False, match_type="BM25_MERGE",
                observation_num=obs_num, is_new_observation=is_new_obs,
            )

        # No candidate confirmed → a genuinely new page.
        return self._new_page(
            fp_key, lines, current_feat, cur_canvas, blind_lines, activity,
        )

    # -- verification ---------------------------------------------------------

    def _verify_candidate(
        self,
        cur_set: set[str],
        cur_blind_set: set[str],
        page: PageKnowledge,
        feat: PILImage | None,
        cur_canvas: bool,
        cur_activity: str = "",
    ) -> tuple[bool, bool, bool, bool]:
        """Run the full gate for one candidate → (canvas_pair, package, element, pixel).

        The single place a merge is decided, so ``match()`` and the offline
        replay's diagnosis cannot drift apart.

        The PACKAGE guard comes first and covers EVERY merge, canvas or not: the
        candidate page's minting package must equal the current screen's, or the
        two screens are different apps and no element/pixel similarity may merge
        them. It ABSTAINS (passes) when either side's package is unknown —
        fail-open, so a missing activity label fragments nothing.

        A CANVAS PAIR (knob on and both sides canvas screens) then compares the
        TEXT-BLIND element-line sets — via the same :meth:`_element_ok`, the same
        criterion and the same thresholds — and abstains from the pixel gate (a
        repainted map fails it by construction). Any other pair is the historical
        path, byte for byte.
        """
        package_ok = self._package_ok(page, cur_activity)
        canvas_pair = self._canvas_merge and cur_canvas and page.is_canvas
        if not package_ok:
            return canvas_pair, False, False, False
        if canvas_pair:
            return True, True, self._element_ok(cur_blind_set, set(page.element_lines_blind)), True
        # Short-circuit as the original loop did: the pixel gate is only asked
        # once the element criterion has passed.
        element_ok = self._element_ok(cur_set, set(page.element_lines))
        pixel_ok = element_ok and self._pixel_ok(page, feat)
        return False, True, element_ok, pixel_ok

    def _package_ok(self, page: PageKnowledge, cur_activity: str) -> bool:
        """Same-package guard: may *cur_activity*'s screen merge into *page*?

        True when the knob is off, when either package is unknown (abstain), or
        when they are equal. The guard is deliberately at PACKAGE level, not
        activity: the matcher is designed to merge one app's window labels
        together (a drawer over a map is the same page), and
        ``page_graph._canonical_activity`` already treats window labels as
        non-identity. Only a CROSS-APP merge is the error.
        """
        if not self._package_guard:
            return True
        cur_pkg = package_of(cur_activity)
        page_pkg = package_of(page.first_activity)
        if not cur_pkg or not page_pkg:
            logger.debug(
                f"screen_match: package guard abstains (cur={cur_pkg!r} page={page_pkg!r})"
            )
            return True
        return cur_pkg == page_pkg

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
        lines: list[str],
        current_feat: PILImage | None = None,
        is_canvas: bool = False,
        blind_lines: list[str] | None = None,
        activity: str = "",
    ) -> ScreenMatch:
        """Register a fresh page; add its element-line document to the BM25 corpus.

        The BM25 corpus keeps the UNBLINDED document: retrieval is not the
        bottleneck the canvas path fixes (map candidates already surface in the
        top-K), and blinding it would coarsen retrieval for every app. Only the
        canvas-pair VERIFICATION consults ``element_lines_blind``.
        """
        page_key = str(self._counter)
        self._counter += 1

        self._registry.add(
            PageKnowledge(
                page_key=page_key,
                element_lines=lines,
                is_canvas=is_canvas,
                element_lines_blind=list(blind_lines or []),
                first_activity=activity or "",
            )
        )
        self._bm25.add_document(page_key, lines)

        # Observation 0: a brand-new page has nothing to reuse. Also records the
        # first-sighting luminance fingerprint (no-op when the prefilter is off).
        obs_num, is_new_obs = self._record_observation(page_key, current_feat, allow_reuse=False)
        self._fp_to_key[fp_key] = (page_key, obs_num)
        logger.debug(f"screen_match: new page={page_key} lines={len(lines)}")

        return ScreenMatch(
            page_key, is_new_page=True, match_type="NEW",
            observation_num=obs_num, is_new_observation=is_new_obs,
        )
