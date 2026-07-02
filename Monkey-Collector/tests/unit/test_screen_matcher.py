"""Tests for ScreenMatcher: structural prefilter, NEW, BM25 merge, pixel gate.

Fixtures (element-line sets, id/bounds dropped):
    A = {button "Add", button "Search"}         |A|=2
    B = A + {button "Filter"}                    |B|=3   → |A △ B| = 1
    C = {button "Play"}                          |C|=1   → |A △ C| = 3

So a matcher built with ``element_diff_max=2`` MERGES B into A (1 < 2) but keeps
C separate (3 ≮ 2) — the calibration every fixture-based merge/new test uses.
"""

import io
import xml.etree.ElementTree as ET

from PIL import Image

from monkey_collector.llm.element_extractor import ExtractedElement
from monkey_collector.pipeline.screen_matching.screen_matcher import ScreenMatcher
from monkey_collector.xml.structured_parser import encode_with_bounds


def _btn(rid, desc, bounds):
    return (
        f'<node class="android.widget.Button" resource-id="com.x:id/{rid}" '
        f'content-desc="{desc}" text="" bounds="{bounds}" clickable="true"/>'
    )


def _screen(children):
    return (
        '<hierarchy rotation="0">'
        '<node class="android.widget.FrameLayout" bounds="[0,0][400,800]">'
        f"{children}</node></hierarchy>"
    )


# Distinct resource-ids → distinct structural fingerprints (so the prefilter
# does not short-circuit) while encoded element-lines use aria-label (id-free).
RAW_A = _screen(_btn("add", "Add", "[0,0][100,100]") + _btn("search", "Search", "[0,100][100,200]"))
RAW_B = _screen(
    _btn("add", "Add", "[0,0][100,100]")
    + _btn("search", "Search", "[0,100][100,200]")
    + _btn("filter", "Filter", "[0,200][100,300]")
)
RAW_C = _screen(_btn("play", "Play", "[0,0][100,100]"))


class FakeExtractor:
    """Returns one element per button (by aria-label) present in the XML."""

    def __init__(self):
        self.calls = 0

    def extract(self, encoded_xml, known_elements=None, screenshot_path=None):
        self.calls += 1
        root = ET.fromstring(encoded_xml)
        known = {e.name for e in (known_elements or [])}
        out = []
        for n in root.iter():
            if n.tag == "button" and n.attrib.get("index") and n.attrib.get("aria-label"):
                name = "open_" + n.attrib["aria-label"].lower()
                if name in known:
                    continue
                i = int(n.attrib["index"])
                out.append(
                    ExtractedElement(
                        name=name,
                        description=f"open {n.attrib['aria-label']}",
                        parameters={"idx": str(i)},
                        element_index=[i],
                        key_element_index=[i],
                    )
                )
        return out


def _enc(raw):
    return encode_with_bounds(raw)[0]


def _matcher(**kw):
    """Element-extraction matcher, luminance OFF, element_diff_max=2."""
    params = dict(cluster_merge_tolerance=0.2, max_expand_iters=3, element_diff_max=2)
    params.update(kw)
    return ScreenMatcher(FakeExtractor(), **params)


def _jpeg(color, size=(40, 80)):
    """Solid-colour JPEG bytes for the luminance gate (decoded via BytesIO)."""
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


def _lum_matcher(**kw):
    """Element-extraction matcher with the luminance prefilter ON."""
    params = dict(
        cluster_merge_tolerance=0.2, max_expand_iters=3, element_diff_max=2,
        luminance_prefilter=True, luminance_threshold=10,
        screenshot_diff_threshold=0.02, luminance_low_res_width=20,
    )
    params.update(kw)
    return ScreenMatcher(FakeExtractor(), **params)


SHOT_A = _jpeg((10, 20, 30))
SHOT_B = _jpeg((240, 230, 220))  # bright → very different luminance from SHOT_A


# ── NEW / structural prefilter ──


def test_new_page_on_empty_registry():
    sm = _matcher()
    m = sm.match(RAW_A, _enc(RAW_A), "act.Main")
    assert m.is_new_page and m.match_type == "NEW"
    assert {f.name for f in m.families} == {"open_add", "open_search"}
    # description/parameters extracted by the LLM ride through onto the family.
    fam = next(f for f in m.families if f.name == "open_add")
    assert fam.description == "open Add"
    assert fam.parameters == {"idx": str(fam.element_index[0])}
    # A brand-new page always starts at observation 0 (nothing to reuse yet).
    assert m.observation_num == 0
    assert m.is_new_observation
    # The page's element-line document is registered in the BM25 corpus.
    assert len(sm._bm25) == 1


def test_structural_prefilter_short_circuits_revisit():
    sm = _matcher()
    first = sm.match(RAW_A, _enc(RAW_A), "act.Main")
    calls_after_first = sm._extractor.calls
    m = sm.match(RAW_A, _enc(RAW_A), "act.Main")
    assert not m.is_new_page and m.match_type == "STRUCTURAL_IDENTICAL"
    # revisit fills families from the cached page, re-grounded on this screen.
    assert {f.name for f in m.families} == {"open_add", "open_search"}
    assert all(f.element_index and f.element_index == f.key_element_index for f in m.families)
    # no extra extractor call on an exact revisit (families come from the registry)
    assert sm._extractor.calls == calls_after_first
    # exact revisit reuses the first sighting's observation — no new write.
    assert not m.is_new_observation
    assert m.observation_num == first.observation_num


# ── BM25 merge vs new page (element criterion) ──


def test_bm25_merges_into_existing_page():
    sm = _matcher()
    a = sm.match(RAW_A, _enc(RAW_A), "act.Main")
    # |A △ B| = 1 < 2 → B merges into A's page.
    b = sm.match(RAW_B, _enc(RAW_B), "act.Main")
    assert not b.is_new_page
    assert b.match_type == "BM25_MERGE"
    assert b.page_key == a.page_key
    # merge fills families: the matched page's elements re-grounded on the current
    # screen (open_add, open_search) — NOT open_filter (no expand path anymore).
    assert {f.name for f in b.families} == {"open_add", "open_search"}
    assert all(f.element_index for f in b.families)
    # no luminance comparator (prefilter off) → always a new observation.
    assert b.is_new_observation
    assert b.observation_num != a.observation_num


def test_element_diff_over_threshold_creates_new_page():
    sm = _matcher()
    a = sm.match(RAW_A, _enc(RAW_A), "act.Main")
    # |A △ C| = 3 ≮ 2 → C is a genuinely new page even though BM25 offered A.
    c = sm.match(RAW_C, _enc(RAW_C), "act.Main")
    assert c.is_new_page and c.match_type == "NEW"
    assert c.page_key != a.page_key
    assert len(sm._bm25) == 2


def test_jaccard_criterion_merges_and_rejects():
    sm = _matcher(element_criterion="jaccard", element_jaccard_min=0.5, element_diff_max=99)
    a = sm.match(RAW_A, _enc(RAW_A), "act.Main")
    b = sm.match(RAW_B, _enc(RAW_B), "act.Main")   # Jaccard(A,B)=0.667 > 0.5 → merge
    assert b.match_type == "BM25_MERGE" and b.page_key == a.page_key
    c = sm.match(RAW_C, _enc(RAW_C), "act.Main")   # Jaccard(A,C)=0.0 ≤ 0.5 → new
    assert c.is_new_page and c.match_type == "NEW"


def test_bm25_selects_correct_candidate_among_multiple():
    sm = _matcher()
    a = sm.match(RAW_A, _enc(RAW_A), "act.Main")
    c = sm.match(RAW_C, _enc(RAW_C), "act.Main")  # new page (|A△C|=3)
    assert a.page_key != c.page_key
    # B shares 2 lines with A, 0 with C → BM25 ranks A first; |A△B|=1<2 → merge A.
    b = sm.match(RAW_B, _enc(RAW_B), "act.Main")
    assert b.match_type == "BM25_MERGE"
    assert b.page_key == a.page_key


# ── conjunctive pixel gate ──


def test_pixel_gate_allows_merge_on_matching_screenshot():
    sm = _lum_matcher()
    a = sm.match(RAW_A, _enc(RAW_A), "act.Main", screenshot=SHOT_A)
    # element passes (|A△B|=1<2) AND pixel matches (same solid colour) → merge.
    b = sm.match(RAW_B, _enc(RAW_B), "act.Main", screenshot=SHOT_A)
    assert b.match_type == "BM25_MERGE"
    assert b.page_key == a.page_key


def test_pixel_gate_blocks_merge_on_different_screenshot():
    sm = _lum_matcher()
    sm.match(RAW_A, _enc(RAW_A), "act.Main", screenshot=SHOT_A)
    # element passes (|A△B|=1<2) but the screenshot is very different → pixel gate
    # fails → NEW page (pixel is now a conjunctive requirement, not a shortcut).
    b = sm.match(RAW_B, _enc(RAW_B), "act.Main", screenshot=SHOT_B)
    assert b.is_new_page and b.match_type == "NEW"


def test_element_gate_blocks_merge_even_with_identical_screenshot():
    sm = _lum_matcher()
    a = sm.match(RAW_A, _enc(RAW_A), "act.Main", screenshot=SHOT_A)
    # identical screenshot, but |A△C|=3 ≮ 2 → element gate fails → NEW page.
    # (Under the old standalone luminance prefilter this would have merged.)
    c = sm.match(RAW_C, _enc(RAW_C), "act.Main", screenshot=SHOT_A)
    assert c.is_new_page and c.match_type == "NEW"
    assert c.page_key != a.page_key


def test_pixel_gate_abstains_without_screenshot():
    sm = _lum_matcher()
    a = sm.match(RAW_A, _enc(RAW_A), "act.Main", screenshot=SHOT_A)
    # No screenshot on the revisit → pixel gate abstains → element criterion alone
    # decides → merge (|A△B|=1<2).
    b = sm.match(RAW_B, _enc(RAW_B), "act.Main", screenshot=None)
    assert b.match_type == "BM25_MERGE"
    assert b.page_key == a.page_key


# ── pending guard ──


def test_no_interactables_is_pending_not_registered():
    sm = _matcher()
    raw_empty = _screen('<node class="android.widget.TextView" text="hi" bounds="[0,0][10,10]"/>')
    m = sm.match(raw_empty, _enc(raw_empty), "act.Main")
    assert m.pending
    assert not m.is_new_page
    assert m.match_type == "PENDING_EMPTY"
    assert m.families == []
    assert len(sm._registry) == 0
    assert sm._counter == 0
    assert sm._extractor.calls == 0
    assert len(sm._bm25) == 0


def test_pending_loading_then_first_valid_screen_is_page_0():
    sm = _matcher()
    raw_empty = _screen('<node class="android.widget.TextView" text="hi" bounds="[0,0][10,10]"/>')
    pending = sm.match(raw_empty, _enc(raw_empty), "act.Splash")
    assert pending.pending and len(sm._registry) == 0

    m = sm.match(RAW_A, _enc(RAW_A), "act.Main")
    assert m.is_new_page and m.match_type == "NEW"
    assert m.page_key == "0"
    assert {f.name for f in m.families} == {"open_add", "open_search"}


# ── reset / family helpers ──


def test_reset_clears_registry_and_bm25():
    sm = _matcher()
    sm.match(RAW_A, _enc(RAW_A), "act.Main")
    assert len(sm._registry) == 1 and len(sm._bm25) == 1
    sm.reset()
    assert len(sm._registry) == 0
    assert sm._counter == 0
    assert len(sm._bm25) == 0


def test_families_from_elements_copies_verbatim():
    from monkey_collector.pipeline.screen_matching.screen_matcher import (
        _families_from_elements,
    )

    els = [
        ExtractedElement(
            name="open_add",
            description="open Add",
            parameters={"idx": "3"},
            element_index=[3, 4],
            key_element_index=[3],
        )
    ]
    fams = _families_from_elements(els)
    assert len(fams) == 1
    f = fams[0]
    assert (f.name, f.description, f.parameters) == ("open_add", "open Add", {"idx": "3"})
    assert f.element_index == [3, 4]
    assert f.key_element_index == [3]


def test_remap_families_regrounds_on_current_screen():
    sm = _matcher()
    sm.match(RAW_A, _enc(RAW_A), "act.Main")
    page = sm._registry.get("0")
    tree = ET.fromstring(_enc(RAW_A))
    fams = sm._remap_families(tree, page)
    assert {f.name for f in fams} == {"open_add", "open_search"}
    assert all(f.element_index and f.element_index == f.key_element_index for f in fams)


def test_remap_families_drops_unrendered_elements():
    sm = _matcher()
    sm.match(RAW_A, _enc(RAW_A), "act.Main")
    page = sm._registry.get("0")
    tree_c = ET.fromstring(_enc(RAW_C))
    assert sm._remap_families(tree_c, page) == []


# ── observation identity ──


def test_structural_hit_returns_stored_observation_num():
    # A merge allocates a new observation (obs 1); an EXACT revisit of that same
    # screen must resolve back to obs 1 via the free structural cache — not obs 0.
    sm = _matcher()
    sm.match(RAW_A, _enc(RAW_A), "act.Main")
    merged = sm.match(RAW_B, _enc(RAW_B), "act.Main")
    assert merged.match_type == "BM25_MERGE"
    assert merged.observation_num != 0

    revisit = sm.match(RAW_B, _enc(RAW_B), "act.Main")
    assert revisit.match_type == "STRUCTURAL_IDENTICAL"
    assert revisit.observation_num == merged.observation_num
    assert not revisit.is_new_observation


def test_luminance_disabled_merge_always_allocates_new_observation():
    # With the prefilter off there is no pixel comparator, so every BM25 merge
    # with a distinct XML fingerprint allocates a new observation.
    sm = _matcher()  # luminance OFF
    sm.match(RAW_A, _enc(RAW_A), "act.Main")

    seen_obs = set()
    for i in range(3):
        raw = _screen(
            _btn("add", "Add", "[0,0][100,100]")
            + _btn("search", "Search", "[0,100][100,200]")
            + _btn(f"filter{i}", "Filter", "[0,200][100,300]")
        )
        m = sm.match(raw, _enc(raw), "act.Main")
        assert m.match_type == "BM25_MERGE"
        assert m.is_new_observation
        assert m.observation_num not in seen_obs
        seen_obs.add(m.observation_num)


def test_luminance_observations_capped():
    # persist_filtered ON: every BM25 merge allocates AND appends its (identical)
    # fingerprint, so the per-page luminance ring grows and is capped/evicted.
    sm = _lum_matcher(persist_filtered=True)
    sm.match(RAW_A, _enc(RAW_A), "act.Main", screenshot=SHOT_A)
    page = sm._registry.get("0")

    n = ScreenMatcher._MAX_LUMINANCE_OBS + 5
    for i in range(n):
        # Distinct fp each time (unique id bypasses the structural short-circuit),
        # identical element-lines {Add,Search,Filter} (|A△·|=1<2 → merge), and the
        # SAME solid screenshot so the pixel gate passes and the append lands.
        raw = _screen(
            _btn("add", "Add", "[0,0][100,100]")
            + _btn("search", "Search", "[0,100][100,200]")
            + _btn(f"filter{i}", "Filter", "[0,200][100,300]")
        )
        m = sm.match(raw, _enc(raw), "act.Main", screenshot=SHOT_A)
        assert m.match_type == "BM25_MERGE"
        assert m.is_new_observation

    assert len(page.luminance_features) == ScreenMatcher._MAX_LUMINANCE_OBS


# ── enrichment-off mode (extractor is None: LLM used for input text only) ──


def _extractorless_matcher(**kw):
    """A matcher with NO extractor — BM25 matching still runs, families empty."""
    params = dict(
        cluster_merge_tolerance=0.2, max_expand_iters=3, element_diff_max=2,
        luminance_prefilter=True, luminance_threshold=10,
        screenshot_diff_threshold=0.02, luminance_low_res_width=20,
    )
    params.update(kw)
    return ScreenMatcher(None, **params)


def test_extractorless_distinct_screens_do_not_collapse():
    sm = _extractorless_matcher()
    a = sm.match(RAW_A, _enc(RAW_A), "act.Main", screenshot=SHOT_A)
    c = sm.match(RAW_C, _enc(RAW_C), "act.Main", screenshot=SHOT_B)
    assert a.is_new_page and a.match_type == "NEW"
    assert c.is_new_page and c.match_type == "NEW"
    assert a.page_key != c.page_key
    assert len(sm._registry.all_page_keys()) == 2
    # No elements were extracted, so pages carry no families.
    assert a.families == [] and c.families == []


def test_extractorless_bm25_merge_reuses_page():
    sm = _extractorless_matcher()
    a = sm.match(RAW_A, _enc(RAW_A), "act.Main", screenshot=SHOT_A)
    # RAW_B merges into A by element-lines (|A△B|=1<2) + matching screenshot,
    # with NO extractor. Observation identity reuses obs 0 (same solid frame).
    m = sm.match(RAW_B, _enc(RAW_B), "act.Main", screenshot=SHOT_A)
    assert m.match_type == "BM25_MERGE"
    assert m.page_key == a.page_key
    assert not m.is_new_observation
    assert m.observation_num == a.observation_num
    assert m.families == []
    assert len(sm._registry.all_page_keys()) == 1


def test_extractorless_structural_revisit_reuses_observation():
    sm = _extractorless_matcher()
    first = sm.match(RAW_A, _enc(RAW_A), "act.Main", screenshot=SHOT_A)
    revisit = sm.match(RAW_A, _enc(RAW_A), "act.Main", screenshot=SHOT_A)
    assert revisit.match_type == "STRUCTURAL_IDENTICAL"
    assert not revisit.is_new_observation
    assert revisit.page_key == first.page_key
    assert revisit.observation_num == first.observation_num


# ── persist_filtered ON: a deduped revisit is saved as its own observation ──


def _persist_extract_matcher(**kw):
    return _lum_matcher(persist_filtered=True, **kw)


def _persist_extractorless_matcher(**kw):
    return _extractorless_matcher(persist_filtered=True, **kw)


def test_persist_filtered_structural_revisit_allocates_new_observation():
    sm = _persist_extract_matcher()
    first = sm.match(RAW_A, _enc(RAW_A), "act.Main", screenshot=SHOT_A)
    calls_after_first = sm._extractor.calls
    revisit = sm.match(RAW_A, _enc(RAW_A), "act.Main", screenshot=SHOT_A)
    assert revisit.match_type == "STRUCTURAL_IDENTICAL"
    assert revisit.page_key == first.page_key
    assert revisit.is_new_observation
    assert revisit.observation_num == first.observation_num + 1
    assert {f.name for f in revisit.families} == {"open_add", "open_search"}
    assert sm._extractor.calls == calls_after_first
    revisit2 = sm.match(RAW_A, _enc(RAW_A), "act.Main", screenshot=SHOT_A)
    assert revisit2.match_type == "STRUCTURAL_IDENTICAL"
    assert revisit2.observation_num == first.observation_num + 2
    assert sm._extractor.calls == calls_after_first


def test_persist_filtered_bm25_merge_allocates_new_observation():
    from monkey_collector.domain.page_graph import compute_xml_fingerprint

    sm = _persist_extract_matcher()
    a = sm.match(RAW_A, _enc(RAW_A), "act.Main", screenshot=SHOT_A)
    calls_after_first = sm._extractor.calls
    m = sm.match(RAW_B, _enc(RAW_B), "act.Main", screenshot=SHOT_A)
    assert m.match_type == "BM25_MERGE"
    assert m.page_key == a.page_key
    assert m.is_new_observation
    assert m.observation_num == a.observation_num + 1
    assert sm._extractor.calls == calls_after_first  # matching never calls the LLM
    fp_b = compute_xml_fingerprint(RAW_B)
    assert sm._fp_to_key[("act.Main", fp_b)] == (a.page_key, m.observation_num)
    assert len(sm._registry.all_page_keys()) == 1


def test_persist_filtered_extractorless_revisit_writes_new_observation():
    sm = _persist_extractorless_matcher()
    first = sm.match(RAW_A, _enc(RAW_A), "act.Main", screenshot=SHOT_A)
    r1 = sm.match(RAW_A, _enc(RAW_A), "act.Main", screenshot=SHOT_A)
    r2 = sm.match(RAW_A, _enc(RAW_A), "act.Main", screenshot=SHOT_A)
    assert first.observation_num == 0
    assert [r1.match_type, r2.match_type] == ["STRUCTURAL_IDENTICAL"] * 2
    assert r1.is_new_observation and r2.is_new_observation
    assert (r1.observation_num, r2.observation_num) == (1, 2)
    assert r1.families == [] and r2.families == []
    assert len(sm._registry.all_page_keys()) == 1
    assert sm._registry.get(first.page_key).next_observation_num == 3


def test_persist_filtered_structural_hit_does_not_append_luminance():
    # A structural revisit allocates a fresh observation but does NOT re-append
    # its (near-dup) fingerprint — the page's luminance ring must not churn.
    sm = _persist_extractorless_matcher()
    a = sm.match(RAW_A, _enc(RAW_A), "act.Main", screenshot=SHOT_A)
    page = sm._registry.get(a.page_key)
    assert len(page.luminance_features) == 1  # first sighting appended once
    for _ in range(5):
        sm.match(RAW_A, _enc(RAW_A), "act.Main", screenshot=SHOT_A)  # structural hits
    assert len(page.luminance_features) == 1
    assert page.next_observation_num == 6  # obs 0 + 5 revisits


def test_persist_filtered_bm25_merge_appends_luminance():
    # A BM25 merge keeps append_luma=True (unlike a structural/prefilter hit).
    sm = _persist_extract_matcher()
    sm.match(RAW_A, _enc(RAW_A), "act.Main", screenshot=SHOT_A)
    page = sm._registry.get("0")
    before = len(page.luminance_features)
    m = sm.match(RAW_B, _enc(RAW_B), "act.Main", screenshot=SHOT_A)
    assert m.match_type == "BM25_MERGE"
    assert m.is_new_observation
    assert m.observation_num != 0
    assert len(page.luminance_features) == before + 1


def test_persist_filtered_off_reuses_revisit_observation():
    # Regression guard: with persist_filtered=False (default) a structural revisit
    # and a pixel-matching BM25 merge both REUSE the first sighting's observation.
    sm = _lum_matcher(persist_filtered=False)
    first = sm.match(RAW_A, _enc(RAW_A), "act.Main", screenshot=SHOT_A)
    structural = sm.match(RAW_A, _enc(RAW_A), "act.Main", screenshot=SHOT_A)
    merge = sm.match(RAW_B, _enc(RAW_B), "act.Main", screenshot=SHOT_A)
    assert structural.match_type == "STRUCTURAL_IDENTICAL"
    assert merge.match_type == "BM25_MERGE"
    assert not structural.is_new_observation
    assert not merge.is_new_observation
    assert structural.observation_num == first.observation_num
    assert merge.observation_num == first.observation_num
