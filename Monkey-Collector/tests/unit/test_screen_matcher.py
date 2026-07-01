"""Tests for ScreenMatcher: prefilter, NEW, MERGE (superset), DISJOINT, expand."""

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
# does not short-circuit) while encoded anchors use aria-label.
RAW_A = _screen(_btn("add", "Add", "[0,0][100,100]") + _btn("search", "Search", "[0,100][100,200]"))
RAW_B = _screen(
    _btn("add", "Add", "[0,0][100,100]")
    + _btn("search", "Search", "[0,100][100,200]")
    + _btn("filter", "Filter", "[0,200][100,300]")
)
RAW_C = _screen(_btn("play", "Play", "[0,0][100,100]"))


class FakeExtractor:
    """Returns one element per button (by aria-label) present in the masked XML."""

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


def _matcher():
    return ScreenMatcher(FakeExtractor(), cluster_merge_tolerance=0.2, max_expand_iters=3)


def _jpeg(color, size=(40, 80)):
    """Solid-colour JPEG bytes for the luminance prefilter (decoded via BytesIO)."""
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


def _lum_matcher():
    return ScreenMatcher(
        FakeExtractor(), cluster_merge_tolerance=0.2, max_expand_iters=3,
        luminance_prefilter=True, luminance_threshold=10,
        screenshot_diff_threshold=0.02, luminance_low_res_width=20,
    )


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


def test_structural_prefilter_short_circuits_revisit():
    sm = _matcher()
    first = sm.match(RAW_A, _enc(RAW_A), "act.Main")
    calls_after_first = sm._extractor.calls
    m = sm.match(RAW_A, _enc(RAW_A), "act.Main")
    assert not m.is_new_page and m.match_type == "STRUCTURAL_IDENTICAL"
    # revisit fills families from the cached page, re-grounded on this screen.
    assert {f.name for f in m.families} == {"open_add", "open_search"}
    # indices are current-screen (anchor) indices, not empty.
    assert all(f.element_index and f.element_index == f.key_element_index for f in m.families)
    # no extra extractor call on an exact revisit (families come from the registry)
    assert sm._extractor.calls == calls_after_first
    # exact revisit reuses the first sighting's observation — no new write.
    assert not m.is_new_observation
    assert m.observation_num == first.observation_num


def test_superset_merges_into_existing_page():
    sm = _matcher()
    a = sm.match(RAW_A, _enc(RAW_A), "act.Main")
    # B = A's two buttons + Filter → distinct fp, anchors of A present → SUPERSET.
    b = sm.match(RAW_B, _enc(RAW_B), "act.Main")
    assert not b.is_new_page
    assert b.match_type == "SUPERSET_MERGE"
    assert b.page_key == a.page_key  # merged into the same page
    # merge fills families: the matched page's elements re-grounded on the
    # current screen (open_add, open_search) PLUS the expand-found open_filter.
    assert {f.name for f in b.families} == {"open_add", "open_search", "open_filter"}
    assert all(f.element_index for f in b.families)
    # a visually/structurally distinct render with no luminance comparator
    # (prefilter off on the plain _matcher()) always allocates a new observation.
    assert b.is_new_observation
    assert b.observation_num != a.observation_num


def test_disjoint_creates_new_page():
    sm = _matcher()
    a = sm.match(RAW_A, _enc(RAW_A), "act.Main")
    c = sm.match(RAW_C, _enc(RAW_C), "act.Main")
    assert c.is_new_page and c.match_type == "DISJOINT"
    assert c.page_key != a.page_key


def test_expand_discovers_new_element_via_mask():
    sm = _matcher()
    sm.match(RAW_A, _enc(RAW_A), "act.Main")
    # On B, step-1 supports add/search; expand must surface open_filter.
    b = sm.match(RAW_B, _enc(RAW_B), "act.Main")
    assert b.match_type == "SUPERSET_MERGE"  # add,search,filter ⊋ add,search


def test_no_interactables_is_pending_not_registered():
    # A loading/splash frame with no button/input must be declined (pending),
    # not registered as an empty page — otherwise it becomes a blackhole that
    # every later screen merges into. No extractor (LLM) call is made.
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


def test_pending_loading_then_first_valid_screen_is_page_0():
    # The first VALID screen after a declined loading frame must become page_0
    # (the loading frame left the registry empty).
    sm = _matcher()
    raw_empty = _screen('<node class="android.widget.TextView" text="hi" bounds="[0,0][10,10]"/>')
    pending = sm.match(raw_empty, _enc(raw_empty), "act.Splash")
    assert pending.pending and len(sm._registry) == 0

    m = sm.match(RAW_A, _enc(RAW_A), "act.Main")
    assert m.is_new_page and m.match_type == "NEW"
    assert m.page_key == "page_0"
    assert {f.name for f in m.families} == {"open_add", "open_search"}


def test_reset_clears_registry():
    sm = _matcher()
    sm.match(RAW_A, _enc(RAW_A), "act.Main")
    assert len(sm._registry) == 1
    sm.reset()
    assert len(sm._registry) == 0
    assert sm._counter == 0


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
    # Register page_0 from RAW_A (open_add, open_search anchored on this screen).
    sm.match(RAW_A, _enc(RAW_A), "act.Main")
    page = sm._registry.get("page_0")

    # Re-ground on RAW_A's own tree: both elements present → current indices,
    # element_index approximated to (== ) the anchor indices.
    tree = ET.fromstring(_enc(RAW_A))
    fams = sm._remap_families(tree, page)
    assert {f.name for f in fams} == {"open_add", "open_search"}
    assert all(f.element_index and f.element_index == f.key_element_index for f in fams)


def test_remap_families_drops_unrendered_elements():
    sm = _matcher()
    sm.match(RAW_A, _enc(RAW_A), "act.Main")
    page = sm._registry.get("page_0")
    # RAW_C (Play only) has none of page_0's anchors → every element dropped.
    tree_c = ET.fromstring(_enc(RAW_C))
    assert sm._remap_families(tree_c, page) == []


# ── Stage-0 luminance prefilter ──

SHOT_A = _jpeg((10, 20, 30))
SHOT_B = _jpeg((240, 230, 220))  # bright → very different luminance from SHOT_A


def test_luminance_prefilter_hit_reuses_page_no_llm():
    sm = _lum_matcher()
    a = sm.match(RAW_A, _enc(RAW_A), "act.Main", screenshot=SHOT_A)
    calls_after_first = sm._extractor.calls
    # RAW_C would normally DISJOINT into a NEW page; the identical screenshot makes
    # the luminance prefilter short-circuit it to page_0 with no extractor call.
    m = sm.match(RAW_C, _enc(RAW_C), "act.Main", screenshot=SHOT_A)
    assert not m.is_new_page
    assert m.match_type == "LUMINANCE_PREFILTER"
    assert m.page_key == a.page_key
    assert sm._extractor.calls == calls_after_first  # LLM/expand path skipped
    # reuses the existing observation — no new write for the caller to persist.
    assert not m.is_new_observation
    assert m.observation_num == a.observation_num


def test_luminance_prefilter_miss_falls_through():
    sm = _lum_matcher()
    sm.match(RAW_A, _enc(RAW_A), "act.Main", screenshot=SHOT_A)
    # A very different screenshot → no luminance hit → normal element matching.
    c = sm.match(RAW_C, _enc(RAW_C), "act.Main", screenshot=SHOT_B)
    assert c.is_new_page and c.match_type == "DISJOINT"


def test_luminance_prefilter_off_is_noop():
    sm = _matcher()  # luminance_prefilter defaults OFF
    sm.match(RAW_A, _enc(RAW_A), "act.Main", screenshot=SHOT_A)
    # Even with an identical screenshot, OFF routes RAW_B through element matching.
    b = sm.match(RAW_B, _enc(RAW_B), "act.Main", screenshot=SHOT_A)
    assert b.match_type == "SUPERSET_MERGE"


def test_luminance_prefilter_none_screenshot_degrades():
    sm = _lum_matcher()
    sm.match(RAW_A, _enc(RAW_A), "act.Main", screenshot=SHOT_A)
    # Enabled but no screenshot bytes → no image work, normal element path.
    c = sm.match(RAW_C, _enc(RAW_C), "act.Main", screenshot=None)
    assert c.is_new_page and c.match_type == "DISJOINT"


def test_luminance_observations_capped():
    # An exact structural (_fp_to_key) repeat no longer appends anything (reuse
    # means no new observation) — driving eviction now requires many distinct
    # SUPERSET_MERGE visits, each visually distinct enough that the page-scoped
    # luminance lookup misses and a new observation is allocated every time.
    sm = _lum_matcher()
    sm.match(RAW_A, _enc(RAW_A), "act.Main", screenshot=SHOT_A)
    page = sm._registry.get("page_0")

    n = ScreenMatcher._MAX_LUMINANCE_OBS + 5
    for i in range(n):
        # A uniquely-id'd extra button keeps the XML fingerprint distinct each
        # time (bypassing the free structural short-circuit) while staying a
        # superset of RAW_A's anchors (add/search), so classify() always merges
        # into page_0 instead of forking a new page.
        raw = _screen(
            _btn("add", "Add", "[0,0][100,100]")
            + _btn("search", "Search", "[0,100][100,200]")
            + _btn(f"filter{i}", "Filter", "[0,200][100,300]")
        )
        # Solid grays spaced 15 luma levels apart, starting well clear of
        # SHOT_A's luma (~18, from RGB (10,20,30)) so none of them accidentally
        # luminance-match the very first stored observation via the GLOBAL
        # Stage-0c lookup before classify() even runs.
        shot = _jpeg((40 + i * 15, 40 + i * 15, 40 + i * 15))
        m = sm.match(raw, _enc(raw), "act.Main", screenshot=shot)
        assert m.match_type == "SUPERSET_MERGE"
        assert m.is_new_observation

    assert len(page.luminance_features) == ScreenMatcher._MAX_LUMINANCE_OBS


def test_structural_hit_returns_stored_observation_num():
    # A merge allocates a new observation (obs 1); an EXACT revisit of that same
    # screen must resolve back to obs 1 via the free structural cache — not
    # obs 0 (the page's first-sighting observation).
    sm = _matcher()
    sm.match(RAW_A, _enc(RAW_A), "act.Main")
    merged = sm.match(RAW_B, _enc(RAW_B), "act.Main")
    assert merged.match_type == "SUPERSET_MERGE"
    assert merged.observation_num != 0

    revisit = sm.match(RAW_B, _enc(RAW_B), "act.Main")
    assert revisit.match_type == "STRUCTURAL_IDENTICAL"
    assert revisit.observation_num == merged.observation_num
    assert not revisit.is_new_observation


def test_luminance_disabled_merge_always_allocates_new_observation():
    # With the prefilter off there is no pixel comparator to dedupe against, so
    # every classify-merge with a distinct XML fingerprint must allocate a new
    # observation — never silently reuse one (that would risk collapsing
    # genuinely different render states with no visual evidence they match).
    sm = _matcher()  # luminance_prefilter defaults OFF
    sm.match(RAW_A, _enc(RAW_A), "act.Main")

    seen_obs = set()
    for i in range(3):
        raw = _screen(
            _btn("add", "Add", "[0,0][100,100]")
            + _btn("search", "Search", "[0,100][100,200]")
            + _btn(f"filter{i}", "Filter", "[0,200][100,300]")
        )
        m = sm.match(raw, _enc(raw), "act.Main")
        assert m.match_type == "SUPERSET_MERGE"
        assert m.is_new_observation
        assert m.observation_num not in seen_obs
        seen_obs.add(m.observation_num)
