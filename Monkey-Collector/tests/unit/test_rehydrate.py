"""Tests for pipeline.screen_matching.rehydrate — resume rehydration."""

import io

from PIL import Image

from monkey_collector.pipeline.screen_matching.page_knowledge import PageKnowledge
from monkey_collector.pipeline.screen_matching.rehydrate import (
    rehydrate_screen_matcher,
)
from monkey_collector.pipeline.screen_matching.screen_matcher import ScreenMatcher
from monkey_collector.storage import DataWriter
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


RAW_A = _screen(_btn("add", "Add", "[0,0][100,100]") + _btn("search", "Search", "[0,100][100,200]"))
RAW_B = _screen(
    _btn("add", "Add", "[0,0][100,100]")
    + _btn("search", "Search", "[0,100][100,200]")
    + _btn("filter", "Filter", "[0,200][100,300]")
)
RAW_C = _screen(_btn("play", "Play", "[0,0][100,100]"))


def _enc(raw):
    return encode_with_bounds(raw)[0]


def _jpeg(color, size=(20, 40)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


def _drive_and_persist(matcher, writer, raw, activity="act.Main", screenshot=None):
    """One match() call, persisting a new page/observation to disk exactly as
    collection_loop.py's wiring does — the caller only writes when the match
    says so."""
    match = matcher.match(raw, _enc(raw), activity, screenshot=screenshot)
    if match.is_new_page:
        writer.save_page_knowledge(match.page_key, matcher.get_page_knowledge(match.page_key))
    if match.is_new_observation:
        writer.save_observation(
            match.page_key, match.observation_num, screenshot, raw,
            match=match, activity=activity,
        )
    return match


def _writer(tmp_path):
    w = DataWriter(data_dir=str(tmp_path / "data"), runtime_dir=str(tmp_path / "runtime"))
    w.init_session("com.test.app", "com.test.app")
    return w


def test_rehydrate_restores_registry_and_structural_cache(tmp_path):
    writer = _writer(tmp_path)
    matcher = ScreenMatcher()
    a = _drive_and_persist(matcher, writer, RAW_A)
    b = _drive_and_persist(matcher, writer, RAW_B)  # BM25 merge into a's page

    fresh = ScreenMatcher()
    rehydrate_screen_matcher(fresh, writer)

    assert fresh._registry.all_page_keys() == [a.page_key]
    assert len(fresh._bm25) == 1  # BM25 corpus rebuilt from element_lines
    page = fresh._registry.get(a.page_key)
    assert page is not None
    assert page.next_observation_num == b.observation_num + 1

    # Exact structural revisit of RAW_B resolves via the rehydrated cache.
    revisit = fresh.match(RAW_B, _enc(RAW_B), "act.Main")
    assert revisit.match_type == "STRUCTURAL_IDENTICAL"
    assert revisit.observation_num == b.observation_num

    # _counter is past the highest rehydrated page index.
    assert fresh._counter == int(a.page_key) + 1


def test_rehydrate_empty_tree_is_noop(tmp_path):
    writer = _writer(tmp_path)

    fresh = ScreenMatcher()
    rehydrate_screen_matcher(fresh, writer)

    assert len(fresh._registry) == 0
    assert len(fresh._bm25) == 0
    assert fresh._counter == 0


def test_rehydrate_rebuilds_bm25_corpus(tmp_path):
    # Two genuinely distinct pages (element_diff_max=2 keeps RAW_C separate from
    # RAW_A); after resume the BM25 corpus has both, and a merge picks the right
    # candidate.
    writer = _writer(tmp_path)
    matcher = ScreenMatcher(element_diff_max=2)
    a = _drive_and_persist(matcher, writer, RAW_A)
    _drive_and_persist(matcher, writer, RAW_C)
    assert len(matcher._registry.all_page_keys()) == 2

    fresh = ScreenMatcher(element_diff_max=2)
    rehydrate_screen_matcher(fresh, writer)
    assert len(fresh._bm25) == 2
    assert sorted(fresh._registry.all_page_keys()) == ["0", "1"]

    # RAW_B shares two element-lines with A, none with C → BM25 ranks A first,
    # |A△B|=1<2 → merges into A's page after the resume.
    b = fresh.match(RAW_B, _enc(RAW_B), "act.Main")
    assert b.match_type == "BM25_MERGE"
    assert b.page_key == a.page_key


def test_rehydrate_legacy_page_json_without_element_lines(tmp_path):
    # A page.json written before element_lines existed loads with an empty list;
    # rehydrate rebuilds the BM25 document from the first observation's raw.xml.
    writer = _writer(tmp_path)
    writer.save_page_knowledge("0", PageKnowledge(page_key="0"))  # element_lines=[]
    writer.save_observation("0", 0, None, RAW_A, match=None, activity="act.Main")

    fresh = ScreenMatcher(element_diff_max=2)
    rehydrate_screen_matcher(fresh, writer)

    page = fresh._registry.get("0")
    assert page is not None
    assert page.element_lines  # rebuilt from raw.xml, not left empty
    assert len(fresh._bm25) == 1

    # A BM25 merge now resolves against the rebuilt document.
    b = fresh.match(RAW_B, _enc(RAW_B), "act.Main")
    assert b.match_type == "BM25_MERGE"
    assert b.page_key == "0"


def test_rehydrate_with_luminance_restores_features(tmp_path):
    writer = _writer(tmp_path)
    matcher = ScreenMatcher(
        luminance_prefilter=True, luminance_threshold=10,
        screenshot_diff_threshold=0.02, luminance_low_res_width=20,
    )
    shot = _jpeg((10, 20, 30))
    a = _drive_and_persist(matcher, writer, RAW_A, screenshot=shot)

    fresh = ScreenMatcher(
        luminance_prefilter=True, luminance_threshold=10,
        screenshot_diff_threshold=0.02, luminance_low_res_width=20,
    )
    rehydrate_screen_matcher(fresh, writer)

    page = fresh._registry.get(a.page_key)
    assert len(page.luminance_features) == 1  # re-derived from the saved screenshot
    assert page.element_lines                 # restored from page.json

    # Element-passing revisit (|A△B|=1<5) with the SAME screenshot merges; the
    # restored fingerprint drives OBSERVATION dedup → reuses obs 0.
    m = fresh.match(RAW_B, _enc(RAW_B), "act.Main", screenshot=shot)
    assert m.match_type == "BM25_MERGE"
    assert m.page_key == a.page_key
    assert not m.is_new_observation
    assert m.observation_num == a.observation_num


def test_rehydrate_skips_page_with_no_observations_gracefully(tmp_path):
    # A page.json with an empty pages/{page_key}/ directory (no observation
    # subdirs) must not crash rehydration — next_observation_num falls back to 0.
    writer = _writer(tmp_path)
    writer.save_page_knowledge("0", PageKnowledge(page_key="0"))

    fresh = ScreenMatcher()
    rehydrate_screen_matcher(fresh, writer)

    page = fresh._registry.get("0")
    assert page is not None
    assert page.next_observation_num == 0
    assert fresh._counter == 1


def _persist_lum_matcher():
    return ScreenMatcher(
        luminance_prefilter=True, luminance_threshold=10,
        screenshot_diff_threshold=0.02, luminance_low_res_width=20,
        persist_filtered=True,
    )


def test_rehydrate_after_persisted_revisits_continues_chain(tmp_path):
    # With persist_filtered on, a structural revisit is written as its own
    # observation dir (per-visit chain). After a resume, rehydration recovers
    # next_observation_num = max(on-disk obs)+1 and the first post-resume revisit
    # continues the chain with no gap or overwrite.
    writer = _writer(tmp_path)
    shot = _jpeg((10, 20, 30))
    matcher = _persist_lum_matcher()
    a = _drive_and_persist(matcher, writer, RAW_A, screenshot=shot)   # obs 0
    r1 = _drive_and_persist(matcher, writer, RAW_A, screenshot=shot)  # obs 1 (structural)
    assert (a.observation_num, r1.observation_num) == (0, 1)
    assert r1.match_type == "STRUCTURAL_IDENTICAL" and r1.is_new_observation
    assert writer.list_observations(a.page_key) == [0, 1]  # both on disk

    fresh = _persist_lum_matcher()
    rehydrate_screen_matcher(fresh, writer)
    assert fresh._registry.get(a.page_key).next_observation_num == 2

    # First post-resume revisit persists obs 2, extending the chain.
    r2 = _drive_and_persist(fresh, writer, RAW_A, screenshot=shot)
    assert r2.match_type == "STRUCTURAL_IDENTICAL" and r2.is_new_observation
    assert r2.observation_num == 2
    assert writer.list_observations(a.page_key) == [0, 1, 2]
