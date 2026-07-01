"""Tests for pipeline.screen_matching.rehydrate — resume rehydration."""

import io
import xml.etree.ElementTree as ET

from PIL import Image

from monkey_collector.llm.element_extractor import ExtractedElement
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


class FakeExtractor:
    """Returns one element per button (by aria-label) present in the masked XML."""

    def extract(self, encoded_xml, known_elements=None, screenshot_path=None):
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
                        name=name, description=f"open {n.attrib['aria-label']}",
                        parameters={}, element_index=[i], key_element_index=[i],
                    )
                )
        return out


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
    matcher = ScreenMatcher(FakeExtractor())
    a = _drive_and_persist(matcher, writer, RAW_A)
    b = _drive_and_persist(matcher, writer, RAW_B)  # SUPERSET_MERGE into a's page

    fresh = ScreenMatcher(FakeExtractor())
    rehydrate_screen_matcher(fresh, writer)

    assert fresh._registry.all_page_keys() == [a.page_key]
    page = fresh._registry.get(a.page_key)
    assert page is not None
    assert page.element_names == {"open_add", "open_search"}
    assert page.next_observation_num == b.observation_num + 1

    # Exact structural revisit of RAW_B resolves via the rehydrated cache.
    revisit = fresh.match(RAW_B, _enc(RAW_B), "act.Main")
    assert revisit.match_type == "STRUCTURAL_IDENTICAL"
    assert revisit.observation_num == b.observation_num

    # _counter is past the highest rehydrated page index, so a genuinely new
    # page can't collide with a's page_key.
    assert fresh._counter == int(a.page_key.removeprefix("page_")) + 1


def test_rehydrate_empty_tree_is_noop(tmp_path):
    writer = _writer(tmp_path)

    fresh = ScreenMatcher(FakeExtractor())
    rehydrate_screen_matcher(fresh, writer)

    assert len(fresh._registry) == 0
    assert fresh._counter == 0


def test_rehydrate_with_luminance_enabled_restores_features(tmp_path):
    writer = _writer(tmp_path)
    matcher = ScreenMatcher(
        FakeExtractor(), luminance_prefilter=True, luminance_threshold=10,
        screenshot_diff_threshold=0.02, luminance_low_res_width=20,
    )
    shot = _jpeg((10, 20, 30))
    a = _drive_and_persist(matcher, writer, RAW_A, screenshot=shot)

    fresh = ScreenMatcher(
        FakeExtractor(), luminance_prefilter=True, luminance_threshold=10,
        screenshot_diff_threshold=0.02, luminance_low_res_width=20,
    )
    rehydrate_screen_matcher(fresh, writer)

    # A near-identical screenshot, under a completely different XML
    # fingerprint, should luminance-hit the rehydrated observation.
    hit = fresh.match(RAW_C, _enc(RAW_C), "act.Main", screenshot=shot)
    assert hit.match_type == "LUMINANCE_PREFILTER"
    assert hit.page_key == a.page_key
    assert hit.observation_num == a.observation_num


def test_rehydrate_skips_page_with_no_observations_gracefully(tmp_path):
    # A page.json with an empty pages/{page_key}/ directory (no observation
    # subdirs) must not crash rehydration — next_observation_num falls back to 0.
    writer = _writer(tmp_path)
    from monkey_collector.pipeline.screen_matching.page_knowledge import (
        PageKnowledge,
    )
    writer.save_page_knowledge("page_0", PageKnowledge(page_key="page_0"))

    fresh = ScreenMatcher(FakeExtractor())
    rehydrate_screen_matcher(fresh, writer)

    page = fresh._registry.get("page_0")
    assert page is not None
    assert page.next_observation_num == 0
    assert fresh._counter == 1
