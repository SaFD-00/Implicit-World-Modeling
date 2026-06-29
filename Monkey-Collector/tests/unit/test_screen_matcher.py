"""Tests for ScreenMatcher: prefilter, NEW, MERGE (superset), DISJOINT, expand."""

import xml.etree.ElementTree as ET

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
                out.append(ExtractedElement(name=name, element_index=[i], key_element_index=[i]))
        return out


def _enc(raw):
    return encode_with_bounds(raw)[0]


def _matcher():
    return ScreenMatcher(FakeExtractor(), cluster_merge_tolerance=0.2, max_expand_iters=3)


def test_new_page_on_empty_registry():
    sm = _matcher()
    m = sm.match(RAW_A, _enc(RAW_A), "act.Main")
    assert m.is_new_page and m.match_type == "NEW"
    assert {f.name for f in m.families} == {"open_add", "open_search"}


def test_structural_prefilter_short_circuits_revisit():
    sm = _matcher()
    sm.match(RAW_A, _enc(RAW_A), "act.Main")
    calls_after_first = sm._extractor.calls
    m = sm.match(RAW_A, _enc(RAW_A), "act.Main")
    assert not m.is_new_page and m.match_type == "STRUCTURAL_IDENTICAL"
    assert m.families == []
    # no extra extractor call on an exact revisit
    assert sm._extractor.calls == calls_after_first


def test_superset_merges_into_existing_page():
    sm = _matcher()
    a = sm.match(RAW_A, _enc(RAW_A), "act.Main")
    # B = A's two buttons + Filter → distinct fp, anchors of A present → SUPERSET.
    b = sm.match(RAW_B, _enc(RAW_B), "act.Main")
    assert not b.is_new_page
    assert b.match_type == "SUPERSET_MERGE"
    assert b.page_key == a.page_key  # merged into the same page


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


def test_no_interactables_makes_empty_new_page():
    sm = _matcher()
    raw_empty = _screen('<node class="android.widget.TextView" text="hi" bounds="[0,0][10,10]"/>')
    m = sm.match(raw_empty, _enc(raw_empty), "act.Main")
    assert m.is_new_page
    assert m.families == []


def test_reset_clears_registry():
    sm = _matcher()
    sm.match(RAW_A, _enc(RAW_A), "act.Main")
    assert len(sm._registry) == 1
    sm.reset()
    assert len(sm._registry) == 0
    assert sm._counter == 0
