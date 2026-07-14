"""Tests for canvas-gated text-blind page matching (S-9 map fragmentation).

Fixtures: MAP_1 / MAP_2 are one map screen before and after a pan — a full-screen
clickable leaf surface, a zoom button, and text-only readouts (scale, distance,
address, ETA) that the pan rewrites and re-counts. Their element-line sets differ
by 7 lines (3 + 4), well past the default ``element_diff_max=5``, so the pre-fix
matcher mints a second page. Blinding the text collapses every readout to
``<p></p>``, which set-dedups, leaving the two documents identical.

FLAT_1 / FLAT_2 are the same two screens with the surface removed: same readout
difference, no canvas → the canvas path must not touch them.
"""

import io
import json

from PIL import Image

from monkey_collector.pipeline.screen_matching.canvas import is_canvas_screen
from monkey_collector.pipeline.screen_matching.element_lines import serialize_element_lines
from monkey_collector.pipeline.screen_matching.page_knowledge import PageKnowledge
from monkey_collector.pipeline.screen_matching.rehydrate import rehydrate_screen_matcher
from monkey_collector.pipeline.screen_matching.screen_matcher import ScreenMatcher
from monkey_collector.storage import DataWriter
from monkey_collector.xml.structured_parser import encode_with_bounds

# ── fixtures ────────────────────────────────────────────────────────────────

SCREEN_BOUNDS = "[0,0][400,800]"  # area 320_000


def _screen(children, root=SCREEN_BOUNDS):
    return (
        '<hierarchy rotation="0">'
        f'<node class="android.widget.FrameLayout" bounds="{root}">{children}</node>'
        "</hierarchy>"
    )


def _surface(bounds="[0,0][400,780]", clickable="true", long_clickable="false"):
    """The map surface as the a11y tree really carries it: a LEAF plain View,
    clickable, covering ~all of the screen (osmand's covers 0.97). No
    SurfaceView/TextureView class ever appears — hence the bounds-based test."""
    return (
        f'<node class="android.view.View" content-desc="Map" text="" bounds="{bounds}" '
        f'clickable="{clickable}" long-clickable="{long_clickable}"/>'
    )


ZOOM_BTN = (
    '<node class="android.widget.Button" resource-id="com.x:id/zoom" content-desc="Zoom in" '
    'text="" bounds="[300,700][360,760]" clickable="true"/>'
)


def _readouts(*texts):
    """The floating map widgets: text-only, no content-desc — all identity in
    the text, which is exactly what a pan rewrites. A pan also adds/drops
    widgets, so the two screens differ structurally too (else the structural
    pre-filter would short-circuit them as an exact revisit before BM25)."""
    return "".join(
        f'<node class="android.widget.TextView" resource-id="com.x:id/t{i}" content-desc="" '
        f'text="{t}" bounds="[0,{i * 20}][100,{i * 20 + 20}]" clickable="false"/>'
        for i, t in enumerate(texts, start=1)
    )


READOUTS_1 = _readouts("100 ft", "0.25 mi", "Near A")
READOUTS_2 = _readouts("200 ft", "5666 mi", "Near B", "ETA 5 min")

MAP_1 = _screen(_surface() + ZOOM_BTN + READOUTS_1)
MAP_2 = _screen(_surface() + ZOOM_BTN + READOUTS_2)
FLAT_1 = _screen(ZOOM_BTN + READOUTS_1)
FLAT_2 = _screen(ZOOM_BTN + READOUTS_2)


def _enc(raw):
    return encode_with_bounds(raw)[0]


def _matcher(**kw):
    """Default-threshold matcher (element_diff_max=5, as shipped), luminance off."""
    return ScreenMatcher(None, luminance_prefilter=True, **kw)


def _jpeg(color, size=(40, 80)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


SHOT_DARK = _jpeg((10, 20, 30))
SHOT_BRIGHT = _jpeg((240, 230, 220))  # a repainted map: fails any pixel gate


def _drive(matcher, raw, activity="net.osmand/.MapActivity", screenshot=None):
    return matcher.match(raw, _enc(raw), activity, screenshot=screenshot)


# ── detector ────────────────────────────────────────────────────────────────


def test_detector_map_surface_is_canvas():
    assert is_canvas_screen(MAP_1, 0.7) is True


def test_detector_full_screen_scroll_list_is_not_canvas():
    # A full-screen RecyclerView is scrollable, not clickable, and holds its
    # rows (not a leaf) — the false positive the criterion must not make.
    rows = _readouts("Row A", "Row B", "Row C")
    lst = (
        '<node class="androidx.recyclerview.widget.RecyclerView" bounds="[0,0][400,790]" '
        f'scrollable="true" clickable="false">{rows}</node>'
    )
    assert is_canvas_screen(_screen(lst + ZOOM_BTN), 0.7) is False


def test_detector_parse_failure_is_not_canvas():
    assert is_canvas_screen("<hierarchy rotation=", 0.7) is False
    assert is_canvas_screen("", 0.7) is False


def test_detector_area_threshold_is_inclusive_boundary():
    # 400x560 = 224_000 = exactly 0.7 of the 320_000 screen → in.
    at = _screen(_surface(bounds="[0,0][400,560]") + ZOOM_BTN)
    # One pixel shorter → 0.6994 → out.
    below = _screen(_surface(bounds="[0,0][400,559]") + ZOOM_BTN)
    assert is_canvas_screen(at, 0.7) is True
    assert is_canvas_screen(below, 0.7) is False


def test_detector_long_clickable_surface_counts():
    raw = _screen(_surface(clickable="false", long_clickable="true") + ZOOM_BTN)
    assert is_canvas_screen(raw, 0.7) is True


def test_detector_non_interactive_surface_is_not_canvas():
    raw = _screen(_surface(clickable="false") + ZOOM_BTN)
    assert is_canvas_screen(raw, 0.7) is False


# ── blind serialization ─────────────────────────────────────────────────────


def test_blind_text_empties_text_and_keeps_attributes():
    lines = serialize_element_lines(_enc(MAP_1), blind_text=True)
    assert lines == [
        '<button aria-label="Map"></button>',
        '<button aria-label="Zoom in"></button>',
        "<p></p>",
        "<p></p>",
        "<p></p>",
    ]
    # The unblinded document is untouched (default arg).
    assert '<p>100 ft</p>' in serialize_element_lines(_enc(MAP_1))


def test_blind_projection_makes_two_pans_of_one_map_identical():
    # As SETS — which is what the criterion compares. The blinded documents are
    # still different LENGTHS (the pan added a readout); it is the set-dedup of
    # the now-identical <p></p> lines that makes the two screens one page.
    assert set(serialize_element_lines(_enc(MAP_1), blind_text=True)) == set(
        serialize_element_lines(_enc(MAP_2), blind_text=True)
    )


# ── both-sides gating ───────────────────────────────────────────────────────


def test_canvas_pair_merges_on_blind_criterion():
    sm = _matcher()
    a = _drive(sm, MAP_1)
    b = _drive(sm, MAP_2)
    assert a.is_new_page is True
    assert b.is_new_page is False  # |A △ B| = 7 unblinded (would mint), 0 blinded
    assert b.match_type == "BM25_MERGE"
    assert b.page_key == a.page_key


def test_knob_off_keeps_the_pre_fix_decision():
    sm = _matcher(canvas_merge=False)
    a = _drive(sm, MAP_1)
    b = _drive(sm, MAP_2)
    assert a.is_new_page is True
    assert b.is_new_page is True  # 7 differing lines ≥ element_diff_max=5


def test_non_canvas_pair_is_untouched_by_the_knob():
    # Same text difference, no full-screen surface → not a canvas pair, so the
    # blind criterion must NOT apply even with the knob on.
    on, off = _matcher(), _matcher(canvas_merge=False)
    assert _drive(on, FLAT_1).is_new_page is True
    assert _drive(on, FLAT_2).is_new_page is True
    assert _drive(off, FLAT_1).is_new_page is True
    assert _drive(off, FLAT_2).is_new_page is True


def test_mixed_pair_needs_both_sides_canvas():
    # A canvas screen must not blind-merge into a NON-canvas stored page.
    sm = _matcher()
    flat = _drive(sm, FLAT_1)
    mapped = _drive(sm, MAP_2)
    assert flat.is_new_page is True
    assert mapped.is_new_page is True
    assert mapped.page_key != flat.page_key


def test_canvas_pair_abstains_from_the_pixel_gate():
    # A repainted map (dark → bright) fails any pixel gate; the canvas pair must
    # merge anyway, since the two relaxations are one mechanism.
    sm = _matcher(luminance_low_res_width=20)
    a = _drive(sm, MAP_1, screenshot=SHOT_DARK)
    b = _drive(sm, MAP_2, screenshot=SHOT_BRIGHT)
    assert a.is_new_page is True
    assert b.is_new_page is False
    # With the knob off the same pair mints (element criterion blocks first).
    off = _matcher(luminance_low_res_width=20, canvas_merge=False)
    _drive(off, MAP_1, screenshot=SHOT_DARK)
    assert _drive(off, MAP_2, screenshot=SHOT_BRIGHT).is_new_page is True


def test_minted_page_stores_canvas_flag_and_blind_lines_regardless_of_knob():
    for knob in (True, False):
        sm = _matcher(canvas_merge=knob)
        page = sm.get_page_knowledge(_drive(sm, MAP_1).page_key)
        flat = sm.get_page_knowledge(_drive(sm, FLAT_2).page_key)
        assert page.is_canvas is True
        assert page.element_lines_blind == serialize_element_lines(_enc(MAP_1), blind_text=True)
        assert flat.is_canvas is False


# ── persistence: page.json round-trip + legacy resume ───────────────────────


def test_page_knowledge_round_trip_carries_canvas_fields():
    page = PageKnowledge(
        page_key="0",
        element_lines=['<p>100 ft</p>'],
        is_canvas=True,
        element_lines_blind=["<p></p>"],
    )
    back = PageKnowledge.from_dict(json.loads(json.dumps(page.to_dict())))
    assert back.is_canvas is True
    assert back.element_lines_blind == ["<p></p>"]


def test_legacy_page_json_rehydrates_canvas_fields_from_the_first_observation(tmp_path):
    writer = DataWriter(data_dir=str(tmp_path / "data"), runtime_dir=str(tmp_path / "runtime"))
    writer.init_session("net.osmand", "net.osmand")

    sm = _matcher()
    match = _drive(sm, MAP_1)
    writer.save_page_knowledge(match.page_key, sm.get_page_knowledge(match.page_key))
    writer.save_observation(
        match.page_key, match.observation_num, None, MAP_1,
        match=match, activity="net.osmand/.MapActivity",
    )

    # Rewrite page.json as a pre-S-9 file: neither canvas key present.
    path = tmp_path / "data" / "net.osmand" / "pages" / match.page_key / "page.json"
    legacy = json.loads(path.read_text())
    del legacy["is_canvas"], legacy["element_lines_blind"]
    path.write_text(json.dumps(legacy))

    fresh = _matcher()
    rehydrate_screen_matcher(fresh, writer)
    page = fresh.get_page_knowledge(match.page_key)
    assert page.is_canvas is True
    assert page.element_lines_blind == serialize_element_lines(_enc(MAP_1), blind_text=True)

    # And the canvas path is live again on the resumed matcher.
    assert _drive(fresh, MAP_2).is_new_page is False
