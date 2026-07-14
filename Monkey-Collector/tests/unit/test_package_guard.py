"""Tests for the same-package BM25 merge guard.

The merge path never checked WHICH APP a screen belongs to, so a launcher/home
frame that shares a few generic element-lines with an app screen could merge into
its page — measured in the live corpora before any canvas change (osmand armB
pages 0/34, broccoli page 0). The guard vetoes every cross-package merge, canvas
or not, and abstains when either package is unknown.

Fixtures: APP_* and LAUNCHER_* are element-line-IDENTICAL screens under different
packages (distinct resource-ids keep their structural fingerprints apart, so the
pre-filter cannot short-circuit and BM25 verification really runs). Without the
guard they merge (|A △ B| = 0); with it they must not.
"""

import json

from monkey_collector.pipeline.screen_matching.rehydrate import rehydrate_screen_matcher
from monkey_collector.pipeline.screen_matching.screen_matcher import ScreenMatcher, package_of
from monkey_collector.storage import DataWriter
from monkey_collector.xml.structured_parser import encode_with_bounds

APP = "com.app/com.app.MainActivity"
LAUNCHER = "com.google.android.apps.nexuslauncher/…NexusLauncherActivity"


def _screen(children, root="[0,0][400,800]"):
    return (
        '<hierarchy rotation="0">'
        f'<node class="android.widget.FrameLayout" bounds="{root}">{children}</node>'
        "</hierarchy>"
    )


def _btn(rid, desc, bounds):
    return (
        f'<node class="android.widget.Button" resource-id="{rid}" content-desc="{desc}" '
        f'text="" bounds="{bounds}" clickable="true"/>'
    )


def _surface(rid):
    return (
        f'<node class="android.view.View" resource-id="{rid}" content-desc="Map" text="" '
        'bounds="[0,0][400,780]" clickable="true"/>'
    )


# Same element-lines (aria-label carries identity, resource-id is dropped), so
# the element criterion sees |A △ B| = 0 — only the package differs.
APP_FLAT = _screen(_btn("com.app:id/a", "Add", "[0,0][100,100]") + _btn("com.app:id/s", "Search", "[0,100][100,200]"))
LAUNCHER_FLAT = _screen(_btn("com.l:id/a", "Add", "[0,0][100,100]") + _btn("com.l:id/s", "Search", "[0,100][100,200]"))
APP_CANVAS = _screen(_surface("com.app:id/m") + _btn("com.app:id/z", "Zoom in", "[300,700][360,760]"))
LAUNCHER_CANVAS = _screen(_surface("com.l:id/m") + _btn("com.l:id/z", "Zoom in", "[300,700][360,760]"))


def _enc(raw):
    return encode_with_bounds(raw)[0]


def _matcher(**kw):
    return ScreenMatcher(None, luminance_prefilter=True, **kw)


def _drive(sm, raw, activity):
    return sm.match(raw, _enc(raw), activity)


def test_package_of_splits_on_the_slash():
    assert package_of("com.app/com.app.MainActivity") == "com.app"
    assert package_of("com.app") == "com.app"
    assert package_of("") == ""
    assert package_of(None) == ""


def test_cross_package_non_canvas_merge_is_blocked():
    sm = _matcher()
    a = _drive(sm, APP_FLAT, APP)
    launcher = _drive(sm, LAUNCHER_FLAT, LAUNCHER)
    assert a.is_new_page is True
    assert launcher.is_new_page is True  # identical element-lines, different app
    assert launcher.page_key != a.page_key


def test_cross_package_canvas_pair_is_blocked():
    sm = _matcher()
    a = _drive(sm, APP_CANVAS, APP)
    launcher = _drive(sm, LAUNCHER_CANVAS, LAUNCHER)
    assert a.is_new_page is True
    assert launcher.is_new_page is True
    assert launcher.page_key != a.page_key


def test_guard_off_restores_the_pre_guard_cross_package_merge():
    # The defect the guard fixes, reproduced: with the knob off these merge.
    sm = _matcher(package_guard=False)
    a = _drive(sm, APP_FLAT, APP)
    launcher = _drive(sm, LAUNCHER_FLAT, LAUNCHER)
    assert launcher.is_new_page is False
    assert launcher.page_key == a.page_key
    assert launcher.match_type == "BM25_MERGE"


def test_same_package_merge_still_happens():
    # The guard must not fragment one app: different window labels of the same
    # package merge exactly as before (a drawer over a map is the same page).
    sm = _matcher()
    a = _drive(sm, APP_FLAT, "com.app/com.app.MainActivity")
    b = _drive(sm, LAUNCHER_FLAT, "com.app/androidx.drawerlayout.widget.DrawerLayout")
    assert b.is_new_page is False
    assert b.page_key == a.page_key


def test_unknown_package_makes_the_guard_abstain():
    # Fail-open on BOTH sides: a missing activity label must never fragment a page.
    sm = _matcher()
    a = _drive(sm, APP_FLAT, APP)
    assert _drive(sm, LAUNCHER_FLAT, "").is_new_page is False  # current unknown

    sm2 = _matcher()
    a2 = _drive(sm2, APP_FLAT, "")  # page minted with no activity
    merged = _drive(sm2, LAUNCHER_FLAT, LAUNCHER)
    assert a.is_new_page is True and a2.is_new_page is True
    assert merged.is_new_page is False  # candidate's package unknown


def test_minted_page_records_its_activity():
    sm = _matcher()
    page = sm.get_page_knowledge(_drive(sm, APP_FLAT, APP).page_key)
    assert page.first_activity == APP


def test_legacy_page_json_rehydrates_first_activity_from_the_first_observation(tmp_path):
    writer = DataWriter(data_dir=str(tmp_path / "data"), runtime_dir=str(tmp_path / "runtime"))
    writer.init_session("com.app", "com.app")

    sm = _matcher()
    match = _drive(sm, APP_FLAT, APP)
    writer.save_page_knowledge(match.page_key, sm.get_page_knowledge(match.page_key))
    writer.save_observation(
        match.page_key, match.observation_num, None, APP_FLAT, match=match, activity=APP,
    )

    # Rewrite page.json as a pre-guard file: no first_activity key.
    path = tmp_path / "data" / "com.app" / "pages" / match.page_key / "page.json"
    legacy = json.loads(path.read_text())
    del legacy["first_activity"]
    path.write_text(json.dumps(legacy))

    fresh = _matcher()
    rehydrate_screen_matcher(fresh, writer)
    assert fresh.get_page_knowledge(match.page_key).first_activity == APP

    # Guard is live on the resumed matcher (else every legacy page would abstain).
    assert _drive(fresh, LAUNCHER_FLAT, LAUNCHER).is_new_page is True
