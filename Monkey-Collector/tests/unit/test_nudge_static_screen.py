"""Tests for nudge_static_screen — signal-timeout recovery while still in-app."""

from monkey_collector.pipeline.recovery import nudge_static_screen
from tests.conftest import make_tree


class TestNudgeStaticScreen:
    def test_taps_clickable_element(self, mock_adb):
        tree = make_tree([
            {"clickable": True, "bounds": (0, 0, 100, 100)},
            {"clickable": True, "bounds": (200, 200, 400, 400)},
        ])
        nudge_static_screen(mock_adb, tree, attempt=0)
        # First clickable element center, not a blind screen-center tap.
        mock_adb.tap.assert_called_once_with(50, 50)
        mock_adb.swipe.assert_not_called()

    def test_rotates_clickable_across_attempts(self, mock_adb):
        tree = make_tree([
            {"clickable": True, "bounds": (0, 0, 100, 100)},
            {"clickable": True, "bounds": (200, 200, 400, 400)},
        ])
        nudge_static_screen(mock_adb, tree, attempt=1)
        # attempt 1 → second element (300, 300), so successive timeouts vary.
        mock_adb.tap.assert_called_once_with(300, 300)

    def test_scrolls_when_no_clickable(self, mock_adb):
        tree = make_tree([
            {"scrollable": True, "bounds": (0, 0, 1000, 1000)},
        ])
        nudge_static_screen(mock_adb, tree, attempt=0)
        mock_adb.swipe.assert_called_once()
        mock_adb.tap.assert_not_called()

    def test_center_tap_fallback_when_no_tree(self, mock_adb):
        nudge_static_screen(mock_adb, None, attempt=0)
        # Falls back to screen-center tap (1080x1920 → 540, 960).
        mock_adb.tap.assert_called_once_with(540, 960)

    def test_center_tap_fallback_when_empty_tree(self, mock_adb):
        tree = make_tree([{"clickable": False, "scrollable": False}])
        nudge_static_screen(mock_adb, tree, attempt=0)
        mock_adb.tap.assert_called_once_with(540, 960)
