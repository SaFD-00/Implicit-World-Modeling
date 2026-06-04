"""Tests for SmartExplorer coverage-driven (unvisited-first) selection."""

import pytest

from monkey_collector.domain.actions import Tap
from monkey_collector.pipeline.explorer import SmartExplorer
from tests.conftest import make_tree


@pytest.fixture
def explorer(mock_adb):
    return SmartExplorer(mock_adb, config={"seed": 42})


@pytest.fixture
def two_buttons():
    return make_tree([
        {"clickable": True, "resource_id": "com.app:id/a", "bounds": (0, 0, 100, 100),
         "class_name": "android.widget.Button"},
        {"clickable": True, "resource_id": "com.app:id/b", "bounds": (200, 0, 300, 100),
         "class_name": "android.widget.Button"},
    ])


def _tap_only(explorer):
    explorer.action_weights = {
        "tap": 1.0, "press_back": 0, "swipe": 0,
        "input_text": 0, "long_press": 0, "press_home": 0,
    }


class TestUnvisitedFirst:
    def test_second_tap_picks_unvisited_element(self, explorer, two_buttons):
        _tap_only(explorer)
        first = explorer.select_action(two_buttons, step=5, page_id=0)
        second = explorer.select_action(two_buttons, step=6, page_id=0)
        assert isinstance(first, Tap)
        assert isinstance(second, Tap)
        # The first element is now visited, so the second pick must differ.
        assert second.element_index != first.element_index

    def test_visited_tracked_per_page(self, explorer, two_buttons):
        _tap_only(explorer)
        explorer.select_action(two_buttons, step=5, page_id=0)
        # Different page_id has its own (empty) visited set — may repeat freely.
        other_page = explorer.select_action(two_buttons, step=6, page_id=1)
        assert isinstance(other_page, Tap)
        assert explorer._visited_signatures.get(0)
        assert explorer._visited_signatures.get(1)

    def test_no_page_id_disables_tracking(self, explorer, two_buttons):
        _tap_only(explorer)
        explorer.select_action(two_buttons, step=5, page_id=None)
        assert not explorer._visited_signatures


class TestHasUnvisited:
    def test_initially_true(self, explorer, two_buttons):
        assert explorer.has_unvisited(two_buttons, page_id=0)

    def test_false_when_all_visited(self, explorer, two_buttons):
        _tap_only(explorer)
        explorer.select_action(two_buttons, step=5, page_id=0)
        explorer.select_action(two_buttons, step=6, page_id=0)
        assert not explorer.has_unvisited(two_buttons, page_id=0)

    def test_none_page_id_is_false(self, explorer, two_buttons):
        assert not explorer.has_unvisited(two_buttons, page_id=None)


class TestEarlyBackSuppressed:
    def test_no_back_in_first_steps(self, explorer, two_buttons):
        # Only tap and press_back carry weight. During early steps press_back is
        # zeroed, so every selection must be a tap.
        explorer.action_weights = {
            "tap": 1.0, "press_back": 1.0, "swipe": 0,
            "input_text": 0, "long_press": 0, "press_home": 0,
        }
        for _ in range(20):
            action = explorer.select_action(two_buttons, step=0, page_id=0)
            assert action.action_type != "press_back"

    def test_back_allowed_after_first_steps(self, explorer, two_buttons):
        explorer.action_weights = {
            "tap": 0, "press_back": 1.0, "swipe": 0,
            "input_text": 0, "long_press": 0, "press_home": 0,
        }
        # step beyond FIRST_STEPS_NO_BACK and not first screen → back permitted
        action = explorer.select_action(two_buttons, step=10, page_id=0)
        assert action.action_type == "press_back"
