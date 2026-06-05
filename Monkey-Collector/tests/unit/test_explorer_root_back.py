"""Tests for SmartExplorer root-screen press_back suppression.

Back from the app's root (first in-app) page only exits to the launcher, which
causes external bounce-back churn. select_action must suppress press_back while
is_root_screen is True, regardless of step.
"""

import pytest

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


class TestRootBackSuppressed:
    def test_no_back_on_root_screen(self, explorer, two_buttons):
        # Only tap and press_back carry weight. On the root screen press_back is
        # zeroed even past the early-step window, so every pick must be a tap.
        explorer.action_weights = {
            "tap": 1.0, "press_back": 1.0, "swipe": 0,
            "input_text": 0, "long_press": 0, "press_home": 0,
        }
        for _ in range(20):
            action = explorer.select_action(
                two_buttons, step=20, page_id=0, is_root_screen=True
            )
            assert action.action_type != "press_back"

    def test_back_allowed_off_root_screen(self, explorer, two_buttons):
        # Same late step, but not the root screen → back is permitted again.
        explorer.action_weights = {
            "tap": 0, "press_back": 1.0, "swipe": 0,
            "input_text": 0, "long_press": 0, "press_home": 0,
        }
        action = explorer.select_action(
            two_buttons, step=20, page_id=1, is_root_screen=False
        )
        assert action.action_type == "press_back"

    def test_root_suppression_when_no_actions_available(self, explorer, mock_adb):
        # With no clickable/editable/scrollable elements the weighted total can
        # collapse to zero; on the root screen the fallback must still tap, never
        # fall back to a back that would exit the app.
        empty = make_tree([])
        explorer.action_weights = {
            "tap": 0, "press_back": 1.0, "swipe": 0,
            "input_text": 0, "long_press": 0, "press_home": 0,
        }
        action = explorer.select_action(
            empty, step=20, page_id=0, is_root_screen=True
        )
        assert action.action_type != "press_back"
