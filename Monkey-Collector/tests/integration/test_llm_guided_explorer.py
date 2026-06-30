"""Integration tests for LLMGuidedExplorer orchestration over mock ADB."""

import pytest

from monkey_collector.domain.actions import Tap
from monkey_collector.pipeline.exploration import Explorer, LLMGuidedExplorer
from monkey_collector.xml.ui_tree import UITree
from tests.fixtures.xml_samples import SIMPLE_XML

ACTIVITY = "com.test.app/.MainActivity"
PACKAGE = "com.test.app"

pytestmark = pytest.mark.integration


def _explorer(mock_adb) -> LLMGuidedExplorer:
    return LLMGuidedExplorer(mock_adb, config={"seed": 42})


def _on_screen(explorer: LLMGuidedExplorer):
    explorer.set_screen_context(SIMPLE_XML, ACTIVITY, PACKAGE)
    return UITree.from_xml_string(SIMPLE_XML)


def test_satisfies_explorer_protocol(mock_adb):
    assert isinstance(_explorer(mock_adb), Explorer)


def test_explores_distinct_elements_then_backs_off(mock_adb):
    explorer = _explorer(mock_adb)
    tree = _on_screen(explorer)

    actions = [explorer.select_action(tree, is_first_screen=False) for _ in range(15)]
    types = [a.action_type for a in actions]

    # multiple distinct on-screen elements were exercised before giving up
    explored_indices = {a.element_index for a in actions if a.element_index >= 0}
    assert len(explored_indices) >= 2
    # once the frontier is exhausted, the engine retreats with back
    assert "press_back" in types


def test_recover_clears_pending_transition_record(mock_adb):
    # After an action is selected, a transition is pending attribution
    # (_last_record). Recovery (relaunch after an excursion) must clear it so
    # the excursion is never recorded as a routing-memory transition.
    explorer = _explorer(mock_adb)
    tree = _on_screen(explorer)
    explorer.select_action(tree, is_first_screen=False)
    assert explorer._last_record is not None

    launched = explorer.recover(PACKAGE)

    assert launched is True  # mock adb relaunches
    assert explorer._last_record is None


def test_return_to_app_clears_pending_transition_record(mock_adb):
    explorer = _explorer(mock_adb)
    tree = _on_screen(explorer)
    explorer.select_action(tree, is_first_screen=False)
    assert explorer._last_record is not None

    # Still on the external app after Back → a relaunch happens → True.
    mock_adb.get_current_package.return_value = "com.other"
    launched = explorer.return_to_app(PACKAGE)

    assert launched is True
    assert explorer._last_record is None


def test_return_to_app_no_relaunch_when_back_returns(mock_adb):
    # Back alone landed us back in the target app → no relaunch, returns False.
    explorer = _explorer(mock_adb)
    tree = _on_screen(explorer)
    explorer.select_action(tree, is_first_screen=False)

    mock_adb.get_current_package.return_value = PACKAGE
    launched = explorer.return_to_app(PACKAGE)

    assert launched is False
    assert explorer._last_record is None


def test_first_screen_never_presses_back(mock_adb):
    explorer = _explorer(mock_adb)
    tree = _on_screen(explorer)

    actions = [explorer.select_action(tree, is_first_screen=True) for _ in range(15)]
    assert all(a.action_type != "press_back" for a in actions)


def test_has_unvisited_follows_frontier(mock_adb):
    explorer = _explorer(mock_adb)
    tree = _on_screen(explorer)

    explorer.select_action(tree, is_first_screen=False)
    assert explorer.has_unvisited(tree, None) is True

    for _ in range(15):
        explorer.select_action(tree, is_first_screen=False)
    assert explorer.has_unvisited(tree, None) is False


def test_reset_restores_frontier(mock_adb):
    explorer = _explorer(mock_adb)
    tree = _on_screen(explorer)
    for _ in range(15):
        explorer.select_action(tree, is_first_screen=False)
    assert explorer.has_unvisited(tree, None) is False

    explorer.reset()
    _on_screen(explorer)
    explorer.select_action(tree, is_first_screen=False)
    assert explorer.has_unvisited(tree, None) is True


def test_excluded_element_not_reselected(mock_adb):
    explorer = _explorer(mock_adb)
    tree = _on_screen(explorer)
    first = explorer.select_action(tree, is_first_screen=False)
    assert first.element_index >= 0

    explorer.exclude_element(first.element_index)
    later = [explorer.select_action(tree, is_first_screen=False) for _ in range(8)]
    assert all(a.element_index != first.element_index for a in later if a.element_index >= 0)


def test_execute_action_drives_adb(mock_adb):
    explorer = _explorer(mock_adb)
    explorer.execute_action(Tap(x=11, y=22))
    mock_adb.tap.assert_called_once_with(11, 22)


def test_has_left_app_uses_current_package(mock_adb):
    explorer = _explorer(mock_adb)
    mock_adb.get_current_package.return_value = PACKAGE
    assert explorer.has_left_app(PACKAGE) is False
    mock_adb.get_current_package.return_value = "com.other.app"
    assert explorer.has_left_app(PACKAGE) is True
