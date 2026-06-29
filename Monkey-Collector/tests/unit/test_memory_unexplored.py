"""Tests for Memory: unexplored frontier, same-function compression, nav-fail."""

from monkey_collector.pipeline.exploration.memory import Memory
from monkey_collector.pipeline.exploration.state import (
    LONG_TOUCH,
    SELECT,
    TOUCH,
    SemanticState,
)
from tests.fixtures.xml_samples import COMPLEX_XML, SIMPLE_XML

ACTIVITY = "com.test.app/.MainActivity"
SETTINGS = "com.test.app/.SettingsActivity"
PACKAGE = "com.test.app"


def _state(xml: str, activity: str = ACTIVITY) -> SemanticState:
    return SemanticState.from_screen(xml, activity, PACKAGE)


class _StubGrouper:
    """Returns a fixed grouping; mimics ScreenGrouper.group's shape."""

    def __init__(self, groups):
        self._groups = groups

    def group(self, raw_xml):
        return {"groups": self._groups}


# ── frontier basics ──


def test_new_state_exposes_all_actions_except_long_touch_followups():
    memory = Memory()
    state = _state(SIMPLE_XML)
    memory.record_state(state, SIMPLE_XML)

    actions = memory.unexplored_actions([state])
    pairs = {(e.signature, at) for _, e, at in actions}
    # buttons offer touch (long_touch withheld until touch is explored)
    assert ("button::Search", TOUCH) in pairs
    assert ("button::Search", LONG_TOUCH) not in pairs
    assert ("input:text:Search field", "set_text") in pairs


def test_explored_action_drops_from_frontier():
    memory = Memory()
    state = _state(SIMPLE_XML)
    memory.record_state(state, SIMPLE_XML)

    memory.mark_explored(state.structure_str, "button::Search", TOUCH)
    pairs = {(e.signature, at) for _, e, at in memory.unexplored_actions([state])}
    assert ("button::Search", TOUCH) not in pairs


def test_long_touch_appears_after_touch_explored():
    memory = Memory()
    state = _state(SIMPLE_XML)
    memory.record_state(state, SIMPLE_XML)

    memory.mark_explored(state.structure_str, "button::Search", TOUCH)
    pairs = {(e.signature, at) for _, e, at in memory.unexplored_actions([state])}
    assert ("button::Search", LONG_TOUCH) in pairs


# ── same-function compression ──


def test_same_function_group_compresses_frontier():
    # Group the two checkboxes (encoded indices 6 and 8) as same-function.
    grouper = _StubGrouper([{"indices": [6, 8], "function": "toggle"}])
    memory = Memory(screen_grouper=grouper)
    state = _state(COMPLEX_XML, SETTINGS)
    memory.record_state(state, COMPLEX_XML)

    # Exploring one checkbox's select marks the whole group explored.
    memory.mark_explored(state.structure_str, "input:checkbox:Dark mode toggle", SELECT)
    pairs = {(e.signature, at) for _, e, at in memory.unexplored_actions([state])}
    assert ("input:checkbox:Notifications toggle", SELECT) not in pairs


def test_grouping_skipped_without_grouper():
    memory = Memory(screen_grouper=None)
    state = _state(COMPLEX_XML, SETTINGS)
    memory.record_state(state, COMPLEX_XML)

    memory.mark_explored(state.structure_str, "input:checkbox:Dark mode toggle", SELECT)
    pairs = {(e.signature, at) for _, e, at in memory.unexplored_actions([state])}
    # Without grouping, the other checkbox is still unexplored.
    assert ("input:checkbox:Notifications toggle", SELECT) in pairs


# ── nav-fail exclusion & in-app gate ──


def test_nav_failed_action_excluded():
    memory = Memory()
    state = _state(SIMPLE_XML)
    memory.record_state(state, SIMPLE_XML)

    memory.mark_nav_failed(state.structure_str, "button::Add new", TOUCH)
    pairs = {(e.signature, at) for _, e, at in memory.unexplored_actions([state])}
    assert ("button::Add new", TOUCH) not in pairs


def test_in_app_states_filters_foreign_screens():
    memory = Memory()
    in_app = _state(SIMPLE_XML, "com.test.app/.MainActivity")
    foreign = _state(SIMPLE_XML, "com.other.app/.Main")
    memory.record_state(in_app, SIMPLE_XML)
    memory.record_state(foreign, SIMPLE_XML)

    in_app_structs = {s.structure_str for s in memory.in_app_states()}
    assert in_app.structure_str in in_app_structs
    assert foreign.structure_str not in in_app_structs


def test_record_transition_marks_explored_and_builds_graph():
    memory = Memory()
    src = _state(SIMPLE_XML)
    dst = _state(COMPLEX_XML, SETTINGS)
    memory.record_state(src, SIMPLE_XML)
    memory.record_state(dst, COMPLEX_XML)

    memory.record_transition(src, "button::Add new", TOUCH, dst)
    # explored
    pairs = {(e.signature, at) for _, e, at in memory.unexplored_actions([src])}
    assert ("button::Add new", TOUCH) not in pairs
    # graph edge usable for navigation
    steps = memory.transition_graph.shortest_nav_steps(src, dst)
    assert steps is not None and len(steps) == 1
