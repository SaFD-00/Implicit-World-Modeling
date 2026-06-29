"""Tests for Memory: unexplored frontier, same-function compression, nav-fail."""

from monkey_collector.pipeline.exploration.memory import Memory
from monkey_collector.pipeline.exploration.state import (
    LONG_TOUCH,
    SELECT,
    TOUCH,
    SemanticState,
)
from monkey_collector.pipeline.screen_matching.screen_matcher import ElementFamily
from tests.fixtures.xml_samples import COMPLEX_XML, SIMPLE_XML

ACTIVITY = "com.test.app/.MainActivity"
SETTINGS = "com.test.app/.SettingsActivity"
PACKAGE = "com.test.app"


def _state(xml: str, activity: str = ACTIVITY) -> SemanticState:
    return SemanticState.from_screen(xml, activity, PACKAGE)


# ── frontier basics ──


def test_new_state_exposes_all_actions_except_long_touch_followups():
    memory = Memory()
    state = _state(SIMPLE_XML)
    memory.record_state(state)

    actions = memory.unexplored_actions([state])
    pairs = {(e.signature, at) for _, e, at in actions}
    # buttons offer touch (long_touch withheld until touch is explored)
    assert ("button::Search", TOUCH) in pairs
    assert ("button::Search", LONG_TOUCH) not in pairs
    assert ("input:text:Search field", "set_text") in pairs


def test_explored_action_drops_from_frontier():
    memory = Memory()
    state = _state(SIMPLE_XML)
    memory.record_state(state)

    memory.mark_explored(state.page_key, "button::Search", TOUCH)
    pairs = {(e.signature, at) for _, e, at in memory.unexplored_actions([state])}
    assert ("button::Search", TOUCH) not in pairs


def test_long_touch_appears_after_touch_explored():
    memory = Memory()
    state = _state(SIMPLE_XML)
    memory.record_state(state)

    memory.mark_explored(state.page_key, "button::Search", TOUCH)
    pairs = {(e.signature, at) for _, e, at in memory.unexplored_actions([state])}
    assert ("button::Search", LONG_TOUCH) in pairs


# ── same-function compression (driven by extractor families) ──


def test_same_function_group_compresses_frontier():
    # Group the two checkboxes (encoded indices 6 and 8) as one family.
    family = ElementFamily(
        name="toggle", element_index=[6, 8], key_element_index=[6]
    )
    memory = Memory()
    state = _state(COMPLEX_XML, SETTINGS)
    memory.record_state(state, [family])

    # Exploring one checkbox's select marks the whole family explored.
    memory.mark_explored(state.page_key, "input:checkbox:Dark mode toggle", SELECT)
    pairs = {(e.signature, at) for _, e, at in memory.unexplored_actions([state])}
    assert ("input:checkbox:Notifications toggle", SELECT) not in pairs


def test_grouping_skipped_without_families():
    memory = Memory()
    state = _state(COMPLEX_XML, SETTINGS)
    memory.record_state(state)  # no families → degrade to no compression

    memory.mark_explored(state.page_key, "input:checkbox:Dark mode toggle", SELECT)
    pairs = {(e.signature, at) for _, e, at in memory.unexplored_actions([state])}
    # Without grouping, the other checkbox is still unexplored.
    assert ("input:checkbox:Notifications toggle", SELECT) in pairs


def test_singleton_family_does_not_compress():
    # A family that maps to a single signature must not form a group.
    family = ElementFamily(name="solo", element_index=[6], key_element_index=[6])
    memory = Memory()
    state = _state(COMPLEX_XML, SETTINGS)
    memory.record_state(state, [family])
    assert memory._groups[state.page_key] == []


# ── nav-fail exclusion & in-app gate ──


def test_nav_failed_action_excluded():
    memory = Memory()
    state = _state(SIMPLE_XML)
    memory.record_state(state)

    memory.mark_nav_failed(state.page_key, "button::Add new", TOUCH)
    pairs = {(e.signature, at) for _, e, at in memory.unexplored_actions([state])}
    assert ("button::Add new", TOUCH) not in pairs


def test_in_app_states_filters_foreign_screens():
    memory = Memory()
    in_app = _state(SIMPLE_XML, "com.test.app/.MainActivity")
    foreign = _state(SIMPLE_XML, "com.other.app/.Main")
    memory.record_state(in_app)
    memory.record_state(foreign)

    in_app_keys = {s.page_key for s in memory.in_app_states()}
    assert in_app.page_key in in_app_keys
    assert foreign.page_key not in in_app_keys


def test_record_transition_marks_explored_and_builds_graph():
    memory = Memory()
    src = _state(SIMPLE_XML)
    dst = _state(COMPLEX_XML, SETTINGS)
    memory.record_state(src)
    memory.record_state(dst)

    memory.record_transition(src, "button::Add new", TOUCH, dst)
    # explored
    pairs = {(e.signature, at) for _, e, at in memory.unexplored_actions([src])}
    assert ("button::Add new", TOUCH) not in pairs
    # graph edge usable for navigation
    steps = memory.transition_graph.shortest_nav_steps(src, dst)
    assert steps is not None and len(steps) == 1
