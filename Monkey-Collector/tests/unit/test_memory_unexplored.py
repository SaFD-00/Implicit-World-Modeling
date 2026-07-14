"""Tests for Memory: unexplored frontier, nav-fail, R1 ranking accessors."""

from monkey_collector.pipeline.exploration.memory import Memory
from monkey_collector.pipeline.exploration.state import (
    LONG_TOUCH,
    TOUCH,
    SemanticState,
)
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


# ── R1 value-guided ranking accessors (explored_anywhere) ──


def test_explored_anywhere_spans_pages():
    memory = Memory()
    page_a = _state(SIMPLE_XML, ACTIVITY)
    page_b = _state(SIMPLE_XML, SETTINGS)  # different activity → different page_key
    memory.record_state(page_a)
    memory.record_state(page_b)
    assert page_a.page_key != page_b.page_key

    assert memory.explored_anywhere("button::Search", TOUCH) is False
    # exploring it on page_a makes it non-novel everywhere.
    memory.mark_explored(page_a.page_key, "button::Search", TOUCH)
    assert memory.explored_anywhere("button::Search", TOUCH) is True
    # a different action type on the same signature stays novel.
    assert memory.explored_anywhere("button::Search", LONG_TOUCH) is False


def test_explored_anywhere_ignores_nav_failed():
    # nav-fail is a routing-only exclusion, not real coverage: it must not count
    # as "explored anywhere" or a genuinely unexplored action would be down-ranked.
    memory = Memory()
    state = _state(SIMPLE_XML)
    memory.record_state(state)
    memory.mark_nav_failed(state.page_key, "button::Add new", TOUCH)
    assert memory.explored_anywhere("button::Add new", TOUCH) is False


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
