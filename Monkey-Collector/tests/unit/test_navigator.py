"""Tests for Navigator: queue draining, re-matching, abandonment guards."""

import random

from monkey_collector.pipeline.exploration.memory import Memory
from monkey_collector.pipeline.exploration.navigator import Navigator
from monkey_collector.pipeline.exploration.state import TOUCH, SemanticState
from monkey_collector.pipeline.exploration.transition_graph import NavStep
from tests.fixtures.xml_samples import COMPLEX_XML, SIMPLE_XML

ACTIVITY = "com.test.app/.MainActivity"
SETTINGS = "com.test.app/.SettingsActivity"
PACKAGE = "com.test.app"


def _state(xml: str, activity: str) -> SemanticState:
    return SemanticState.from_screen(xml, activity, PACKAGE)


def _exhaust(memory: Memory, state: SemanticState) -> None:
    """Mark every action on *state* explored, to convergence.

    Looping is required because long_touch only joins the frontier after its
    element's touch is explored, so a single pass leaves follow-ups behind.
    """
    while True:
        actions = memory.unexplored_actions([state])
        if not actions:
            return
        for _, element, action_type in actions:
            memory.mark_explored(state.structure_str, element.signature, action_type)


def _memory_with_edge():
    """Memory where SIMPLE --(Add new, touch)--> COMPLEX, SIMPLE fully explored."""
    memory = Memory()
    a = _state(SIMPLE_XML, ACTIVITY)
    b = _state(COMPLEX_XML, SETTINGS)
    memory.record_state(a, SIMPLE_XML)
    memory.record_state(b, COMPLEX_XML)
    memory.record_transition(a, "button::Add new", TOUCH, b)
    # Exhaust SIMPLE so the nearest unexplored action lives on COMPLEX.
    _exhaust(memory, a)
    return memory, a, b


def test_plan_routes_to_unexplored_on_another_screen():
    memory, a, b = _memory_with_edge()
    nav = Navigator(memory, random.Random(0))

    assert nav.plan_to_unexplored(a)
    assert nav.is_navigating()
    # first leg is the SIMPLE→COMPLEX transition action
    step = nav.next_action(a)
    assert step is not None
    element, action_type = step
    assert element.signature == "button::Add new"
    assert action_type == TOUCH


def test_plan_drains_to_target_action_then_empties():
    memory, a, b = _memory_with_edge()
    nav = Navigator(memory, random.Random(0))
    nav.plan_to_unexplored(a)

    nav.next_action(a)  # SIMPLE→COMPLEX leg
    final = nav.next_action(b)  # the unexplored action on COMPLEX
    assert final is not None
    assert not nav.is_navigating()  # queue drained


def test_drift_off_path_abandons_plan():
    memory, a, b = _memory_with_edge()
    nav = Navigator(memory, random.Random(0))
    nav.plan_to_unexplored(a)

    # Expected to be on SIMPLE for the first leg, but we are on COMPLEX.
    assert nav.next_action(b) is None
    assert not nav.is_navigating()


def test_unmatchable_element_marks_nav_failed():
    memory = Memory()
    a = _state(SIMPLE_XML, ACTIVITY)
    memory.record_state(a, SIMPLE_XML)
    nav = Navigator(memory, random.Random(0))
    # Inject a plan whose element does not exist on the live screen.
    nav._queue = [NavStep(a.structure_str, "button::ghost", TOUCH)]

    assert nav.next_action(a) is None
    assert not nav.is_navigating()
    # The phantom action is now permanently excluded.
    blocked = memory._blocked_pairs(a.structure_str)
    assert ("button::ghost", TOUCH) in blocked


def test_no_plan_when_everything_explored():
    memory = Memory()
    a = _state(SIMPLE_XML, ACTIVITY)
    memory.record_state(a, SIMPLE_XML)
    _exhaust(memory, a)
    nav = Navigator(memory, random.Random(0))
    assert not nav.plan_to_unexplored(a)


def test_step_budget_abandons_long_plan():
    memory = Memory()
    a = _state(SIMPLE_XML, ACTIVITY)
    memory.record_state(a, SIMPLE_XML)
    nav = Navigator(memory, random.Random(0))
    # A plan longer than the budget, all on the same structure.
    nav._queue = [
        NavStep(a.structure_str, "button::Search", TOUCH)
        for _ in range(50)
    ]
    nav._steps_taken = 999  # already over budget
    assert nav.next_action(a) is None
    assert not nav.is_navigating()
