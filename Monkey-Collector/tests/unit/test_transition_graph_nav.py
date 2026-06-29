"""Tests for TransitionGraph shortest-path navigation."""

from monkey_collector.pipeline.exploration.state import SemanticState
from monkey_collector.pipeline.exploration.transition_graph import TransitionGraph


def _state(structure: str) -> SemanticState:
    """A minimal state identified solely by its structure_str."""
    return SemanticState(
        state_str=structure,
        structure_str=structure,
        activity="com.test.app/.A",
        package="com.test.app",
        elements=(),
    )


def test_two_hop_path():
    graph = TransitionGraph()
    a, b, c = _state("A"), _state("B"), _state("C")
    graph.add(a, "btn:to_b", "touch", b)
    graph.add(b, "btn:to_c", "touch", c)

    steps = graph.shortest_nav_steps(a, c)
    assert steps is not None
    assert [s.structure_str for s in steps] == ["A", "B"]
    assert steps[0].element_signature == "btn:to_b"
    assert steps[1].element_signature == "btn:to_c"


def test_same_structure_returns_empty_plan():
    graph = TransitionGraph()
    a = _state("A")
    graph.add(a, "btn", "touch", _state("B"))
    assert graph.shortest_nav_steps(a, _state("A")) == []


def test_unreachable_target_returns_none():
    graph = TransitionGraph()
    a, b, c = _state("A"), _state("B"), _state("C")
    graph.add(a, "btn", "touch", b)  # C is registered but disconnected
    graph.add(c, "btn", "touch", c)  # self-loop only → C has no in-path from A
    assert graph.shortest_nav_steps(a, c) is None


def test_unknown_node_returns_none():
    graph = TransitionGraph()
    a = _state("A")
    graph.add(a, "btn", "touch", _state("B"))
    assert graph.shortest_nav_steps(a, _state("Z")) is None


def test_self_loop_not_added_as_edge():
    graph = TransitionGraph()
    a = _state("A")
    graph.add(a, "btn", "touch", _state("A"))  # structure unchanged
    # A exists as a node but has no outgoing edge → nothing reachable
    assert graph.shortest_nav_steps(a, _state("A")) == []


def test_shortest_path_prefers_fewer_hops():
    graph = TransitionGraph()
    a, b, c, d = _state("A"), _state("B"), _state("C"), _state("D")
    graph.add(a, "a_to_b", "touch", b)
    graph.add(b, "b_to_d", "touch", d)
    graph.add(a, "a_to_c", "touch", c)
    graph.add(c, "c_to_b", "touch", b)
    # A→B→D (2 hops) beats A→C→B→D (3 hops)
    steps = graph.shortest_nav_steps(a, d)
    assert steps is not None
    assert len(steps) == 2
