"""Tests for DFS / BFS / GREEDY target selection in Navigator.

The three strategies differ only in *which* unexplored target they route to;
the route itself is always shortest-path. These tests build synthetic
``SemanticState`` graphs with known page-key depths so each strategy's defining
choice is observable and distinct.
"""

import random

from monkey_collector.pipeline.exploration.memory import Memory
from monkey_collector.pipeline.exploration.navigator import Navigator
from monkey_collector.pipeline.exploration.state import (
    TOUCH,
    SemanticElement,
    SemanticState,
)

PACKAGE = "com.test.app"


def _elem(sig: str) -> SemanticElement:
    return SemanticElement(
        index=1,
        signature=sig,
        allowed_actions=(TOUCH,),
        center=(10, 10),
        desc=f"<button>{sig}</button>",
    )


def _state(page_key: str) -> SemanticState:
    """An in-app state identified by *page_key* with one touchable element."""
    return SemanticState(
        state_str=f"state-{page_key}",
        structure_str=page_key,
        activity=f"{PACKAGE}/.A",
        package=PACKAGE,
        elements=(_elem(f"btn::{page_key}"),),
        page_key=page_key,
    )


def _target_key(nav: Navigator) -> str:
    """page_key of the final leg of the loaded plan (the unexplored target)."""
    assert nav._queue, "expected a non-empty plan"
    return nav._queue[-1].page_key


def _chain_memory():
    """root -> A -> B -> C; root exhausted, A/B/C each have one unexplored action.

    Depths from root: A=1, B=2, C=3.
    """
    memory = Memory()
    root, a, b, c = _state("root"), _state("A"), _state("B"), _state("C")
    for s in (root, a, b, c):  # root recorded first → root_page_key == "root"
        memory.record_state(s)
    g = memory.transition_graph
    g.add(root, "e_root_a", TOUCH, a)
    g.add(a, "e_a_b", TOUCH, b)
    g.add(b, "e_b_c", TOUCH, c)
    # exhaust root so navigation must leave it
    memory.mark_explored("root", "btn::root", TOUCH)
    return memory, root


def _cyclic_memory():
    """root -> X -> Y -> Z, with Z -> X (back) and Z -> W.

    Depths from root: X=1, Y=2, Z=3, W=4. root and Y exhausted; X/Z/W unexplored.
    From current Y the reachable targets are Z (1 hop), X (2 hops), W (2 hops):
      GREEDY → Z (shortest nav), BFS → X (shallowest), DFS → W (deepest).
    """
    memory = Memory()
    root, x, y, z, w = (
        _state("root"), _state("X"), _state("Y"), _state("Z"), _state("W")
    )
    for s in (root, x, y, z, w):
        memory.record_state(s)
    g = memory.transition_graph
    g.add(root, "e_root_x", TOUCH, x)
    g.add(x, "e_x_y", TOUCH, y)
    g.add(y, "e_y_z", TOUCH, z)
    g.add(z, "e_z_x", TOUCH, x)   # back-edge (does not shorten X's depth)
    g.add(z, "e_z_w", TOUCH, w)
    memory.mark_explored("root", "btn::root", TOUCH)
    memory.mark_explored("Y", "btn::Y", TOUCH)
    return memory, y


# ── chain graph: BFS shallowest, DFS deepest (from root) ──

def test_bfs_picks_shallowest_from_root():
    memory, root = _chain_memory()
    nav = Navigator(memory, random.Random(0), strategy="BFS")
    assert nav.plan_to_unexplored(root)
    assert _target_key(nav) == "A"


def test_dfs_picks_deepest_from_root():
    memory, root = _chain_memory()
    nav = Navigator(memory, random.Random(0), strategy="DFS")
    assert nav.plan_to_unexplored(root)
    assert _target_key(nav) == "C"


def test_greedy_picks_nearest_from_root():
    memory, root = _chain_memory()
    nav = Navigator(memory, random.Random(0), strategy="GREEDY")
    assert nav.plan_to_unexplored(root)
    assert _target_key(nav) == "A"  # nearest nav == shallowest when at root


# ── cyclic graph: all three diverge ──

def test_cyclic_greedy_picks_nearest():
    memory, current = _cyclic_memory()
    nav = Navigator(memory, random.Random(0), strategy="GREEDY")
    assert nav.plan_to_unexplored(current)
    assert _target_key(nav) == "Z"


def test_cyclic_bfs_picks_shallowest():
    memory, current = _cyclic_memory()
    nav = Navigator(memory, random.Random(0), strategy="BFS")
    assert nav.plan_to_unexplored(current)
    assert _target_key(nav) == "X"


def test_cyclic_dfs_picks_deepest():
    memory, current = _cyclic_memory()
    nav = Navigator(memory, random.Random(0), strategy="DFS")
    assert nav.plan_to_unexplored(current)
    assert _target_key(nav) == "W"


# ── normalisation / fallback ──

def test_lowercase_strategy_normalised():
    memory, root = _chain_memory()
    nav = Navigator(memory, random.Random(0), strategy="dfs")
    assert nav.plan_to_unexplored(root)
    assert _target_key(nav) == "C"


def test_unknown_strategy_falls_back_to_greedy():
    memory, current = _cyclic_memory()
    nav = Navigator(memory, random.Random(0), strategy="SIDEWAYS")
    assert nav.plan_to_unexplored(current)
    assert _target_key(nav) == "Z"  # greedy behaviour


def test_no_candidates_returns_false():
    memory = Memory()
    root = _state("root")
    memory.record_state(root)
    memory.mark_explored("root", "btn::root", TOUCH)
    nav = Navigator(memory, random.Random(0), strategy="BFS")
    assert nav.plan_to_unexplored(root) is False


def test_empty_memory_returns_false():
    memory = Memory()
    nav = Navigator(memory, random.Random(0), strategy="DFS")
    # No states recorded → root unknown, no candidates → graceful False.
    assert nav.plan_to_unexplored(_state("root")) is False


def test_bfs_falls_back_to_greedy_when_root_unknown(monkeypatch):
    """If root_page_key is None but candidates exist, BFS degrades to greedy."""
    memory, current = _cyclic_memory()

    # Force root_page_key to report None (simulating pre-root planning).
    monkeypatch.setattr(type(memory), "root_page_key", property(lambda self: None))
    nav = Navigator(memory, random.Random(0), strategy="BFS")
    assert nav.plan_to_unexplored(current)
    assert _target_key(nav) == "Z"  # greedy nearest
