"""Tests for structural sibling effect-skip (C1) in Memory / the explorer.

The structural effect log — (page_key, struct_key, action_type) -> destination
pages, firing count, seen-set — is loaded on every transition regardless of the
knobs. ``sibling_skip`` then HARD-drops a saturated sibling group (fired past
the threshold, single destination) from the frontier, while scrollables stay
exempt and a group with two-or-more destinations never saturates. Default OFF
must leave the frontier untouched.
"""

from monkey_collector.pipeline.exploration.explorer import CoverageGuidedExplorer
from monkey_collector.pipeline.exploration.memory import Memory
from monkey_collector.pipeline.exploration.state import (
    LONG_TOUCH,
    SCROLL,
    SET_TEXT,
    TOUCH,
    SemanticState,
)
from monkey_collector.xml.ui_tree import UITree
from tests.fixtures.xml_samples import COMPLEX_XML, SIMPLE_XML

ACTIVITY = "com.test.app/.MainActivity"
SETTINGS = "com.test.app/.SettingsActivity"
PACKAGE = "com.test.app"


def _state(xml: str, activity: str = ACTIVITY) -> SemanticState:
    return SemanticState.from_screen(xml, activity, PACKAGE)


def _frontier_pairs(memory: Memory, state: SemanticState) -> set[tuple[str, str]]:
    return {(e.signature, at) for _, e, at in memory.unexplored_actions([state])}


# ── effect log loads on every transition (knob-independent) ──


def test_effect_log_loads_on_transition_even_when_off():
    # Knobs default OFF, yet the structural effect log still accumulates so a
    # knob could be flipped mid-session without a cold start.
    memory = Memory()
    src = _state(SIMPLE_XML)
    dst = _state(COMPLEX_XML, SETTINGS)
    memory.record_state(src)
    memory.record_state(dst)
    el = src.find_by_signature("button::Add new")

    memory.record_transition(src, el.signature, TOUCH, dst)

    key = (src.page_key, el.struct_key, TOUCH)
    assert memory._effect_counts[key] == 1
    assert memory._effects[key] == {dst.page_key}
    assert memory.struct_explored_anywhere(el.struct_key, TOUCH) is True


# ── saturation: single destination past threshold → sibling hard-skipped ──


def test_saturated_single_destination_skips_unexplored_sibling():
    # "button::Search" fires 5× (> default threshold 4) always to one page, so
    # its structure "button::156x120" saturates. The genuine sibling
    # "button::Add new" (same struct, still unexplored) is then HARD-skipped.
    memory = Memory(sibling_skip=True)
    src = _state(SIMPLE_XML)
    dst = _state(COMPLEX_XML, SETTINGS)
    memory.record_state(src)
    memory.record_state(dst)
    search = src.find_by_signature("button::Search")
    for _ in range(5):
        memory.record_transition(src, search.signature, TOUCH, dst)

    key = (src.page_key, search.struct_key, TOUCH)
    assert memory._sibling_saturated(*key) is True
    assert ("button::Add new", TOUCH) not in _frontier_pairs(memory, src)


def test_below_threshold_does_not_skip():
    # Exactly threshold firings is NOT past it (count must be strictly greater),
    # so the sibling stays on the frontier.
    memory = Memory(sibling_skip=True)  # default threshold 4
    src = _state(SIMPLE_XML)
    dst = _state(COMPLEX_XML, SETTINGS)
    memory.record_state(src)
    memory.record_state(dst)
    search = src.find_by_signature("button::Search")
    for _ in range(4):
        memory.record_transition(src, search.signature, TOUCH, dst)

    assert ("button::Add new", TOUCH) in _frontier_pairs(memory, src)


# ── two distinct destinations → permanent non-skip ──


def test_multiple_destinations_never_saturate():
    # Rows leading to DIFFERENT pages are not a redundant list: even far past the
    # threshold, two distinct destinations keep the group permanently unskipped.
    memory = Memory(sibling_skip=True)
    src = _state(SIMPLE_XML)
    dst1 = _state(COMPLEX_XML, SETTINGS)
    dst2 = _state(COMPLEX_XML, "com.test.app/.OtherActivity")
    memory.record_state(src)
    search = src.find_by_signature("button::Search")
    for i in range(6):
        memory.record_transition(src, search.signature, TOUCH, dst1 if i % 2 else dst2)

    key = (src.page_key, search.struct_key, TOUCH)
    assert len(memory._effects[key]) == 2
    assert memory._sibling_saturated(*key) is False
    assert ("button::Add new", TOUCH) in _frontier_pairs(memory, src)


# ── knob OFF → never skips ──


def test_off_never_skips_even_when_saturated():
    memory = Memory(sibling_skip=False)
    src = _state(SIMPLE_XML)
    dst = _state(COMPLEX_XML, SETTINGS)
    memory.record_state(src)
    search = src.find_by_signature("button::Search")
    for _ in range(10):
        memory.record_transition(src, search.signature, TOUCH, dst)

    key = (src.page_key, search.struct_key, TOUCH)
    # The log is populated (loaded regardless of the knob) …
    assert memory._effect_counts[key] == 10
    # … but with the knob OFF nothing saturates and the sibling stays offered.
    assert memory._sibling_saturated(*key) is False
    assert ("button::Add new", TOUCH) in _frontier_pairs(memory, src)


# ── threshold is injectable ──


def test_threshold_injection_lowers_saturation_point():
    # threshold=2 saturates at count 3, where the default threshold=4 would not —
    # proving the injected value is the one used.
    src = _state(SIMPLE_XML)
    dst = _state(COMPLEX_XML, SETTINGS)
    search = src.find_by_signature("button::Search")

    lo = Memory(sibling_skip=True, sibling_skip_threshold=2)
    hi = Memory(sibling_skip=True, sibling_skip_threshold=4)
    for mem in (lo, hi):
        mem.record_state(src)
        for _ in range(3):
            mem.record_transition(src, search.signature, TOUCH, dst)

    key = (src.page_key, search.struct_key, TOUCH)
    assert lo._sibling_saturated(*key) is True
    assert hi._sibling_saturated(*key) is False
    assert ("button::Add new", TOUCH) not in _frontier_pairs(lo, src)
    assert ("button::Add new", TOUCH) in _frontier_pairs(hi, src)


# ── scrollables are exempt (negative index) ──


def test_scrollable_is_exempt_from_sibling_skip():
    # A scrollable cannot be saturated via the frontier (its action is marked
    # explored on first fire), so drive the log directly: even a saturated scroll
    # group keeps its SCROLL offered because negative indices are exempt.
    memory = Memory(sibling_skip=True)
    src = _state(SIMPLE_XML)
    memory.record_state(src)
    scroll = next(e for e in src.elements if e.index < 0)
    key = (src.page_key, scroll.struct_key, SCROLL)
    memory._effect_counts[key] = 10
    memory._effects[key] = {"only-dst"}

    assert memory._sibling_saturated(*key) is True  # the group IS saturated
    assert (scroll.signature, SCROLL) in _frontier_pairs(memory, src)  # still offered


# ── explorer-level: knobs survive reset() ──


def test_explorer_carries_knobs_into_memory_and_reset(mock_adb):
    explorer = CoverageGuidedExplorer(
        mock_adb,
        config={
            "seed": 42,
            "sibling_skip": True,
            "sibling_skip_threshold": 2,
            "struct_novelty_rank": True,
        },
    )
    assert explorer._memory._sibling_skip is True
    assert explorer._memory._sibling_skip_threshold == 2
    assert explorer._memory.struct_novelty_rank is True

    explorer.reset()  # the rebuilt Memory must keep the same knobs
    assert explorer._memory._sibling_skip is True
    assert explorer._memory._sibling_skip_threshold == 2
    assert explorer._memory.struct_novelty_rank is True


def test_sibling_skip_flips_has_unvisited_to_false(mock_adb):
    # The hard-skip must propagate to has_unvisited (which reads the same
    # frontier), so an exhausted-by-siblings screen reports nothing unvisited and
    # the plateau early-stop can reclaim the budget.
    explorer = CoverageGuidedExplorer(
        mock_adb, config={"seed": 42, "sibling_skip": True}
    )
    tree = UITree.from_xml_string(SIMPLE_XML)
    src = _state(SIMPLE_XML)
    dst = _state(COMPLEX_XML, SETTINGS)
    explorer._current_state = src
    mem = explorer._memory
    mem.record_state(src)
    # Explore every frontier action except the FAB's tap.
    mem.mark_explored(src.page_key, "button::Search", TOUCH)
    mem.mark_explored(src.page_key, "button::Search", LONG_TOUCH)
    mem.mark_explored(src.page_key, "input:text:Search field", SET_TEXT)
    mem.mark_explored(src.page_key, "button::Add new", LONG_TOUCH)
    scroll = next(e for e in src.elements if e.index < 0)
    mem.mark_explored(src.page_key, scroll.signature, SCROLL)

    # Only "button::Add new" TOUCH is left → the screen still has an unvisited.
    assert explorer.has_unvisited(tree, None) is True

    # Saturate its structure (button::156x120) through its sibling's firings.
    search = src.find_by_signature("button::Search")
    for _ in range(5):
        mem.record_transition(src, search.signature, TOUCH, dst)

    # The last frontier action is now sibling-skipped → nothing unvisited.
    assert explorer.has_unvisited(tree, None) is False
