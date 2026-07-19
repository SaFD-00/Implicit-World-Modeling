"""Tests for the structural effect log — exploration diagnostics in ``Memory``.

The effect log keys transitions on (page_key, struct_key, action_type) so that
content-differing siblings (list rows, contacts, media) share a bucket, and
records which distinct destination pages each group reached plus how many times
it fired. It is pure instrumentation: loaded on EVERY transition and never read
by action selection, so it cannot change which actions get explored.

``log_effect_summary`` turns that log into an end-of-session report — how many
groups fired repeatedly yet only ever reached ONE destination page, measured
against the diagnostic-only ``SATURATION_PROBE_THRESHOLD``.
"""

import io

from loguru import logger

from monkey_collector.pipeline.exploration.memory import (
    SATURATION_PROBE_THRESHOLD,
    Memory,
)
from monkey_collector.pipeline.exploration.state import (
    TOUCH,
    SemanticState,
)
from tests.fixtures.xml_samples import COMPLEX_XML, SIMPLE_XML

ACTIVITY = "com.test.app/.MainActivity"
SETTINGS = "com.test.app/.SettingsActivity"
PACKAGE = "com.test.app"


def _state(xml: str, activity: str = ACTIVITY) -> SemanticState:
    return SemanticState.from_screen(xml, activity, PACKAGE)


def _frontier_pairs(memory: Memory, state: SemanticState) -> set[tuple[str, str]]:
    return {(e.signature, at) for _, e, at in memory.unexplored_actions([state])}


def _capture_summary(memory: Memory) -> str:
    """Run log_effect_summary with loguru re-enabled and return its output."""
    sink = io.StringIO()
    logger.enable("monkey_collector")
    handler_id = logger.add(sink, format="{message}", level="INFO")
    try:
        memory.log_effect_summary()
    finally:
        logger.remove(handler_id)
        logger.disable("monkey_collector")
    return sink.getvalue()


# ── the effect log loads on every transition ──


def test_effect_log_loads_on_transition():
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


def test_effect_log_never_removes_frontier_candidates():
    # The log is an observer: even a heavily-fired single-destination group
    # leaves its still-unexplored structural siblings on the frontier.
    memory = Memory()
    src = _state(SIMPLE_XML)
    dst = _state(COMPLEX_XML, SETTINGS)
    memory.record_state(src)
    memory.record_state(dst)
    search = src.find_by_signature("button::Search")
    for _ in range(10):
        memory.record_transition(src, search.signature, TOUCH, dst)

    key = (src.page_key, search.struct_key, TOUCH)
    assert memory._effect_counts[key] == 10
    assert ("button::Add new", TOUCH) in _frontier_pairs(memory, src)


# ── log_effect_summary: knob-free diagnostic ──


def test_summary_reports_empty_log():
    assert "empty (no attributed transitions)" in _capture_summary(Memory())


def test_summary_counts_would_saturate_past_threshold():
    # Firing once MORE than the probe threshold, always to a single destination,
    # is what the diagnostic counts as a saturating group.
    memory = Memory()
    src = _state(SIMPLE_XML)
    dst = _state(COMPLEX_XML, SETTINGS)
    memory.record_state(src)
    memory.record_state(dst)
    search = src.find_by_signature("button::Search")
    for _ in range(SATURATION_PROBE_THRESHOLD + 1):
        memory.record_transition(src, search.signature, TOUCH, dst)

    output = _capture_summary(memory)
    assert f"would_saturate=1 threshold={SATURATION_PROBE_THRESHOLD}" in output
    assert "groups=1" in output


def test_summary_does_not_count_group_at_threshold():
    # Exactly threshold firings is NOT past it (count must be strictly greater).
    memory = Memory()
    src = _state(SIMPLE_XML)
    dst = _state(COMPLEX_XML, SETTINGS)
    memory.record_state(src)
    memory.record_state(dst)
    search = src.find_by_signature("button::Search")
    for _ in range(SATURATION_PROBE_THRESHOLD):
        memory.record_transition(src, search.signature, TOUCH, dst)

    assert "would_saturate=0" in _capture_summary(memory)


def test_summary_does_not_count_multi_destination_group():
    # Rows leading to DIFFERENT pages are not a redundant list, so even far past
    # the threshold the group is never reported as saturating.
    memory = Memory()
    src = _state(SIMPLE_XML)
    dst1 = _state(COMPLEX_XML, SETTINGS)
    dst2 = _state(COMPLEX_XML, "com.test.app/.OtherActivity")
    memory.record_state(src)
    search = src.find_by_signature("button::Search")
    for i in range(6):
        memory.record_transition(src, search.signature, TOUCH, dst1 if i % 2 else dst2)

    key = (src.page_key, search.struct_key, TOUCH)
    assert len(memory._effects[key]) == 2
    assert "would_saturate=0" in _capture_summary(memory)


def test_summary_lists_busiest_group_with_fanout():
    memory = Memory()
    src = _state(SIMPLE_XML)
    dst = _state(COMPLEX_XML, SETTINGS)
    memory.record_state(src)
    memory.record_state(dst)
    search = src.find_by_signature("button::Search")
    for _ in range(3):
        memory.record_transition(src, search.signature, TOUCH, dst)

    output = _capture_summary(memory)
    assert f"fired=3 dests=1 action={TOUCH}" in output
    assert search.struct_key in output
