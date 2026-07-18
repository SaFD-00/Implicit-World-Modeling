"""Tests for R1 value-guided candidate ranking in CoverageGuidedExplorer.

R1 replaces the legacy uniform ``rng.choice`` in ``_pick_unexplored`` with a
deterministic lexicographic ``(novelty, type_prior)`` ranking, keeping the
seeded rng only for exact-tie breaking. These tests pin each signal in
isolation plus the invariants (long_touch stays demoted; a fixed seed yields
an identical action sequence).
"""

from monkey_collector.pipeline.exploration.explorer import (
    CoverageGuidedExplorer,
    _candidate_score,
)
from monkey_collector.pipeline.exploration.memory import Memory
from monkey_collector.pipeline.exploration.state import (
    SET_TEXT,
    TOUCH,
    SemanticElement,
    SemanticState,
)
from monkey_collector.xml.ui_tree import UITree
from tests.fixtures.xml_samples import SIMPLE_XML

ACTIVITY = "com.test.app/.MainActivity"
SETTINGS = "com.test.app/.SettingsActivity"
PACKAGE = "com.test.app"


def _state(xml: str, activity: str = ACTIVITY) -> SemanticState:
    return SemanticState.from_screen(xml, activity, PACKAGE)


def _explorer(mock_adb) -> CoverageGuidedExplorer:
    return CoverageGuidedExplorer(mock_adb, config={"seed": 42})


def _drive(explorer: CoverageGuidedExplorer, xml: str, steps: int) -> list:
    explorer.set_screen_context(xml, ACTIVITY, PACKAGE)
    tree = UITree.from_xml_string(xml)
    return [explorer.select_action(tree, is_first_screen=False) for _ in range(steps)]


# ── 1) type_prior: tap outranks a text entry on the same screen ──


def test_type_prior_returns_after_text_guarantee_is_spent(mock_adb):
    # SPEC CHANGE (W1 once-per-page SET_TEXT guarantee): the FIRST frontier pick
    # on a screen with an editable field is now elevated to input_text so the
    # corpus captures a typing→screen-change transition. The guarantee is
    # one-shot, so pick 2 returns to normal type_prior ranking, where a tap
    # (highest type_prior) outranks the now-spent search field.
    #   Was: "first pick must be a tap" — that contract is intentionally
    #   superseded by the guarantee and is retained here as pick 2's behaviour.
    explorer = _explorer(mock_adb)
    picks = _drive(explorer, SIMPLE_XML, 2)
    assert picks[0].action_type == "input_text"  # W1 guarantee fires first
    assert picks[1].action_type == "tap"  # type_prior resumes once spent


def test_type_prior_ordering_is_lexicographic():
    memory = Memory()
    state = _state(SIMPLE_XML)
    memory.record_state(state)
    touch_el = state.find_by_signature("button::Search")
    text_el = state.find_by_signature("input:text:Search field")
    # Same novelty (both untried) and uniqueness (no groups): type_prior decides.
    assert _candidate_score(memory, state, touch_el, TOUCH) > _candidate_score(
        memory, state, text_el, SET_TEXT
    )


# ── 2) novelty: cross-page seen-before is down-ranked; @-fallback is exempt ──


def test_novelty_prefers_action_unseen_on_any_page():
    memory = Memory()
    page_a = _state(SIMPLE_XML, ACTIVITY)
    page_b = _state(SIMPLE_XML, SETTINGS)
    memory.record_state(page_a)
    memory.record_state(page_b)
    # "button::Search" TOUCH was explored on page A → non-novel on page B.
    memory.mark_explored(page_a.page_key, "button::Search", TOUCH)

    seen = page_b.find_by_signature("button::Search")
    fresh = page_b.find_by_signature("button::Add new")  # never explored anywhere
    seen_score = _candidate_score(memory, page_b, seen, TOUCH)
    fresh_score = _candidate_score(memory, page_b, fresh, TOUCH)
    assert fresh_score[0] == 1 and seen_score[0] == 0
    assert fresh_score > seen_score


def test_index_fallback_signature_is_never_penalised():
    # "@<index>" signatures have no cross-page identity, so even after being
    # marked explored they must keep novelty=1 (else unrelated elements sharing
    # the fallback would be wrongly treated as already seen).
    memory = Memory()
    state = _state(SIMPLE_XML)
    memory.record_state(state)
    memory.mark_explored(state.page_key, "button::@2", TOUCH)
    assert memory.explored_anywhere("button::@2", TOUCH) is True

    fallback_el = SemanticElement(
        index=2,
        signature="button::@2",
        allowed_actions=(TOUCH,),
        center=(0, 0),
        desc="",
    )
    assert _candidate_score(memory, state, fallback_el, TOUCH)[0] == 1


def test_at_prefixed_label_is_not_mistaken_for_index_fallback():
    # A real label starting with "@" (e.g. a mention/handle) is NOT the generated
    # "@<index>" fallback, so once explored it must down-rank to novelty=0 rather
    # than being spuriously exempted.
    memory = Memory()
    state = _state(SIMPLE_XML)
    memory.record_state(state)
    memory.mark_explored(state.page_key, "button::@home", TOUCH)
    assert memory.explored_anywhere("button::@home", TOUCH) is True

    labelled_el = SemanticElement(
        index=2,
        signature="button::@home",
        allowed_actions=(TOUCH,),
        center=(0, 0),
        desc="",
    )
    assert _candidate_score(memory, state, labelled_el, TOUCH)[0] == 0


# ── 3) invariants: long_touch stays demoted; seed 42 is fully deterministic ──


def test_long_touch_stays_demoted_after_primary_actions(mock_adb):
    explorer = _explorer(mock_adb)
    actions = _drive(explorer, SIMPLE_XML, 20)
    types = [a.action_type for a in actions]
    primary = {"tap", "swipe", "input_text"}
    long_idxs = [i for i, t in enumerate(types) if t == "long_press"]
    primary_idxs = [i for i, t in enumerate(types) if t in primary]
    # long_touch is a follow-up: no long_press may precede any primary action.
    assert long_idxs, "expected long_touch to eventually appear"
    assert min(long_idxs) > max(primary_idxs)


def test_same_seed_yields_identical_action_sequence(mock_adb):
    # Two independent explorers (fresh Memory + rng, seed 42) over the same frame
    # sequence must emit byte-identical actions. select_action never touches adb,
    # so sharing one mock is safe; determinism comes purely from the seed.
    def _sequence():
        explorer = CoverageGuidedExplorer(mock_adb, config={"seed": 42})
        actions = _drive(explorer, SIMPLE_XML, 20)
        return [(a.action_type, a.element_index) for a in actions]

    assert _sequence() == _sequence()


# ── 4) W1 once-per-page SET_TEXT guarantee ──


def test_frontier_first_pick_is_text_entry(mock_adb):
    # W1: an editable screen must contribute a typing→screen-change transition,
    # so the very first frontier pick is elevated to the field's input_text
    # (despite SET_TEXT being the lowest type_prior).
    explorer = _explorer(mock_adb)
    first = _drive(explorer, SIMPLE_XML, 1)[0]
    assert first.action_type == "input_text"


def test_text_guarantee_is_one_shot_per_page(mock_adb):
    # A single editable field yields exactly one input_text across a full
    # session: elevated once, then never re-summoned — the field is explored and
    # the page spent, and the non-root fallback presses back rather than
    # re-tapping the search box.
    explorer = _explorer(mock_adb)
    types = [a.action_type for a in _drive(explorer, SIMPLE_XML, 12)]
    assert types[0] == "input_text"
    assert types.count("input_text") == 1


def test_emit_marks_page_spent_and_suppresses_elevation(mock_adb):
    # The guarantee is spent in _emit, not at elevation time: once ANY set_text
    # is emitted for a page_key, _pick_unexplored stops elevating text on that
    # page — even while an unexplored SET_TEXT candidate is still present.
    explorer = _explorer(mock_adb)
    explorer.set_screen_context(SIMPLE_XML, ACTIVITY, PACKAGE)
    state = _state(SIMPLE_XML)
    explorer._memory.record_state(state)

    pick1 = explorer._pick_unexplored(state)
    assert pick1[1] == SET_TEXT  # elevation fires on the fresh page

    explorer._emit(state, *pick1)  # spends the guarantee via _emit
    assert state.page_key in explorer._text_spent

    # The field's SET_TEXT is still unexplored in memory, but the page is spent,
    # so the next pick falls through to normal ranking (a tap outranks it).
    pick2 = explorer._pick_unexplored(state)
    assert pick2[1] == TOUCH


def test_reset_revives_text_guarantee(mock_adb):
    # reset() clears the spent-page set (and the per-session memory), so a fresh
    # session re-guarantees the text entry on the same screen.
    explorer = _explorer(mock_adb)
    assert _drive(explorer, SIMPLE_XML, 1)[0].action_type == "input_text"
    explorer.reset()
    assert _drive(explorer, SIMPLE_XML, 1)[0].action_type == "input_text"


# ── 5) C1b struct-novelty ranking (default OFF) ──


def test_candidate_score_off_returns_legacy_two_tuple():
    # struct_novelty_rank OFF (default): the score is the unchanged legacy
    # (novelty, type_prior) 2-tuple — no extra tier, byte-identical to R1.
    memory = Memory()
    state = _state(SIMPLE_XML)
    memory.record_state(state)
    fab = state.find_by_signature("button::Add new")  # untried tap
    score = _candidate_score(memory, state, fab, TOUCH)
    assert score == (1, 3)  # novelty=1 (unseen), type_prior=3 (tap)
    assert len(score) == 2


def test_struct_novelty_prefers_unseen_structure_when_on():
    # struct_novelty_rank ON adds a top tier. Exercising "button::Search" marks
    # its structure (button::156x120) seen, so its still-unexplored sibling
    # "button::Add new" (same struct, novelty still 1) is out-ranked by a fresh
    # structure — isolating the struct_novelty tier since novelty ties at 1.
    memory = Memory(struct_novelty_rank=True)
    src = _state(SIMPLE_XML)
    dst = _state(SIMPLE_XML, SETTINGS)
    memory.record_state(src)
    memory.record_state(dst)
    search = src.find_by_signature("button::Search")
    memory.record_transition(src, search.signature, TOUCH, dst)

    seen_struct = src.find_by_signature("button::Add new")  # struct seen, sig unseen
    fresh_struct = SemanticElement(
        index=9,
        signature="button::Never seen",
        allowed_actions=(TOUCH,),
        center=(0, 0),
        desc="",
        struct_key="button::999x999",  # never exercised
    )
    seen_score = _candidate_score(memory, src, seen_struct, TOUCH)
    fresh_score = _candidate_score(memory, src, fresh_struct, TOUCH)
    assert len(seen_score) == 3 and len(fresh_score) == 3
    assert seen_score[0] == 0 and fresh_score[0] == 1  # struct_novelty tier
    assert seen_score[1] == fresh_score[1] == 1  # novelty ties → struct decides
    assert fresh_score > seen_score


def test_struct_novelty_on_same_seed_is_deterministic(mock_adb):
    # Enabling the ranking tier must not break the seed-42 determinism guarantee.
    def _sequence():
        explorer = CoverageGuidedExplorer(
            mock_adb, config={"seed": 42, "struct_novelty_rank": True}
        )
        actions = _drive(explorer, SIMPLE_XML, 20)
        return [(a.action_type, a.element_index) for a in actions]

    assert _sequence() == _sequence()
