"""Tests for R1 value-guided candidate ranking in LLMGuidedExplorer.

R1 (docs/research/gui-exploration-world-model.md line 161) replaces the legacy
uniform ``rng.choice`` in ``_pick_unexplored`` with a deterministic lexicographic
``(novelty, type_prior, uniqueness)`` ranking, keeping the seeded rng only for
exact-tie breaking. These tests pin each signal in isolation plus the invariants
(long_touch stays demoted; a fixed seed yields an identical action sequence).
"""

from monkey_collector.pipeline.exploration.explorer import (
    LLMGuidedExplorer,
    _candidate_score,
)
from monkey_collector.pipeline.exploration.memory import Memory
from monkey_collector.pipeline.exploration.state import (
    SELECT,
    SET_TEXT,
    TOUCH,
    SemanticElement,
    SemanticState,
)
from monkey_collector.pipeline.screen_matching.screen_matcher import ElementFamily
from monkey_collector.xml.ui_tree import UITree
from tests.fixtures.xml_samples import COMPLEX_XML, SIMPLE_XML

ACTIVITY = "com.test.app/.MainActivity"
SETTINGS = "com.test.app/.SettingsActivity"
PACKAGE = "com.test.app"


def _state(xml: str, activity: str = ACTIVITY) -> SemanticState:
    return SemanticState.from_screen(xml, activity, PACKAGE)


def _explorer(mock_adb) -> LLMGuidedExplorer:
    return LLMGuidedExplorer(mock_adb, config={"seed": 42})


def _drive(explorer: LLMGuidedExplorer, xml: str, steps: int) -> list:
    explorer.set_screen_context(xml, ACTIVITY, PACKAGE)
    tree = UITree.from_xml_string(xml)
    return [explorer.select_action(tree, is_first_screen=False) for _ in range(steps)]


# ── 1) type_prior: tap outranks a text entry on the same screen ──


def test_type_prior_touch_selected_before_set_text(mock_adb):
    # SIMPLE_XML has both tap buttons (search/fab) and a SET_TEXT search field.
    # The first pick must be a tap: touch has the highest type_prior.
    explorer = _explorer(mock_adb)
    first = _drive(explorer, SIMPLE_XML, 1)[0]
    assert first.action_type == "tap"


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


# ── 3) uniqueness: a group-free element outranks a same-function member ──


def test_uniqueness_prefers_ungrouped_over_family_member():
    family = ElementFamily(name="toggle", element_index=[6, 8], key_element_index=[6])
    memory = Memory()
    state = _state(COMPLEX_XML, SETTINGS)
    memory.record_state(state, [family])

    grouped = state.find_by_signature("input:checkbox:Dark mode toggle")
    # Fabricate an ungrouped SELECT element (same novelty and type_prior) so only
    # the uniqueness signal differs.
    ungrouped = SemanticElement(
        index=99,
        signature="input:checkbox:Airplane toggle",
        allowed_actions=(SELECT,),
        center=(0, 0),
        desc="",
    )
    grouped_score = _candidate_score(memory, state, grouped, SELECT)
    ungrouped_score = _candidate_score(memory, state, ungrouped, SELECT)
    assert grouped_score[2] == 0 and ungrouped_score[2] == 1
    assert ungrouped_score > grouped_score


# ── 4) invariants: long_touch stays demoted; seed 42 is fully deterministic ──


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
        explorer = LLMGuidedExplorer(mock_adb, config={"seed": 42})
        actions = _drive(explorer, SIMPLE_XML, 20)
        return [(a.action_type, a.element_index) for a in actions]

    assert _sequence() == _sequence()
