"""Exploration memory: coverage tracking over abstract pages.

Ports the essence of LLM-Explorer's ``Memory`` — what has been explored and what
remains — but keyed on stable element *signatures* instead of bound-box strings.

Core ideas preserved:
  - An *abstract page* is a ``page_key`` (the element-set page identity from the
    live :class:`~monkey_collector.pipeline.screen_matching.screen_matcher.ScreenMatcher`,
    or ``structure_str`` when no matcher is active), so coverage learned on one
    screen instance applies to every instance of that page.
  - An action's coverage unit is ``(page_key, element_signature, action_type)``.
"""

from __future__ import annotations

from monkey_collector.pipeline.exploration.state import (
    LONG_TOUCH,
    TOUCH,
    SemanticElement,
    SemanticState,
)
from monkey_collector.pipeline.exploration.transition_graph import TransitionGraph

# One unexplored candidate: the screen, the element, and the action to try.
UnexploredAction = tuple[SemanticState, SemanticElement, str]


class Memory:
    """Tracks explored actions and the transition graph."""

    def __init__(self) -> None:
        self._states: dict[str, SemanticState] = {}
        # page_key -> set of (element_signature, action_type)
        self._explored: dict[str, set[tuple[str, str]]] = {}
        self._nav_failed: dict[str, set[tuple[str, str]]] = {}
        self.transition_graph = TransitionGraph()

    # -- observation ----------------------------------------------------------

    def record_state(self, state: SemanticState) -> None:
        """Register an observed screen."""
        self._states[state.state_str] = state
        self.transition_graph.add_state(state)

    def record_transition(
        self,
        from_state: SemanticState | None,
        element_signature: str,
        action_type: str,
        to_state: SemanticState | None,
    ) -> None:
        """Mark the source action explored and add the transition edge."""
        if from_state is None or to_state is None:
            return
        self.mark_explored(from_state.page_key, element_signature, action_type)
        self.transition_graph.add(from_state, element_signature, action_type, to_state)

    def mark_explored(
        self,
        page_key: str,
        element_signature: str,
        action_type: str,
    ) -> None:
        """Record an action as explored on its page."""
        self._explored.setdefault(page_key, set()).add((element_signature, action_type))

    def mark_nav_failed(
        self,
        page_key: str,
        element_signature: str,
        action_type: str,
    ) -> None:
        """Permanently exclude an action that navigation failed to execute."""
        self._nav_failed.setdefault(page_key, set()).add((element_signature, action_type))

    # -- queries --------------------------------------------------------------

    def unexplored_actions(self, states: list[SemanticState]) -> list[UnexploredAction]:
        """All not-yet-explored (state, element, action) candidates in *states*.

        ``long_touch`` is only offered once the element's ``touch`` is explored —
        it rarely has an independent effect, so it is kept as a follow-up.
        """
        candidates: list[UnexploredAction] = []
        for state in states:
            blocked = self._blocked_pairs(state.page_key)
            for element in state.actionable_elements():
                touch_done = (element.signature, TOUCH) in blocked
                for action_type in element.allowed_actions:
                    if (element.signature, action_type) in blocked:
                        continue
                    if action_type == LONG_TOUCH and not touch_done:
                        continue
                    candidates.append((state, element, action_type))
        return candidates

    def explored_anywhere(self, element_signature: str, action_type: str) -> bool:
        """True if this (signature, action) was explored on *any* page this session.

        Cross-page novelty signal for R1 value-guided ranking: an action whose
        signature+type was already exercised elsewhere yields little new coverage,
        so it is down-ranked. Reads only ``_explored`` (real coverage), never
        ``_nav_failed`` (routing-only exclusions).
        """
        pair = (element_signature, action_type)
        return any(pair in explored for explored in self._explored.values())

    @property
    def root_page_key(self) -> str | None:
        """page_key of the first state observed in this session.

        Used by Navigator to compute BFS depths for DFS/BFS target selection.
        Returns None if no states have been recorded yet.
        """
        if not self._states:
            return None
        return next(iter(self._states.values())).page_key

    def in_app_states(self) -> list[SemanticState]:
        """All observed screens that belong to the target app."""
        return [state for state in self._states.values() if state.is_in_app()]

    # -- internals ------------------------------------------------------------

    def _blocked_pairs(self, page_key: str) -> set[tuple[str, str]]:
        return self._explored.get(page_key, set()) | self._nav_failed.get(page_key, set())
