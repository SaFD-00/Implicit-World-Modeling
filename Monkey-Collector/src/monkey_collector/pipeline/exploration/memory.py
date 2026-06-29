"""Exploration memory: coverage tracking with same-function compression.

Ports the essence of LLM-Explorer's ``Memory`` — what has been explored and what
remains — but keyed on stable element *signatures* instead of bound-box strings,
which removes the reference's bound-box matching and same-frame index bookkeeping
entirely.

Core ideas preserved:
  - An *abstract page* is a ``structure_str`` (text-invariant), so coverage
    learned on one screen instance applies to every instance of that layout.
  - An action's coverage unit is ``(structure_str, element_signature, action_type)``.
  - The LLM's same-function element groups compress the frontier: exploring one
    element of a group marks the whole group explored for that action type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from monkey_collector.pipeline.exploration.constants import (
    MIN_SIZE_SAME_FUNCTION_ELEMENT_GROUP,
)
from monkey_collector.pipeline.exploration.state import (
    LONG_TOUCH,
    TOUCH,
    SemanticElement,
    SemanticState,
)
from monkey_collector.pipeline.exploration.transition_graph import TransitionGraph

if TYPE_CHECKING:
    from monkey_collector.llm.screen_grouper import ScreenGrouper

# One unexplored candidate: the screen, the element, and the action to try.
UnexploredAction = tuple[SemanticState, SemanticElement, str]


class Memory:
    """Tracks explored actions, same-function groups, and the transition graph."""

    def __init__(self, screen_grouper: ScreenGrouper | None = None):
        self._screen_grouper = screen_grouper
        self._states: dict[str, SemanticState] = {}
        # structure_str -> set of (element_signature, action_type)
        self._explored: dict[str, set[tuple[str, str]]] = {}
        self._nav_failed: dict[str, set[tuple[str, str]]] = {}
        # structure_str -> list of same-function signature groups
        self._groups: dict[str, list[set[str]]] = {}
        self.transition_graph = TransitionGraph()

    # -- observation ----------------------------------------------------------

    def record_state(self, state: SemanticState, raw_xml: str) -> None:
        """Register an observed screen and compute its same-function groups once."""
        self._states[state.state_str] = state
        self.transition_graph.add_state(state)
        if state.structure_str not in self._groups:
            self._groups[state.structure_str] = self._compute_groups(state, raw_xml)

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
        self.mark_explored(from_state.structure_str, element_signature, action_type)
        self.transition_graph.add(from_state, element_signature, action_type, to_state)

    def mark_explored(
        self,
        structure_str: str,
        element_signature: str,
        action_type: str,
    ) -> None:
        """Record an action as explored, extending to its same-function group."""
        explored = self._explored.setdefault(structure_str, set())
        explored.add((element_signature, action_type))
        for group in self._groups.get(structure_str, []):
            if element_signature in group:
                for sibling in group:
                    explored.add((sibling, action_type))

    def mark_nav_failed(
        self,
        structure_str: str,
        element_signature: str,
        action_type: str,
    ) -> None:
        """Permanently exclude an action that navigation failed to execute."""
        self._nav_failed.setdefault(structure_str, set()).add(
            (element_signature, action_type)
        )

    # -- queries --------------------------------------------------------------

    def unexplored_actions(self, states: list[SemanticState]) -> list[UnexploredAction]:
        """All not-yet-explored (state, element, action) candidates in *states*.

        ``long_touch`` is only offered once the element's ``touch`` is explored —
        it rarely has an independent effect, so it is kept as a follow-up.
        """
        candidates: list[UnexploredAction] = []
        for state in states:
            blocked = self._blocked_pairs(state.structure_str)
            for element in state.actionable_elements():
                touch_done = (element.signature, TOUCH) in blocked
                for action_type in element.allowed_actions:
                    if (element.signature, action_type) in blocked:
                        continue
                    if action_type == LONG_TOUCH and not touch_done:
                        continue
                    candidates.append((state, element, action_type))
        return candidates

    def in_app_states(self) -> list[SemanticState]:
        """All observed screens that belong to the target app."""
        return [state for state in self._states.values() if state.is_in_app()]

    # -- internals ------------------------------------------------------------

    def _blocked_pairs(self, structure_str: str) -> set[tuple[str, str]]:
        return self._explored.get(structure_str, set()) | self._nav_failed.get(
            structure_str, set()
        )

    def _compute_groups(self, state: SemanticState, raw_xml: str) -> list[set[str]]:
        """Resolve same-function element groups (as signature sets) via the LLM.

        Returns ``[]`` when no grouper is available (e.g. no API key) or the
        screen is too small to be worth grouping, so exploration degrades to pure
        unexplored-first without any LLM dependency.
        """
        if self._screen_grouper is None:
            return []
        if len(state.elements) < MIN_SIZE_SAME_FUNCTION_ELEMENT_GROUP:
            return []
        try:
            result = self._screen_grouper.group(raw_xml)
        except Exception:
            return []

        index_to_signature = {element.index: element.signature for element in state.elements}
        groups: list[set[str]] = []
        for group in result.get("groups", []):
            signatures = {
                index_to_signature[index]
                for index in group.get("indices", [])
                if index in index_to_signature
            }
            if len(signatures) >= 2:
                groups.append(signatures)
        return groups
