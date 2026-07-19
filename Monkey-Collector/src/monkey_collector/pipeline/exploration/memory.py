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

from loguru import logger

from monkey_collector.pipeline.exploration.state import (
    LONG_TOUCH,
    TOUCH,
    SemanticElement,
    SemanticState,
)
from monkey_collector.pipeline.exploration.transition_graph import TransitionGraph

# One unexplored candidate: the screen, the element, and the action to try.
UnexploredAction = tuple[SemanticState, SemanticElement, str]

# Firing count above which a single-destination structural group is reported as
# "saturated" by the end-of-session diagnostic. DIAGNOSTIC OBSERVATION ONLY —
# nothing in exploration reads this, so changing it cannot alter which actions
# are explored; it only shifts where log_effect_summary draws its reporting line.
SATURATION_PROBE_THRESHOLD = 4


class Memory:
    """Tracks explored actions and the transition graph."""

    def __init__(self) -> None:
        self._states: dict[str, SemanticState] = {}
        # page_key -> set of (element_signature, action_type)
        self._explored: dict[str, set[tuple[str, str]]] = {}
        self._nav_failed: dict[str, set[tuple[str, str]]] = {}
        self.transition_graph = TransitionGraph()

        # Structural effect log, keyed on (page_key, struct_key, action_type) so
        # content-differing siblings (list rows, contacts, media) share a bucket:
        # which distinct destination pages the action reached and how many times
        # it fired. Diagnostic instrumentation — loaded on EVERY transition and
        # never read by action selection.
        self._effects: dict[tuple[str, str, str], set[str]] = {}
        self._effect_counts: dict[tuple[str, str, str], int] = {}

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
        # Structural effect log: attribute this outcome to the element's
        # structure so siblings differing only by content share the bucket.
        # Read-only w.r.t. coverage/graph/rng — never alters exploration.
        element = from_state.find_by_signature(element_signature)
        if element is not None and element.struct_key:
            key = (from_state.page_key, element.struct_key, action_type)
            self._effects.setdefault(key, set()).add(to_state.page_key)
            self._effect_counts[key] = self._effect_counts.get(key, 0) + 1

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

    def log_effect_summary(self, top: int = 10) -> None:
        """Log end-of-session structural effect statistics for the explored app.

        Exploration diagnostics, not a feature knob: it answers "how repetitive
        was this app's structure?" by reporting how many (page, structure,
        action) groups fired repeatedly yet only ever reached ONE destination
        page — the fingerprint of a long list whose rows all do the same thing —
        against :data:`SATURATION_PROBE_THRESHOLD`. The per-group breakdown that
        follows ranks the busiest structures with their destination fan-out.

        This instrumentation is what the 2026-07-18 ablation used to show that
        redundant sibling rows were not in fact eating the step budget. Nothing
        here feeds action selection; the effect log is a pure observer.
        """
        if not self._effect_counts:
            logger.info("[effect-log] summary: empty (no attributed transitions)")
            return
        would_saturate = sum(
            1
            for key, count in self._effect_counts.items()
            if count > SATURATION_PROBE_THRESHOLD
            and len(self._effects.get(key, set())) == 1
        )
        logger.info(
            "[effect-log] summary: groups={} would_saturate={} threshold={}",
            len(self._effect_counts),
            would_saturate,
            SATURATION_PROBE_THRESHOLD,
        )
        ranked = sorted(self._effect_counts.items(), key=lambda kv: kv[1], reverse=True)
        for (page_key, struct_key, action), count in ranked[:top]:
            dests = len(self._effects.get((page_key, struct_key, action), set()))
            logger.info(
                "[effect-log]   fired={} dests={} action={} page={} struct={}",
                count,
                dests,
                action,
                page_key,
                struct_key,
            )

    def _blocked_pairs(self, page_key: str) -> set[tuple[str, str]]:
        return self._explored.get(page_key, set()) | self._nav_failed.get(page_key, set())
