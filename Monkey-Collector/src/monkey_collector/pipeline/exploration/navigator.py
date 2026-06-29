"""Multi-step navigation to unexplored screens, executed one step per signal.

Ports LLM-Explorer's ``_nav_steps`` queue and ``navigate`` / ``_get_nav_action``
loop. Because Monkey-Collector's server consumes one screen signal per step, a
navigation plan is held as a queue and drained one action at a time: each step is
re-matched against the live screen by element signature, so coordinate drift
between visits does not break the plan.
"""

from __future__ import annotations

import random

from monkey_collector.pipeline.exploration.constants import MAX_NAVIGATE_NUM_AT_ONE_TIME
from monkey_collector.pipeline.exploration.memory import Memory
from monkey_collector.pipeline.exploration.state import SemanticElement, SemanticState
from monkey_collector.pipeline.exploration.transition_graph import NavStep


class Navigator:
    """Plans and drives a shortest-path route to an unexplored action."""

    def __init__(self, memory: Memory, rng: random.Random):
        self._memory = memory
        self._rng = rng
        self._queue: list[NavStep] = []
        self._steps_taken = 0

    def is_navigating(self) -> bool:
        return bool(self._queue)

    def clear(self) -> None:
        self._queue = []
        self._steps_taken = 0

    def plan_to_unexplored(self, current_state: SemanticState) -> bool:
        """Load the shortest route from *current_state* to any unexplored action.

        Considers every unexplored action across all in-app screens and keeps the
        plan with the fewest steps. Returns ``True`` when a plan was loaded.
        """
        candidates = self._memory.unexplored_actions(self._memory.in_app_states())
        self._rng.shuffle(candidates)
        graph = self._memory.transition_graph

        best: list[NavStep] | None = None
        for target_state, element, action_type in candidates:
            route = graph.shortest_nav_steps(current_state, target_state)
            if route is None:
                continue
            plan = [
                *route,
                NavStep(target_state.structure_str, element.signature, action_type),
            ]
            if best is None or len(plan) < len(best):
                best = plan

        if best is None:
            return False
        self._queue = best
        self._steps_taken = 0
        return True

    def next_action(
        self,
        current_state: SemanticState,
    ) -> tuple[SemanticElement, str] | None:
        """Return the next (element, action_type) of the plan for *current_state*.

        Abandons the plan (returning ``None``) when the route drifts off course,
        the step budget is exhausted, or the planned element cannot be re-matched
        on the live screen — in the last case the action is marked nav-failed so
        it is not retried.
        """
        if not self._queue:
            return None
        if self._steps_taken >= MAX_NAVIGATE_NUM_AT_ONE_TIME:
            self.clear()
            return None

        step = self._queue[0]
        if current_state.structure_str != step.structure_str:
            # Landed somewhere unexpected — drop the plan and let the engine replan.
            self.clear()
            return None

        element = current_state.find_by_signature(step.element_signature)
        if element is None or step.action_type not in element.allowed_actions:
            self._memory.mark_nav_failed(
                step.structure_str, step.element_signature, step.action_type
            )
            self.clear()
            return None

        self._queue.pop(0)
        self._steps_taken += 1
        return element, step.action_type
