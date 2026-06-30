"""Multi-step navigation to unexplored screens, executed one step per signal.

Ports LLM-Explorer's ``_nav_steps`` queue and ``navigate`` / ``_get_nav_action``
loop. Because Monkey-Collector's server consumes one screen signal per step, a
navigation plan is held as a queue and drained one action at a time: each step is
re-matched against the live screen by element signature, so coordinate drift
between visits does not break the plan.

Three traversal strategies determine *which* unexplored target to navigate to:

  GREEDY  Minimise navigation cost — pick the target with the shortest path from
          the current screen. Equivalent to the original behaviour.
  BFS     Breadth-first — pick the target closest to the session root screen,
          exploring shallow pages before deep ones.
  DFS     Depth-first — pick the target farthest from the session root screen,
          diving deep before backtracking.

In all three cases the *route* to the chosen target uses the existing shortest-
path algorithm; strategy only affects *target selection*.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from monkey_collector.pipeline.exploration.constants import MAX_NAVIGATE_NUM_AT_ONE_TIME
from monkey_collector.pipeline.exploration.memory import Memory, UnexploredAction
from monkey_collector.pipeline.exploration.state import SemanticElement, SemanticState
from monkey_collector.pipeline.exploration.transition_graph import NavStep

if TYPE_CHECKING:
    pass


class Navigator:
    """Plans and drives a shortest-path route to an unexplored action."""

    def __init__(self, memory: Memory, rng: random.Random, strategy: str = "GREEDY"):
        self._memory = memory
        self._rng = rng
        self._strategy = strategy.strip().upper()
        self._queue: list[NavStep] = []
        self._steps_taken = 0

    def is_navigating(self) -> bool:
        return bool(self._queue)

    def clear(self) -> None:
        self._queue = []
        self._steps_taken = 0

    def plan_to_unexplored(self, current_state: SemanticState) -> bool:
        """Load a route from *current_state* to an unexplored action.

        Which target is chosen depends on the strategy:
          GREEDY: the target reachable with the shortest navigation path.
          BFS:    the target closest to the session root (shallowest depth).
          DFS:    the target farthest from the session root (deepest depth).

        Returns ``True`` when a plan was loaded.
        """
        candidates = self._memory.unexplored_actions(self._memory.in_app_states())
        self._rng.shuffle(candidates)  # shuffle for tie-breaking randomness

        if self._strategy == "BFS":
            best = self._plan_by_depth(current_state, candidates, prefer_shallow=True)
        elif self._strategy == "DFS":
            best = self._plan_by_depth(current_state, candidates, prefer_shallow=False)
        else:  # GREEDY (default) or any unrecognised value
            best = self._plan_greedy(current_state, candidates)

        if best is None:
            return False
        self._queue = best
        self._steps_taken = 0
        return True

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _plan_greedy(
        self,
        current_state: SemanticState,
        candidates: list[UnexploredAction],
    ) -> list[NavStep] | None:
        """Pick the target with the shortest navigation path from current_state."""
        graph = self._memory.transition_graph
        best: list[NavStep] | None = None
        for target_state, element, action_type in candidates:
            route = graph.shortest_nav_steps(current_state, target_state)
            if route is None:
                continue
            plan = [*route, NavStep(target_state.page_key, element.signature, action_type)]
            if best is None or len(plan) < len(best):
                best = plan
        return best

    def _plan_by_depth(
        self,
        current_state: SemanticState,
        candidates: list[UnexploredAction],
        prefer_shallow: bool,
    ) -> list[NavStep] | None:
        """Pick the target by BFS depth from the session root.

        prefer_shallow=True → BFS (explore shallowest first).
        prefer_shallow=False → DFS (explore deepest first).

        Falls back to GREEDY when no root is known yet (first step of session).

        Depth is the forward hop-count from the root page over the *recorded*
        transition graph. Only element-driven transitions are recorded — a
        PressBack / fallback / relaunch move is not (the explorer clears
        ``_last_record`` for those), so a screen first reached via such a move
        has no recorded in-edge and its depth is unknown. Unknown-depth
        candidates sort last in both BFS and DFS (treated as ``inf``); the
        strategy is therefore a best-effort ordering over the recorded graph,
        not a guarantee over physically-reachable depth. Targets always remain
        reachable via shortest-path routing regardless of depth.
        """
        root_key = self._memory.root_page_key
        if root_key is None:
            return self._plan_greedy(current_state, candidates)

        graph = self._memory.transition_graph
        depths = graph.bfs_distances(root_key)  # page_key → hop count from root

        scored: list[tuple[float, list[NavStep]]] = []
        for target_state, element, action_type in candidates:
            route = graph.shortest_nav_steps(current_state, target_state)
            if route is None:
                continue
            plan = [*route, NavStep(target_state.page_key, element.signature, action_type)]
            raw_depth = depths.get(target_state.page_key, -1)

            if prefer_shallow:
                # BFS: want minimum depth — unknown → treat as very deep
                sort_key: float = raw_depth if raw_depth >= 0 else float("inf")
            else:
                # DFS: want maximum depth — negate for min-sort; unknown → treat as very shallow
                sort_key = -raw_depth if raw_depth >= 0 else float("inf")

            scored.append((sort_key, plan))

        if not scored:
            return None

        # Primary: depth metric. Secondary: plan length (shorter navigation path).
        scored.sort(key=lambda x: (x[0], len(x[1])))
        return scored[0][1]

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

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
        if current_state.page_key != step.page_key:
            # Landed somewhere unexpected — drop the plan and let the engine replan.
            self.clear()
            return None

        element = current_state.find_by_signature(step.element_signature)
        if element is None or step.action_type not in element.allowed_actions:
            self._memory.mark_nav_failed(
                step.page_key, step.element_signature, step.action_type
            )
            self.clear()
            return None

        self._queue.pop(0)
        self._steps_taken += 1
        return element, step.action_type
