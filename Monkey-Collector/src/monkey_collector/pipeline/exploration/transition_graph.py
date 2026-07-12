"""UI transition graph for shortest-path navigation to unexplored screens.

Ports LLM-Explorer's ``UTG.G2`` — the *structure* graph — which is the one its
``get_G2_nav_steps`` actually navigates over. Nodes are ``page_key`` (the
element-set page identity, or ``structure_str`` when no matcher is active, so
screens that are the same logical page collapse to one node, keeping paths
findable); edges carry the (element_signature, action_type) pairs observed to
cause that transition. The per-state ``G`` graph from the reference is omitted
because only structural navigation is needed here.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from monkey_collector.pipeline.exploration.state import SemanticState


@dataclass(frozen=True)
class NavStep:
    """One leg of a navigation plan.

    Identifies the action by ``element_signature`` (not coordinates) so it can be
    re-matched against the live screen when executed.
    """

    page_key: str
    element_signature: str
    action_type: str


class TransitionGraph:
    """Directed graph of screen-structure transitions for navigation."""

    def __init__(self) -> None:
        self._graph = nx.DiGraph()

    def add_state(self, state: SemanticState | None) -> None:
        """Register a screen as a node without recording any transition."""
        if state is not None:
            self._graph.add_node(state.page_key)

    def add(
        self,
        from_state: SemanticState | None,
        element_signature: str,
        action_type: str,
        to_state: SemanticState | None,
    ) -> None:
        """Record that *action* on *from_state* led to *to_state*.

        Registers both endpoints as nodes. Self-loops (the structure did not
        change) are kept out of the edge set — they are useless for navigation.

        The latest observation of a (signature, action_type) on *src* is the
        truth about where that action leads: before recording it on the
        src→dst edge, purge the same pair from every *other* out-edge of src
        (and drop an edge whose action set empties out). Without this, a stale
        edge from an earlier destination lets Navigator re-route over an action
        whose real target changed — the deterministic livelock behind R2's
        volume-not-diversity ping-pong. The purge runs even when the new
        observation is a self-loop (src == dst): the action no longer navigates
        anywhere, so its old edge must still go, though the self-loop itself is
        never added as an edge.
        """
        if from_state is not None:
            self._graph.add_node(from_state.page_key)
        if to_state is not None:
            self._graph.add_node(to_state.page_key)
        if from_state is None or to_state is None:
            return

        src, dst = from_state.page_key, to_state.page_key

        pair = (element_signature, action_type)
        for other in list(self._graph.successors(src)):
            if other == dst:
                continue
            actions: set[tuple[str, str]] = self._graph[src][other]["actions"]
            if pair in actions:
                actions.discard(pair)
                if not actions:
                    self._graph.remove_edge(src, other)

        if src == dst:
            return
        if not self._graph.has_edge(src, dst):
            self._graph.add_edge(src, dst, actions=set())
        self._graph[src][dst]["actions"].add(pair)

    def shortest_nav_steps(
        self,
        from_state: SemanticState,
        to_state: SemanticState,
    ) -> list[NavStep] | None:
        """Shortest action sequence from *from_state* to *to_state*.

        Returns ``[]`` when already at the target structure, a list of
        :class:`NavStep` for a reachable target, or ``None`` when unreachable.
        """
        src, dst = from_state.page_key, to_state.page_key
        if src == dst:
            return []
        if src not in self._graph or dst not in self._graph:
            return None
        try:
            path = nx.shortest_path(self._graph, src, dst)
        except nx.NetworkXNoPath:
            return None

        steps: list[NavStep] = []
        for current, nxt in zip(path, path[1:], strict=False):
            signature, action_type = self._representative_action(current, nxt)
            steps.append(
                NavStep(
                    page_key=current,
                    element_signature=signature,
                    action_type=action_type,
                )
            )
        return steps

    def bfs_distances(self, from_page_key: str) -> dict[str, int]:
        """BFS hop-counts from *from_page_key* to every reachable node.

        Returns an empty dict when the source is not in the graph.
        Used by Navigator to rank unexplored targets by depth for BFS/DFS strategy.
        """
        if from_page_key not in self._graph:
            return {}
        return dict(nx.single_source_shortest_path_length(self._graph, from_page_key))

    def _representative_action(self, src: str, dst: str) -> tuple[str, str]:
        """Pick a deterministic (signature, action_type) for the src→dst edge."""
        actions: set[tuple[str, str]] = self._graph[src][dst]["actions"]
        signature, action_type = sorted(actions)[0]
        return signature, action_type
