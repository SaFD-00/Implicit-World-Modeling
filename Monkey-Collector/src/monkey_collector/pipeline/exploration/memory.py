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


class Memory:
    """Tracks explored actions and the transition graph."""

    def __init__(
        self,
        sibling_skip: bool = False,
        sibling_skip_threshold: int = 4,
        struct_novelty_rank: bool = False,
    ) -> None:
        self._states: dict[str, SemanticState] = {}
        # page_key -> set of (element_signature, action_type)
        self._explored: dict[str, set[tuple[str, str]]] = {}
        self._nav_failed: dict[str, set[tuple[str, str]]] = {}
        self.transition_graph = TransitionGraph()

        # Structural effect log (C1/C1b), keyed on (page_key, struct_key,
        # action_type) so content-differing siblings (list rows, contacts, media)
        # share a bucket: which distinct destination pages the action reached,
        # how many times it fired, and every (struct_key, action_type) exercised
        # anywhere this session. Loaded on EVERY transition regardless of the
        # knobs below, so a knob may be flipped without a cold-start gap.
        self._effects: dict[tuple[str, str, str], set[str]] = {}
        self._effect_counts: dict[tuple[str, str, str], int] = {}
        self._struct_seen: set[tuple[str, str]] = set()

        # Knobs (default OFF; see config.exploration). C1 hard-skips a saturated
        # sibling group from the frontier; C1b exposes struct-novelty for ranking.
        self._sibling_skip = sibling_skip
        self._sibling_skip_threshold = sibling_skip_threshold
        self._struct_novelty_rank = struct_novelty_rank

        # Telemetry (observation only — never read by exploration logic). The
        # hard skip is otherwise silent, so an ablation cannot prove the ON arm
        # actually differed from the OFF arm. Counts every skip event and logs
        # the first one per group (this runs once per candidate per step, so
        # per-event logging would flood).
        self.sibling_skips: int = 0
        self._sibling_skip_logged: set[tuple[str, str, str]] = set()

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
        # Read-only w.r.t. coverage/graph/rng — never alters legacy behaviour.
        element = from_state.find_by_signature(element_signature)
        if element is not None and element.struct_key:
            key = (from_state.page_key, element.struct_key, action_type)
            self._effects.setdefault(key, set()).add(to_state.page_key)
            self._effect_counts[key] = self._effect_counts.get(key, 0) + 1
            self._struct_seen.add((element.struct_key, action_type))

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
                    # C1: hard-skip a structural sibling once its group is proven
                    # inert, so has_unvisited / early-stop see the real frontier.
                    # Scrollables (negative index) are exempt. OFF ⇒ always False.
                    if element.index >= 0 and self._sibling_saturated(
                        state.page_key, element.struct_key, action_type
                    ):
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

    def struct_explored_anywhere(self, struct_key: str, action_type: str) -> bool:
        """True if this (struct_key, action_type) was exercised on any page.

        Structural analogue of :meth:`explored_anywhere` for C1b: an untried
        structure ranks above one already seen elsewhere. Empty struct_key never
        matches. Reads only the effect log (real transitions), so it is always
        populated even when the ranking knob is off.
        """
        return (struct_key, action_type) in self._struct_seen

    @property
    def struct_novelty_rank(self) -> bool:
        """Whether C1b struct-novelty ranking is enabled (read by the explorer)."""
        return self._struct_novelty_rank

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
        """Log how close structural sibling groups came to saturating.

        Telemetry only. Without it a session that never hard-skips is
        uninterpretable: "no skip because groups stayed under the threshold"
        and "no skip because every group reached two-or-more destinations"
        (permanent non-skip BY DESIGN) look identical from the outside.

        The effect log is populated regardless of the knobs, so this reports
        the same diagnostic for an OFF session — i.e. how many groups WOULD
        have saturated had the knob been on.
        """
        if not self._effect_counts:
            logger.info("[C1] effect-log summary: empty (no attributed transitions)")
            return
        would_saturate = sum(
            1
            for key, count in self._effect_counts.items()
            if count > self._sibling_skip_threshold and len(self._effects.get(key, set())) == 1
        )
        logger.info(
            "[C1] effect-log summary: groups={} would_saturate={} skips_fired={} "
            "threshold={} sibling_skip={}",
            len(self._effect_counts),
            would_saturate,
            self.sibling_skips,
            self._sibling_skip_threshold,
            self._sibling_skip,
        )
        ranked = sorted(self._effect_counts.items(), key=lambda kv: kv[1], reverse=True)
        for (page_key, struct_key, action), count in ranked[:top]:
            dests = len(self._effects.get((page_key, struct_key, action), set()))
            logger.info(
                "[C1]   fired={} dests={} action={} page={} struct={}",
                count,
                dests,
                action,
                page_key,
                struct_key,
            )

    def _blocked_pairs(self, page_key: str) -> set[tuple[str, str]]:
        return self._explored.get(page_key, set()) | self._nav_failed.get(page_key, set())

    def _sibling_saturated(
        self, page_key: str, struct_key: str, action_type: str
    ) -> bool:
        """True when a structural sibling group is proven inert (C1 hard-skip).

        A group saturates once the same (page_key, struct_key, action_type) has
        fired MORE than the threshold and every firing landed on a SINGLE
        destination page — the fingerprint of a long list whose rows all do the
        same thing. Two-or-more distinct destinations means the rows differ, so
        the group never saturates (permanent non-skip). OFF (default) or an empty
        struct_key never saturates.
        """
        if not self._sibling_skip or not struct_key:
            return False
        key = (page_key, struct_key, action_type)
        saturated = (
            self._effect_counts.get(key, 0) > self._sibling_skip_threshold
            and len(self._effects.get(key, set())) == 1
        )
        # Telemetry only — the return value above is already decided.
        if saturated:
            self.sibling_skips += 1
            if key not in self._sibling_skip_logged:
                self._sibling_skip_logged.add(key)
                logger.info(
                    "[C1] sibling skip fired: page={} struct={} action={} "
                    "fired={} dest={}",
                    page_key,
                    struct_key,
                    action_type,
                    self._effect_counts.get(key, 0),
                    next(iter(self._effects.get(key, set())), ""),
                )
        return saturated
