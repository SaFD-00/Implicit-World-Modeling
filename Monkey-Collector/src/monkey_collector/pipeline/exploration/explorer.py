"""LLM-guided exploration engine and the Explorer contract it fulfils.

``LLMGuidedExplorer`` replaced a legacy weighted-random explorer while keeping
the same public surface (the :class:`Explorer` Protocol) so the collection
loop, recovery helpers, and tests are unaffected.

Per step, :meth:`select_action` orchestrates:
  1. continue an in-progress navigation plan (one queued step), else
  2. take an unexplored action on the current screen, else
  3. plan the shortest route to an unexplored action elsewhere and start it, else
  4. fall back to back (or a safe tap on the first/root screen).

State abstraction, coverage tracking, same-function compression, and the
transition graph live in :class:`Memory`; multi-step routing lives in
:class:`Navigator`. This class wires them to the device via ADB.
"""

from __future__ import annotations

import random
import time
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from loguru import logger

from monkey_collector.domain.actions import (
    Action,
    InputText,
    LongPress,
    PressBack,
    PressHome,
    Swipe,
    Tap,
)
from monkey_collector.pipeline.exploration.action_mapper import ActionMapper
from monkey_collector.pipeline.exploration.memory import Memory
from monkey_collector.pipeline.exploration.navigator import Navigator
from monkey_collector.pipeline.exploration.state import (
    LONG_TOUCH,
    TOUCH,
    SemanticElement,
    SemanticState,
)
from monkey_collector.xml.ui_tree import UITree

if TYPE_CHECKING:
    from monkey_collector.adb import AdbClient
    from monkey_collector.pipeline.screen_matching.screen_matcher import ElementFamily
    from monkey_collector.pipeline.text_generator import TextGenerator


@runtime_checkable
class Explorer(Protocol):
    """The action-selection contract the collection loop depends on."""

    def select_action(
        self,
        ui_tree: UITree,
        step: int = -1,
        is_first_screen: bool = False,
        page_id: int | None = None,
        is_root_screen: bool = False,
    ) -> Action: ...

    def execute_action(self, action: Action) -> None: ...

    def set_screen_context(
        self, raw_xml: str, activity: str = "", package: str = ""
    ) -> None: ...

    def set_match_context(
        self, page_key: str, families: list[ElementFamily] | None
    ) -> None: ...

    def reset(self) -> None: ...

    def clear_excluded(self) -> None: ...

    def exclude_element(self, element_index: int) -> None: ...

    def has_unvisited(self, ui_tree: UITree, page_id: int | None) -> bool: ...

    def has_left_app(self, package: str) -> bool: ...

    def return_to_app(self, package: str) -> None: ...

    def recover(self, package: str) -> None: ...


class LLMGuidedExplorer:
    """Coverage-driven, navigation-capable explorer (LLM-Explorer port)."""

    def __init__(
        self,
        adb: AdbClient,
        text_generator: TextGenerator | None = None,
        config: dict | None = None,
        strategy: str = "GREEDY",
    ):
        config = config or {}
        self.adb = adb
        self._rng = random.Random(config.get("seed", 42))
        self._screen_width = config.get("screen_width", 1080)
        self._screen_height = config.get("screen_height", 1920)
        self._action_mapper = ActionMapper(text_generator=text_generator)
        # Traversal strategy (DFS | BFS | GREEDY) — only affects which unexplored
        # target the Navigator routes to; the route itself is always shortest-path.
        self._strategy = strategy.strip().upper()

        # Per-session exploration state (rebuilt by reset()).
        self._memory = Memory()
        self._navigator = Navigator(self._memory, self._rng, strategy=self._strategy)

        # Current screen context, set by the loop before each select_action.
        self._raw_xml = ""
        self._activity = ""
        self._package = ""

        # Element-set match context, set by the loop (set_match_context) when a
        # ScreenMatcher is active; "" / None drives the structural degrade path.
        self._page_key = ""
        self._families: list[ElementFamily] | None = None

        # Transition tracking across steps.
        self._current_state: SemanticState | None = None
        self._last_record: tuple[SemanticState, str, str] | None = None

    # -- context & lifecycle --------------------------------------------------

    def set_screen_context(
        self, raw_xml: str, activity: str = "", package: str = ""
    ) -> None:
        """Provide the current screen's raw XML and (optionally) activity/package.

        Activity/package are sticky: a no-change retry re-supplies only the XML,
        so the last known activity/package are retained.
        """
        self._raw_xml = raw_xml
        if activity:
            self._activity = activity
        if package:
            self._package = package

    def set_match_context(
        self, page_key: str, families: list[ElementFamily] | None
    ) -> None:
        """Provide the current screen's element-set page key and (new-page) families.

        Called by the collection loop once per new screen after the
        ``ScreenMatcher`` runs. Sticky across no-change retries (which re-supply
        only the XML), so the last match context is retained until the next
        screen. ``page_key=""`` keeps the structural ``structure_str`` fallback.
        """
        self._page_key = page_key
        self._families = families

    def reset(self) -> None:
        """Drop all per-session memory so each app session explores in isolation."""
        self._memory = Memory()
        self._navigator = Navigator(self._memory, self._rng, strategy=self._strategy)
        self._page_key = ""
        self._families = None
        self._current_state = None
        self._last_record = None

    # -- action selection -----------------------------------------------------

    def select_action(
        self,
        ui_tree: UITree,
        step: int = -1,
        is_first_screen: bool = False,
        page_id: int | None = None,
        is_root_screen: bool = False,
    ) -> Action:
        """Pick the next action for the current screen (see module docstring)."""
        current = SemanticState.from_screen(
            self._raw_xml, self._activity, self._package, page_key=self._page_key
        )
        self._current_state = current
        self._memory.record_state(current, self._families)

        # Attribute the previous action's outcome now that we see its result.
        if self._last_record is not None:
            prev_state, prev_signature, prev_action = self._last_record
            self._memory.record_transition(prev_state, prev_signature, prev_action, current)
            self._last_record = None

        # 1) Continue an in-progress navigation plan.
        if self._navigator.is_navigating():
            nav = self._navigator.next_action(current)
            if nav is not None:
                return self._emit(current, *nav)

        # 2) Take an unexplored action on the current screen.
        local = self._pick_unexplored(current)
        if local is not None:
            return self._emit(current, *local)

        # 3) Route to an unexplored action elsewhere.
        if self._navigator.plan_to_unexplored(current):
            nav = self._navigator.next_action(current)
            if nav is not None:
                return self._emit(current, *nav)

        # 4) Nothing left to try here or reachable — fall back.
        return self._fallback(current, is_first_screen, is_root_screen)

    def _pick_unexplored(
        self, state: SemanticState
    ) -> tuple[SemanticElement, str] | None:
        candidates = [
            (element, action_type)
            for _, element, action_type in self._memory.unexplored_actions([state])
        ]
        if not candidates:
            return None
        # Prefer non-long-press actions; long_touch is a low-value follow-up.
        primary = [c for c in candidates if c[1] != LONG_TOUCH]
        return self._rng.choice(primary or candidates)

    def _fallback(
        self,
        state: SemanticState,
        is_first_screen: bool,
        is_root_screen: bool,
    ) -> Action:
        """Choose a safe action when there is nothing unexplored to do.

        On the first/root screen a back press would exit the app, so tap an
        element (or a random point) instead; elsewhere press back to retreat.
        """
        self._last_record = None
        if is_first_screen or is_root_screen:
            actionable = state.actionable_elements()
            if actionable:
                element = self._rng.choice(actionable)
                action_type = TOUCH if TOUCH in element.allowed_actions else element.allowed_actions[0]
                return self._emit(state, element, action_type)
            return Tap(
                x=self._rng.randint(100, self._screen_width - 100),
                y=self._rng.randint(200, self._screen_height - 200),
            )
        return PressBack()

    def _emit(
        self,
        state: SemanticState,
        element: SemanticElement,
        action_type: str,
    ) -> Action:
        """Record the chosen action for later attribution and build it."""
        self._last_record = (state, element.signature, action_type)
        return self._action_mapper.to_domain_action(action_type, element, self._raw_xml)

    # -- collection-loop helpers ---------------------------------------------

    def exclude_element(self, element_index: int) -> None:
        """Permanently exclude an element from this screen (e.g. it caused no change).

        Marks every action on the element explored, so it leaves both the local
        frontier and any navigation plan — unlike a transient exclusion, this is
        coverage-correct: an element that does nothing here should not be retried.
        """
        if self._current_state is None or element_index < 0:
            return
        for element in self._current_state.elements:
            if element.index == element_index:
                for action_type in element.allowed_actions:
                    self._memory.mark_explored(
                        self._current_state.page_key, element.signature, action_type
                    )
                return

    def clear_excluded(self) -> None:
        """No-op: coverage is tracked permanently per structure, so there is no
        transient exclusion set to reset. Kept to satisfy the Explorer contract.
        """

    def has_unvisited(self, ui_tree: UITree, page_id: int | None) -> bool:
        """True if the current screen still has an unexplored action."""
        if self._current_state is None:
            return False
        return bool(self._memory.unexplored_actions([self._current_state]))

    # -- execution & recovery (carried over from the legacy explorer) ---------

    def execute_action(self, action: Action) -> None:
        """Execute an action on the device via ADB."""
        if isinstance(action, Tap):
            self.adb.tap(action.x, action.y)
        elif isinstance(action, Swipe):
            self.adb.swipe(action.x1, action.y1, action.x2, action.y2, action.duration_ms)
        elif isinstance(action, InputText):
            if action.x or action.y:
                self.adb.tap(action.x, action.y)
                time.sleep(0.3)
            self.adb.clear_text_field()
            self.adb.input_text(action.text)
            # Close the soft keyboard so the next screen is the body, not the IME.
            self.adb.press_back()
        elif isinstance(action, PressBack):
            self.adb.press_back()
        elif isinstance(action, PressHome):
            self.adb.press_home()
        elif isinstance(action, LongPress):
            self.adb.long_press(action.x, action.y, action.duration_ms)

    def has_left_app(self, package: str) -> bool:
        """Check whether the device is no longer in the target app."""
        try:
            current = self.adb.get_current_package()
            if not current:
                return False
            return current != package
        except Exception:
            return False

    def return_to_app(self, package: str) -> bool:
        """Return to the target app after leaving it.

        Returns True iff the app had to be (re)launched — i.e. an open_app
        happened — and False when a single Back was enough to land back in it.
        Clears the pending transition record so the excursion is never
        attributed as a routing-memory transition.
        """
        self._last_record = None
        try:
            self.adb.press_back()
            time.sleep(0.5)
            if self.adb.get_current_package() != package:
                self.adb.launch_app(package)
                time.sleep(3)
                return True
            return False
        except Exception:
            self.adb.launch_app(package)
            time.sleep(3)
            return True

    def recover(self, package: str) -> bool:
        """Recover from an error state by relaunching the app.

        Returns True iff the app was relaunched (an open_app), False on failure.
        Clears the pending transition record so the excursion is never
        attributed as a routing-memory transition.
        """
        self._last_record = None
        try:
            self.adb.press_home()
            time.sleep(1)
            self.adb.launch_app(package)
            time.sleep(3)
            return True
        except Exception:
            logger.error("Recovery failed for package %s", package)
            return False
