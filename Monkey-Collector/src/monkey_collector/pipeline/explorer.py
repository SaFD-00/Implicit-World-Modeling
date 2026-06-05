"""Smart Explorer with XML-aware action selection.

Unlike random monkey (adb shell monkey), SmartExplorer:
1. Parses XML to identify interactable elements
2. Detects EditText fields and fills them with text
3. Detects app exit and automatically returns
4. Uses weighted action selection (tap 40%, back 20%, swipe 20%, etc.)
"""

from __future__ import annotations

import random
import time
from typing import Any

from loguru import logger

from monkey_collector.adb import AdbClient
from monkey_collector.domain.actions import (
    Action,
    InputText,
    LongPress,
    PressBack,
    PressHome,
    Swipe,
    Tap,
)
from monkey_collector.pipeline.recovery import FIRST_STEPS_NO_BACK
from monkey_collector.pipeline.text_generator import TextGenerator
from monkey_collector.xml.ui_tree import UITree

# Default action weights
DEFAULT_WEIGHTS: dict[str, float] = {
    "tap": 0.40,
    "press_back": 0.20,
    "swipe": 0.20,
    "input_text": 0.10,
    "long_press": 0.10,
    "press_home": 0.00,
}

# Sample texts for EditText filling
SAMPLE_TEXTS = [
    "Hello World",
    "Test Note",
    "Meeting at 3pm",
    "Shopping list",
    "Important memo",
    "John Doe",
    "test@example.com",
    "12345",
    "New item",
    "Quick note",
]


class SmartExplorer:
    """XML-aware action selector and executor for Android apps."""

    def __init__(
        self,
        adb: AdbClient,
        config: dict[str, Any] | None = None,
        text_generator: TextGenerator | None = None,
    ):
        config = config or {}
        self.adb = adb
        self.action_weights = config.get("action_weights", DEFAULT_WEIGHTS)
        self.action_delay_ms = config.get("action_delay_ms", 500)
        self.screen_width = config.get("screen_width", 1080)
        self.screen_height = config.get("screen_height", 1920)
        self.seed = config.get("seed", 42)
        self.sample_texts = config.get("sample_texts", SAMPLE_TEXTS)
        self._rng = random.Random(self.seed)
        self._excluded_elements: set[int] = set()
        self._text_generator = text_generator
        self._last_raw_xml: str | None = None
        # page_id -> set of element signatures already acted on (coverage memory)
        self._visited_signatures: dict[int, set[str]] = {}

    def exclude_element(self, element_index: int) -> None:
        """Mark an element as tried (no screen change). Excluded from future selection."""
        if element_index >= 0:
            self._excluded_elements.add(element_index)

    def clear_excluded(self) -> None:
        """Reset excluded elements. Call when screen actually changes."""
        self._excluded_elements.clear()

    def get_excluded_count(self) -> int:
        return len(self._excluded_elements)

    def set_raw_xml(self, raw_xml: str) -> None:
        """Store the latest raw XML for LLM-based text generation."""
        self._last_raw_xml = raw_xml

    @staticmethod
    def _element_signature(elem) -> str:
        """Stable, coordinate-free identity for an element within a page."""
        return f"{elem.resource_id}|{elem.text}|{elem.short_class}"

    def _prefer_unvisited(self, elements: list, page_id: int | None) -> list:
        """Return only not-yet-acted-on elements for *page_id*, if any exist.

        Falls back to the full list once every candidate has been visited.
        """
        if page_id is None or not elements:
            return elements
        visited = self._visited_signatures.get(page_id, set())
        unvisited = [e for e in elements if self._element_signature(e) not in visited]
        return unvisited if unvisited else elements

    def has_unvisited(self, ui_tree: UITree, page_id: int | None) -> bool:
        """True if any clickable element on *page_id* has not been acted on yet."""
        if page_id is None:
            return False
        visited = self._visited_signatures.get(page_id, set())
        for e in ui_tree.get_clickable_elements():
            if e.index in self._excluded_elements:
                continue
            if self._element_signature(e) not in visited:
                return True
        return False

    def _mark_visited(self, action: Action, ui_tree: UITree, page_id: int | None) -> None:
        """Record the element targeted by *action* as visited for *page_id*."""
        if page_id is None or action.element_index < 0:
            return
        for e in ui_tree.elements:
            if e.index == action.element_index:
                self._visited_signatures.setdefault(page_id, set()).add(
                    self._element_signature(e)
                )
                return

    def select_action(
        self,
        ui_tree: UITree,
        step: int = -1,
        is_first_screen: bool = False,
        page_id: int | None = None,
        is_root_screen: bool = False,
    ) -> Action:
        """Select an action based on UI state and weights.

        When *page_id* is given, tap/long-press/input candidates that have not
        yet been acted on for that page are preferred, so exploration spreads
        across the screen instead of repeatedly hitting the same element.

        When *is_root_screen* is True, press_back is suppressed: back from the
        app's root page only exits to the launcher (never useful for
        exploration) and causes external bounce-back churn.
        """
        clickable = ui_tree.get_clickable_elements()
        editable = ui_tree.get_editable_elements()
        scrollable = ui_tree.get_scrollable_elements()

        # Filter out elements that were tried but produced no screen change
        if self._excluded_elements:
            clickable = [e for e in clickable if e.index not in self._excluded_elements]
            editable = [e for e in editable if e.index not in self._excluded_elements]
            scrollable = [e for e in scrollable if e.index not in self._excluded_elements]

        # Prefer not-yet-visited targets for coverage-driven exploration
        clickable = self._prefer_unvisited(clickable, page_id)
        editable = self._prefer_unvisited(editable, page_id)

        # Build available actions with adjusted weights
        weights = dict(self.action_weights)

        # 첫 화면/세션 초반 N스텝/루트 화면에는 press_back 금지 (앱 이탈·churn 방지)
        if is_first_screen or is_root_screen or (0 <= step < FIRST_STEPS_NO_BACK):
            weights["press_back"] = 0.0

        # If there are editable fields, boost input_text weight
        if editable:
            weights["input_text"] = max(weights.get("input_text", 0.1), 0.25)

        # If no clickable elements, reduce tap weight
        if not clickable:
            weights["tap"] = 0.05

        # If no scrollable elements, reduce swipe weight
        if not scrollable:
            weights["swipe"] = 0.05

        # Normalize weights
        total = sum(weights.values())
        if total == 0:
            # During the first/root screen / early steps, never fall back to
            # back (it could exit the app); tap instead.
            if is_first_screen or is_root_screen or (0 <= step < FIRST_STEPS_NO_BACK):
                if clickable:
                    elem = self._rng.choice(clickable)
                    cx, cy = elem.center
                    action = Tap(x=cx, y=cy, element_index=elem.index)
                    self._mark_visited(action, ui_tree, page_id)
                    return action
                return Tap(
                    x=self._rng.randint(100, self.screen_width - 100),
                    y=self._rng.randint(200, self.screen_height - 200),
                )
            return PressBack()

        normalized = {k: v / total for k, v in weights.items()}

        # Weighted selection
        action_type = self._weighted_choice(normalized)

        action = self._create_action(action_type, clickable, editable, scrollable)
        self._mark_visited(action, ui_tree, page_id)
        return action

    def execute_action(self, action: Action) -> None:
        """Execute an action on the device via ADB."""
        if isinstance(action, Tap):
            self.adb.tap(action.x, action.y)
        elif isinstance(action, Swipe):
            self.adb.swipe(
                action.x1, action.y1, action.x2, action.y2, action.duration_ms
            )
        elif isinstance(action, InputText):
            if action.x or action.y:
                self.adb.tap(action.x, action.y)
                time.sleep(0.3)
            self.adb.clear_text_field()
            self.adb.input_text(action.text)
            # Close the soft keyboard so the next action sees the body content
            # instead of getting stuck in the SoftInputWindow overlay.
            self.adb.press_back()
        elif isinstance(action, PressBack):
            self.adb.press_back()
        elif isinstance(action, PressHome):
            self.adb.press_home()
        elif isinstance(action, LongPress):
            self.adb.long_press(action.x, action.y, action.duration_ms)

    def has_left_app(self, package: str) -> bool:
        """Check if the device is no longer in the target app."""
        try:
            current = self.adb.get_current_package()
            if not current:
                return False
            return current != package
        except Exception:
            return False

    def return_to_app(self, package: str) -> None:
        """Return to the target app after leaving it."""
        try:
            self.adb.press_back()
            time.sleep(0.5)
            current = self.adb.get_current_package()
            if current != package:
                self.adb.launch_app(package)
                time.sleep(3)
        except Exception:
            self.adb.launch_app(package)
            time.sleep(3)

    def recover(self, package: str) -> None:
        """Recover from an error state."""
        try:
            self.adb.press_home()
            time.sleep(1)
            self.adb.launch_app(package)
            time.sleep(3)
        except Exception:
            logger.error("Recovery failed for package %s", package)

    def _create_action(
        self,
        action_type: str,
        clickable: list,
        editable: list,
        scrollable: list,
    ) -> Action:
        """Create a concrete action instance."""
        if action_type == "tap" and clickable:
            elem = self._rng.choice(clickable)
            cx, cy = elem.center
            return Tap(x=cx, y=cy, element_index=elem.index)

        if action_type == "input_text" and editable:
            empty = [e for e in editable if not e.text.strip()]
            pool = empty if empty else editable
            elem = self._rng.choice(pool)
            cx, cy = elem.center
            if self._text_generator and self._last_raw_xml:
                text = self._text_generator.generate(elem, self._last_raw_xml)
            else:
                text = self._rng.choice(self.sample_texts)
            return InputText(text=text, x=cx, y=cy, element_index=elem.index)

        if action_type == "swipe":
            if scrollable:
                elem = self._rng.choice(scrollable)
                cx, cy = elem.center
                return Swipe(
                    x1=cx, y1=cy + 200,
                    x2=cx, y2=cy - 200,
                    duration_ms=300,
                    element_index=elem.index,
                )
            else:
                mid_x = self.screen_width // 2
                return Swipe(
                    x1=mid_x, y1=self.screen_height * 3 // 4,
                    x2=mid_x, y2=self.screen_height // 4,
                    duration_ms=300,
                )

        if action_type == "long_press" and clickable:
            elem = self._rng.choice(clickable)
            cx, cy = elem.center
            return LongPress(x=cx, y=cy, element_index=elem.index)

        if action_type == "press_home":
            return PressHome()

        if action_type == "press_back":
            return PressBack()

        # Fallback: random tap on a clickable element or random coordinates
        if clickable:
            elem = self._rng.choice(clickable)
            cx, cy = elem.center
            return Tap(x=cx, y=cy, element_index=elem.index)

        return Tap(
            x=self._rng.randint(100, self.screen_width - 100),
            y=self._rng.randint(200, self.screen_height - 200),
        )

    def _weighted_choice(self, weights: dict[str, float]) -> str:
        """Weighted random selection from action types."""
        items = list(weights.items())
        values = [w for _, w in items]
        chosen = self._rng.choices([k for k, _ in items], weights=values, k=1)
        return chosen[0]
