"""Recovery helpers: no-change retry, external app, empty UI, stuck page."""

from __future__ import annotations

from typing import TYPE_CHECKING

from monkey_collector.adb import AdbClient
from monkey_collector.domain.actions import Action
from monkey_collector.xml.ui_tree import UITree

if TYPE_CHECKING:
    from monkey_collector.pipeline.exploration import Explorer

MAX_NO_CHANGE_RETRIES = 3
MAX_EXTERNAL_APP_RETRIES = 10
MAX_SAME_PAGE_STEPS = 5
MAX_EMPTY_UI_RETRIES = 2
# Suppress press_back for the first N steps of a session so an early back does
# not exit the app before any data is collected (the cause of 1-2 step sessions).
FIRST_STEPS_NO_BACK = 3


def tap_random_fallback(adb: AdbClient) -> None:
    """Tap center of screen as a fallback when back is suppressed on first screen."""
    try:
        w, h = adb.get_device_resolution()
        adb.tap(w // 2, h // 2)
    except Exception:
        pass


def nudge_static_screen(adb: AdbClient, ui_tree: UITree | None, attempt: int = 0) -> None:
    """Provoke an accessibility event on a static, still-in-app screen.

    A signal timeout while the device is still in the target app usually means
    the screen settled without firing a window-change event, so the app never
    pushed fresh XML. A blind center tap often lands on dead space and produces
    no event either, so timeouts keep accumulating until the session dies.
    Interacting with a real element (or scrolling) is far more likely to cause a
    transition that makes the app emit a new accessibility event.
    """
    try:
        if ui_tree is not None:
            clickable = ui_tree.get_clickable_elements()
            if clickable:
                elem = clickable[attempt % len(clickable)]
                cx, cy = elem.center
                adb.tap(cx, cy)
                return
            scrollable = ui_tree.get_scrollable_elements()
            if scrollable:
                cx, cy = scrollable[0].center
                adb.swipe(cx, cy + 200, cx, cy - 200, 300)
                return
        tap_random_fallback(adb)
    except Exception:
        pass


def describe_action_element(action: Action, ui_tree: UITree | None) -> str:
    """Describe the target element of an action briefly."""
    if ui_tree is None or action.element_index < 0:
        return action.action_type
    if action.element_index < len(ui_tree):
        elem = ui_tree.elements[action.element_index]
        return elem.display_name
    return action.action_type


def safe_press_back(
    adb: AdbClient,
    explorer: Explorer,
    package: str,
) -> None:
    """Press back and recover if the action caused the app to exit."""
    import time

    from loguru import logger

    adb.press_back()
    time.sleep(0.5)
    if explorer.has_left_app(package):
        logger.warning("press_back caused app exit, recovering")
        explorer.return_to_app(package)
