"""Recovery helpers: no-change retry, external app, empty UI, stuck page."""

from __future__ import annotations

from typing import TYPE_CHECKING

from monkey_collector.adb import AdbClient
from monkey_collector.domain.actions import Action
from monkey_collector.xml.ui_tree import UITree

if TYPE_CHECKING:
    from monkey_collector.pipeline.explorer import SmartExplorer

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
    explorer: SmartExplorer,
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
