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
# Maximum number of re-initialization attempts before giving up on a session.
# A reinit triggers when MAX_EXTERNAL_APP_RETRIES / MAX_SIGNAL_TIMEOUTS is hit; each
# reinit resets that counter and force-relaunches the target app. If the limit
# below is exhausted the session ends as before.
#
# The reinit count itself is forgiven (reset to 0) once the session has made
# REINIT_FORGIVE_STEPS of genuine forward progress since the last reinit of its
# kind, so a session that recovered cleanly and ran fine for a while isn't
# killed by a later unrelated blip.
#
# Forgiveness alone is not enough for every app: some screens (observed on
# Calendar's back-navigation from certain views, and a Music Player permission
# dialog) put the target app into a stuck "loading" state — near-empty XML,
# no further accessibility events — that only recovers via the full
# MAX_SIGNAL_TIMEOUTS force-relaunch cycle, and does so reliably every time. But
# it can recur every 10-25 steps, well inside REINIT_FORGIVE_STEPS, so the old
# MAX_TIMEOUT_REINITS=3 budget was exhausted (and the session killed) after
# only 2-4 minutes even though every single relaunch was in fact working. The
# real backstop for a session that is truly, permanently stuck (relaunch never
# helps) is `idle_iterations` in collection_loop.py — it increments on every
# loop pass and is reset only by a genuine step, so it fires regardless of
# this budget. That leaves this counter free to be generous: it exists to cap
# wasted wall-clock on a dead session, not to police apps that are merely slow
# to recover.
MAX_TIMEOUT_REINITS = 20
MAX_EXTERNAL_REINITS = 10
REINIT_FORGIVE_STEPS = 15
# Consecutive signal timeouts before escalating to a force-relaunch. With
# collection.signal_timeout_sec=12s this caps one stuck episode's worst-case
# wait at 12s×3=36s before the reinit cycle kicks in (nudge on timeouts 1-2,
# relaunch on the 3rd).
MAX_SIGNAL_TIMEOUTS = 3


def tap_random_fallback(adb: AdbClient) -> None:
    """Tap center of screen as a fallback when back is suppressed on first screen."""
    try:
        w, h = adb.get_device_resolution()
        adb.tap(w // 2, h // 2)
    except Exception:
        pass


def relaunch_app_fallback(adb: AdbClient, package: str) -> None:
    """Force-stop and relaunch the app as a fallback when back is suppressed.

    A blind center tap (the previous fallback) can land on a dead zone of an
    unexpected overlay — e.g. a GMS account-switcher sheet opened by a stray
    tap on a toolbar avatar — and loop forever without ever changing the
    screen, since the same coordinate is retried every time. Relaunching
    deterministically returns to the app's launcher activity instead.
    """
    try:
        adb.force_stop(package)
        adb.launch_app(package)
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
) -> bool:
    """Press back and recover if the action caused the app to exit.

    Returns True iff the back press left the target app (and recovery ran) —
    the caller can then remember that back exits from the current page, so a
    later deliberate back on the same page is suppressed. Returns False when
    back stayed inside the app.
    """
    import time

    from loguru import logger

    adb.press_back()
    time.sleep(0.5)
    if explorer.has_left_app(package):
        logger.warning("press_back caused app exit, recovering")
        explorer.return_to_app(package)
        return True
    return False
