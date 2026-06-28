"""Main collection while-loop: dispatch signals, execute actions, track state."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger

from monkey_collector.domain.actions import Action
from monkey_collector.domain.page_graph import PageGraph
from monkey_collector.pipeline.recovery import (
    MAX_EMPTY_UI_RETRIES,
    MAX_EXTERNAL_APP_RETRIES,
    MAX_NO_CHANGE_RETRIES,
    MAX_SAME_PAGE_STEPS,
    describe_action_element,
    nudge_static_screen,
    safe_press_back,
    tap_random_fallback,
)
from monkey_collector.pipeline.screen_guard import (
    find_dialog_button,
    is_keyboard,
    is_permission_dialog,
    is_system_screen,
)
from monkey_collector.xml.ui_tree import UITree

if TYPE_CHECKING:
    from monkey_collector.pipeline.collector import Collector


@dataclass
class CollectionState:
    """Mutable state for a single collection session loop."""

    step: int = 0
    max_step: int = 0
    total_actions: int = 0
    timeout_count: int = 0
    no_change_retries: int = 0
    external_app_count: int = 0
    empty_ui_retries: int = 0
    last_action: Action | None = None
    last_ui_tree: UITree | None = None
    last_raw_xml: str | None = None
    is_first_screen: bool = False
    current_page_id: int | None = None
    root_page_id: int | None = None
    same_page_count: int = 0
    page_graph: PageGraph = field(default_factory=PageGraph)


def _is_root_screen(state: CollectionState) -> bool:
    """True when the current page is the session's root (first in-app) page.

    Back from the root only exits the app to the launcher, so every deliberate
    back is treated like the first screen there (tap instead of back).
    """
    return (
        state.current_page_id is not None
        and state.current_page_id == state.root_page_id
    )


def run_collection_loop(
    collector: Collector,
    state: CollectionState,
    package: str,
) -> None:
    """Run the main collection while-loop, mutating state until session ends."""
    max_timeouts = 5

    # Discard signals left over from the previous session. The sequential
    # run_queue reuses one server/client connection, so a trailing "finish"
    # from the prior session's teardown can still sit in the queue when this
    # freshly-launched session starts. Without this clear, the first
    # get_latest_signal would return that stale finish and end the new session
    # at step 0 — and it cascades to every later app in the queue.
    collector.server.clear_signal_queue()

    while state.step < state.max_step:
        try:
            result = collector.server.get_latest_signal(timeout=collector.xml_timeout)

            if result is None:
                state.timeout_count += 1
                logger.warning(
                    f"Step {state.step}: signal timeout "
                    f"({state.timeout_count}/{max_timeouts})"
                )
                if state.timeout_count >= max_timeouts:
                    logger.error("Too many timeouts, ending session")
                    break
                # A timeout means no screenshot/XML arrived. If we drifted out
                # of the target app (e.g. a system role screen like
                # permissioncontroller that emits no accessibility events and
                # can't be closed via force_stop), relaunch to escape it.
                # Otherwise nudge a static in-app screen with a tap.
                if collector.explorer.has_left_app(package):
                    logger.warning(
                        f"Step {state.step}: left target app during timeout, "
                        f"returning to {package}"
                    )
                    collector.explorer.return_to_app(package)
                else:
                    # Still in-app but no event fired: drive a real element to
                    # provoke a transition instead of a blind center tap.
                    nudge_static_screen(
                        collector.adb, state.last_ui_tree, state.timeout_count
                    )
                state.step += 1
                continue

            signal_type = result[0]

            if signal_type == "finish":
                logger.info("Received finish signal, ending session")
                break

            if signal_type == "no_change":
                if _handle_no_change(collector, state, package):
                    break
                continue

            if signal_type == "external_app":
                if _handle_external_app(collector, state, package):
                    break
                continue

            # signal_type == "xml" — screen changed
            state.timeout_count = 0
            state.no_change_retries = 0
            collector.explorer.clear_excluded()

            _, xml_str, meta = result
            if _process_xml_signal(collector, state, package, xml_str, meta):
                continue

        except Exception as e:
            logger.error(f"Step {state.step}: error - {e}")
            try:
                collector.explorer.recover(package)
            except Exception:
                pass
            state.step += 1


def _handle_no_change(
    collector: Collector,
    state: CollectionState,
    package: str,
) -> bool:
    """Handle a no-change signal. Returns True if session should end."""
    state.no_change_retries += 1
    logger.info(
        f"Step {state.step}: no visual change "
        f"(retry {state.no_change_retries}/{MAX_NO_CHANGE_RETRIES})"
    )

    if state.last_action is not None and state.last_action.element_index >= 0:
        collector.explorer.exclude_element(state.last_action.element_index)

    if state.no_change_retries >= MAX_NO_CHANGE_RETRIES:
        if state.is_first_screen or _is_root_screen(state):
            logger.warning(
                f"Step {state.step}: {MAX_NO_CHANGE_RETRIES} "
                f"no-change retries, on first/root screen — tap instead of back"
            )
            tap_random_fallback(collector.adb)
        else:
            logger.warning(
                f"Step {state.step}: {MAX_NO_CHANGE_RETRIES} "
                f"no-change retries, pressing back"
            )
            safe_press_back(collector.adb, collector.explorer, package)
        collector.server.clear_signal_queue()
        state.no_change_retries = 0
        collector.explorer.clear_excluded()
        state.last_action = None
        state.last_ui_tree = None
        time.sleep(collector.action_delay)
        state.step += 1
        return False

    if state.last_ui_tree is not None and len(state.last_ui_tree) > 0:
        if state.last_raw_xml:
            collector.explorer.set_raw_xml(state.last_raw_xml)
        action = collector.explorer.select_action(
            state.last_ui_tree, state.step, is_first_screen=state.is_first_screen,
            page_id=state.current_page_id, is_root_screen=_is_root_screen(state),
        )
        logger.info(
            f"Step {state.step}: retry {action.action_type} "
            f"(element_index={action.element_index})"
        )
        collector.explorer.execute_action(action)
        state.last_action = action
        state.total_actions += 1

        event = action.to_dict()
        event["step"] = state.step
        event["no_change_retry"] = True
        collector.writer.log_event(event)

        time.sleep(collector.action_delay)
    else:
        if state.is_first_screen or _is_root_screen(state):
            logger.info(
                f"Step {state.step}: no UI tree, on first/root screen — tap instead of back"
            )
            tap_random_fallback(collector.adb)
        else:
            safe_press_back(collector.adb, collector.explorer, package)
        state.no_change_retries = 0
        collector.explorer.clear_excluded()
        state.last_action = None
        time.sleep(collector.action_delay)
    state.step += 1
    return False


def _handle_external_app(
    collector: Collector,
    state: CollectionState,
    package: str,
) -> bool:
    """Handle an external_app signal. Returns True if session should end."""
    state.external_app_count += 1
    logger.warning(
        f"Step {state.step}: external app detected "
        f"({state.external_app_count}/{MAX_EXTERNAL_APP_RETRIES})"
    )
    if state.external_app_count >= MAX_EXTERNAL_APP_RETRIES:
        logger.error("Too many external app detections, ending session")
        return True
    try:
        if state.external_app_count <= 3:
            collector.explorer.return_to_app(package)
        else:
            collector.explorer.recover(package)
    except Exception as e:
        logger.error(f"Recovery attempt failed: {e}")
    collector.server.clear_signal_queue()
    time.sleep(collector.action_delay)
    return False


def _handle_permission_dialog(
    collector: Collector,
    state: CollectionState,
    xml_str: str,
) -> None:
    """Act on a permission/install dialog: tap a grant/dismiss button or back."""
    ui_tree = UITree.from_xml_string(xml_str)
    button = find_dialog_button(ui_tree)
    if button is not None:
        cx, cy = button.center
        logger.info(
            f"Step {state.step}: permission dialog, tapping '{button.display_name}'"
        )
        collector.adb.tap(cx, cy)
    else:
        logger.info(
            f"Step {state.step}: permission dialog, no button matched, pressing back"
        )
        collector.adb.press_back()
    collector.server.clear_signal_queue()
    state.last_action = None
    state.last_ui_tree = None
    time.sleep(collector.action_delay)


def _process_xml_signal(
    collector: Collector,
    state: CollectionState,
    package: str,
    xml_str: str,
    meta: dict,
) -> bool:
    """Process an XML signal (new screen). Returns True if loop should `continue`."""
    top_package = meta.get("top_package", "")
    state.is_first_screen = meta.get("is_first_screen", False)
    activity_name = meta.get("activity_name", "")

    # Keyboard (SoftInputWindow) is an overlay, not a page: dismiss it so the
    # next signal shows the underlying screen, and do NOT register it as a page
    # (it would pollute same-page / coverage statistics).
    if is_keyboard(activity_name):
        logger.info(f"Step {state.step}: keyboard window, dismissing")
        safe_press_back(collector.adb, collector.explorer, package)
        collector.server.clear_signal_queue()
        state.last_action = None
        state.last_ui_tree = None
        time.sleep(collector.action_delay)
        state.step += 1
        return True

    # Permission / install grant dialog: act on it (grant > dismiss) instead of
    # burning steps skipping it as stale XML — otherwise we loop here forever.
    if is_permission_dialog(top_package):
        _handle_permission_dialog(collector, state, xml_str)
        state.step += 1
        return True

    # Drifted into another system screen we cannot drive: relaunch the target.
    if top_package and is_system_screen(top_package) and top_package != package:
        logger.warning(
            f"Step {state.step}: in system screen {top_package}, "
            f"returning to {package}"
        )
        collector.explorer.return_to_app(package)
        collector.server.clear_signal_queue()
        state.step += 1
        return True

    if top_package and top_package != package:
        logger.info(
            f"Step {state.step}: stale XML from {top_package} "
            f"(expected {package}), skipping"
        )
        state.step += 1
        return True

    if not activity_name:
        activity_name = collector.adb.get_current_activity()

    if collector._activity_tracker is not None:
        entry = collector._activity_tracker.record(activity_name, state.step)
        logger.debug(
            f"Activity coverage: {entry['coverage']:.2%} "
            f"({entry['unique_visited']}/{entry['total_activities']})"
        )

    previous_page_id = state.current_page_id
    pages_before = len(state.page_graph.nodes)
    state.current_page_id = state.page_graph.get_or_create_page(
        activity_name, xml_str, state.step,
    )
    # The first in-app page registered this session is the root (back from it
    # only exits to the launcher); pin it once for back-suppression.
    if state.root_page_id is None:
        state.root_page_id = state.current_page_id
    discovered_new_page = len(state.page_graph.nodes) > pages_before
    if previous_page_id is not None and state.last_action is not None:
        element_info = describe_action_element(state.last_action, state.last_ui_tree)
        state.page_graph.add_transition(
            from_page=previous_page_id,
            to_page=state.current_page_id,
            action_type=state.last_action.action_type,
            element_info=element_info,
            step=state.step,
        )

    if previous_page_id is not None and state.current_page_id == previous_page_id:
        state.same_page_count += 1
    else:
        state.same_page_count = 0

    if state.same_page_count >= MAX_SAME_PAGE_STEPS:
        # Only leave the page once there is nothing new left to try here.
        # While unvisited elements remain, fall through to normal (unvisited-
        # first) selection so we exhaust the page before navigating away.
        same_page_tree = UITree.from_xml_string(xml_str)
        if not collector.explorer.has_unvisited(same_page_tree, state.current_page_id):
            logger.warning(
                f"Step {state.step}: stuck on page {state.current_page_id} "
                f"for {state.same_page_count} steps, forcing back"
            )
            if state.is_first_screen or _is_root_screen(state):
                tap_random_fallback(collector.adb)
            else:
                safe_press_back(collector.adb, collector.explorer, package)
            collector.server.clear_signal_queue()
            collector.explorer.clear_excluded()
            state.same_page_count = 0
            state.last_action = None
            state.last_ui_tree = None
            time.sleep(collector.action_delay)
            state.step += 1
            return True

    # Shared step for cost attribution across every LLM consumer this step
    # (input text generation + screen grouping).
    if collector._llm_client is not None:
        collector._llm_client.set_step(state.step)

    if collector._latest_screenshot:
        collector.writer.save_screenshot(collector._latest_screenshot)
        collector._latest_screenshot = None
    collector.writer.save_xml(xml_str)

    # Annotate the just-saved screen with LLM semantic element grouping
    # ("화면 나누기"). Best-effort: a failure here must never break collection.
    if collector._screen_grouper is not None:
        try:
            grouping = collector._screen_grouper.group(xml_str)
            collector.writer.save_groups(grouping)
        except Exception as e:
            logger.warning(f"Step {state.step}: screen grouping failed ({e})")

    ui_tree = UITree.from_xml_string(xml_str)
    if len(ui_tree) == 0:
        state.empty_ui_retries += 1
        if state.empty_ui_retries <= MAX_EMPTY_UI_RETRIES:
            logger.info(
                f"Step {state.step}: no UI elements, waiting for app load "
                f"({state.empty_ui_retries}/{MAX_EMPTY_UI_RETRIES})"
            )
            time.sleep(1.0)
            state.step += 1
            state.last_ui_tree = None
            state.last_action = None
            return True
        state.empty_ui_retries = 0
        if state.is_first_screen or _is_root_screen(state):
            logger.warning(
                f"Step {state.step}: no UI elements, on first/root screen — tap instead of back"
            )
            tap_random_fallback(collector.adb)
        else:
            logger.warning(
                f"Step {state.step}: no UI elements, pressing back"
            )
            safe_press_back(collector.adb, collector.explorer, package)
        state.last_ui_tree = None
        state.last_action = None
        state.step += 1
        return True

    state.empty_ui_retries = 0
    # Only reset the external-app counter on genuine progress (a brand-new page).
    # Resetting on every in-app frame let an external↔return loop run forever
    # because each return landed on an already-known page.
    if discovered_new_page:
        state.external_app_count = 0

    collector.explorer.set_raw_xml(xml_str)
    action = collector.explorer.select_action(
        ui_tree, state.step, is_first_screen=state.is_first_screen,
        page_id=state.current_page_id, is_root_screen=_is_root_screen(state),
    )
    logger.info(
        f"Step {state.step}: {action.action_type} "
        f"(element_index={action.element_index})"
    )

    collector.explorer.execute_action(action)
    state.total_actions += 1

    collector.server.clear_signal_queue()

    state.last_action = action
    state.last_ui_tree = ui_tree
    state.last_raw_xml = xml_str

    event = action.to_dict()
    event["step"] = state.step
    event["activity_name"] = activity_name
    collector.writer.log_event(event)

    time.sleep(collector.action_delay)
    state.step += 1
    return False
