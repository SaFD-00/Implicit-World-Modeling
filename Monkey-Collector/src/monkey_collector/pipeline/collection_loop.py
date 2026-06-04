"""Main collection while-loop: dispatch signals, execute actions, track state."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger

from monkey_collector.domain.actions import Action
from monkey_collector.domain.page_graph import PageGraph
from monkey_collector.xml.ui_tree import UITree
from monkey_collector.pipeline.recovery import (
    MAX_EMPTY_UI_RETRIES,
    MAX_EXTERNAL_APP_RETRIES,
    MAX_NO_CHANGE_RETRIES,
    MAX_SAME_PAGE_STEPS,
    describe_action_element,
    safe_press_back,
    tap_random_fallback,
)

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
    same_page_count: int = 0
    page_graph: PageGraph = field(default_factory=PageGraph)


def run_collection_loop(
    collector: Collector,
    state: CollectionState,
    package: str,
) -> None:
    """Run the main collection while-loop, mutating state until session ends."""
    max_timeouts = 5

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
                    tap_random_fallback(collector.adb)
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
        if state.is_first_screen:
            logger.warning(
                f"Step {state.step}: {MAX_NO_CHANGE_RETRIES} "
                f"no-change retries, on first screen — tap instead of back"
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
        if state.is_first_screen:
            logger.info(
                f"Step {state.step}: no UI tree, on first screen — tap instead of back"
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

    if top_package and top_package != package:
        logger.info(
            f"Step {state.step}: stale XML from {top_package} "
            f"(expected {package}), skipping"
        )
        state.step += 1
        return True

    activity_name = meta.get("activity_name", "")
    if not activity_name:
        activity_name = collector.adb.get_current_activity()

    if collector._activity_tracker is not None:
        entry = collector._activity_tracker.record(activity_name, state.step)
        logger.debug(
            f"Activity coverage: {entry['coverage']:.2%} "
            f"({entry['unique_visited']}/{entry['total_activities']})"
        )

    previous_page_id = state.current_page_id
    state.current_page_id = state.page_graph.get_or_create_page(
        activity_name, xml_str, state.step,
    )
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
        logger.warning(
            f"Step {state.step}: stuck on page {state.current_page_id} "
            f"for {state.same_page_count} steps, forcing back"
        )
        if state.is_first_screen:
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

    if collector._text_generator and hasattr(collector._text_generator, "set_step"):
        collector._text_generator.set_step(state.step)

    if collector._latest_screenshot:
        collector.writer.save_screenshot(collector._latest_screenshot)
        collector._latest_screenshot = None
    collector.writer.save_xml(xml_str)

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
        if state.is_first_screen:
            logger.warning(
                f"Step {state.step}: no UI elements, on first screen — tap instead of back"
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
    state.external_app_count = 0

    collector.explorer.set_raw_xml(xml_str)
    action = collector.explorer.select_action(
        ui_tree, state.step, is_first_screen=state.is_first_screen,
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
