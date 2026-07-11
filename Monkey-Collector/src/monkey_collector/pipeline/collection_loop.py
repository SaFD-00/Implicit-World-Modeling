"""Main collection while-loop: dispatch signals, execute actions, track state."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger

from monkey_collector.domain.actions import Action
from monkey_collector.domain.page_graph import PageGraph
from monkey_collector.pipeline.recovery import (
    MAX_EMPTY_UI_RETRIES,
    MAX_EXTERNAL_APP_RETRIES,
    MAX_EXTERNAL_REINITS,
    MAX_NO_CHANGE_RETRIES,
    MAX_SAME_PAGE_STEPS,
    MAX_SIGNAL_TIMEOUTS,
    MAX_TIMEOUT_REINITS,
    REINIT_FORGIVE_STEPS,
    describe_action_element,
    nudge_static_screen,
    relaunch_app_fallback,
    safe_press_back,
)
from monkey_collector.pipeline.screen_guard import (
    find_dialog_button,
    is_keyboard,
    is_launcher,
    is_permission_dialog,
    is_system_screen,
)
from monkey_collector.xml.structured_parser import encode_with_bounds
from monkey_collector.xml.ui_tree import UITree

if TYPE_CHECKING:
    from monkey_collector.pipeline.collector import Collector
    from monkey_collector.pipeline.screen_matching.screen_matcher import ScreenMatch


@dataclass
class CollectionState:
    """Mutable state for a single collection session loop."""

    step: int = 0
    max_step: int = 0
    # Session end condition. "steps" (default here — see D1 in the budget
    # brief): the loop while-condition uses step < max_step. "time": the loop
    # runs until max_duration_sec elapses (wall clock), regardless of step
    # count. cli.py's cmd_run passes the config-resolved values explicitly;
    # constructing CollectionState directly (as existing tests do) keeps the
    # step-based default.
    budget_mode: str = "steps"
    max_duration_sec: float = 0.0
    total_actions: int = 0
    timeout_count: int = 0
    no_change_retries: int = 0
    external_app_count: int = 0
    empty_ui_retries: int = 0
    reinit_timeout_count: int = 0
    reinit_external_count: int = 0
    # `step` at the time of the most recent reinit of each kind, so a later
    # reinit can tell whether the session has made sustained progress since
    # (see REINIT_FORGIVE_STEPS) rather than just tallying a lifetime count.
    last_timeout_reinit_step: int = -REINIT_FORGIVE_STEPS
    last_external_reinit_step: int = -REINIT_FORGIVE_STEPS
    # True once an open_app has been logged for the current external excursion;
    # reset to False on the next valid in-app frame so each excursion records
    # exactly one open_app (dedup).
    open_app_logged: bool = False
    last_action: Action | None = None
    last_ui_tree: UITree | None = None
    last_raw_xml: str | None = None
    is_first_screen: bool = False
    current_page_id: int | None = None
    root_page_id: int | None = None
    # Pages from which a deliberate Back press was observed to exit the app to
    # the launcher this session. Back is suppressed on these pages (relaunch /
    # tap instead) so a known back-exit page cannot oscillate out-and-back.
    # Learned from press_back drift only (see D4) — tap-driven launcher trips
    # and keyboard-dismiss backs are never recorded.
    back_exit_page_ids: set[int] = field(default_factory=set)
    same_page_count: int = 0
    page_graph: PageGraph = field(default_factory=PageGraph)
    # Set per screen when a ScreenMatcher is active; carries the element-set
    # match so the loop can persist it (save_observation, gated on
    # is_new_observation) and its page_key (save_page_knowledge, on a new page).
    current_screen_match: ScreenMatch | None = None
    # Iterations spent on non-action waits (timeouts, no-change, stale XML,
    # keyboard/permission/system handling) since the last real action. `step`
    # no longer advances on these paths, so this is an independent absolute cap
    # that breaks pathological loops where the device emits signals forever
    # without ever producing an actionable frame.
    idle_iterations: int = 0


def _is_root_screen(state: CollectionState) -> bool:
    """True when the current page is the session's root (first in-app) page.

    Back from the root only exits the app to the launcher, so every deliberate
    back is treated like the first screen there (tap instead of back).
    """
    return (
        state.current_page_id is not None
        and state.current_page_id == state.root_page_id
    )


def _back_would_exit(state: CollectionState) -> bool:
    """True when a deliberate Back on the current page would exit the app.

    Union of the session root (back from it only reaches the launcher) and the
    pages learned this session to back-exit to the launcher
    (``back_exit_page_ids``). On these pages the loop relaunches / taps instead
    of pressing Back, and ``select_action`` is told ``is_root_screen=True``.
    """
    return _is_root_screen(state) or (
        state.current_page_id is not None
        and state.current_page_id in state.back_exit_page_ids
    )


def _mark_if_back_exited(state: CollectionState, exited: bool) -> None:
    """Record the current page as back-exiting when a Back press left the app.

    ``exited`` is ``safe_press_back``'s return (True = the Back drifted out of
    the app and recovery ran). Marking the page it backed FROM suppresses Back
    there next time. No-ops on the keyboard-dismiss path, which never calls
    this (D4: a keyboard Back is not a page back-exit).
    """
    if exited and state.current_page_id is not None:
        state.back_exit_page_ids.add(state.current_page_id)


def _has_budget(
    state: CollectionState, clock: Callable[[], float], deadline: float | None
) -> bool:
    """True while the session's end condition (time or steps) is not yet met."""
    if deadline is not None:
        return clock() < deadline
    return state.step < state.max_step


def run_collection_loop(
    collector: Collector,
    state: CollectionState,
    package: str,
    *,
    now=None,
) -> None:
    """Run the main collection while-loop, mutating state until session ends."""
    # `now` is a sentinel (not `time.monotonic` as a default arg) so tests can
    # inject a fake clock at call time; a real default arg would bind
    # `time.monotonic` at function-definition time and ignore monkeypatching.
    clock = now if now is not None else time.monotonic

    # Discard signals left over from the previous session. The sequential
    # run_queue reuses one server/client connection, so a trailing "finish"
    # from the prior session's teardown can still sit in the queue when this
    # freshly-launched session starts. Without this clear, the first
    # get_latest_signal would return that stale finish and end the new session
    # at step 0 — and it cascades to every later app in the queue.
    collector.server.clear_signal_queue()

    # Time-budgeted sessions (budget_mode="time") end when the wall clock
    # crosses this deadline, recomputed fresh on every run_collection_loop
    # entry (resume = per-run wall-clock budget, not cumulative). Step-budgeted
    # sessions (default) leave deadline None and fall back to step < max_step.
    deadline = (
        clock() + state.max_duration_sec if state.budget_mode == "time" else None
    )

    # `step` only advances on a real action now, so a session that can never act
    # (e.g. a system screen that emits signals but no actionable frame) would
    # never reach max_step. idle_iterations is the absolute backstop.
    max_idle = max(state.max_step * 4, 20)

    while _has_budget(state, clock, deadline) and state.idle_iterations < max_idle:
        state.idle_iterations += 1
        try:
            result = collector.server.get_latest_signal(timeout=collector.xml_timeout)

            if result is None:
                state.timeout_count += 1
                logger.warning(
                    f"Step {state.step}: signal timeout "
                    f"({state.timeout_count}/{MAX_SIGNAL_TIMEOUTS})"
                )
                if state.timeout_count >= MAX_SIGNAL_TIMEOUTS:
                    if state.step - state.last_timeout_reinit_step >= REINIT_FORGIVE_STEPS:
                        state.reinit_timeout_count = 0
                    state.reinit_timeout_count += 1
                    state.last_timeout_reinit_step = state.step
                    if state.reinit_timeout_count > MAX_TIMEOUT_REINITS:
                        logger.error(
                            f"Timeout reinit exhausted "
                            f"({MAX_TIMEOUT_REINITS}/{MAX_TIMEOUT_REINITS}), "
                            f"ending session"
                        )
                        break
                    logger.warning(
                        f"Step {state.step}: timeout reinit "
                        f"({state.reinit_timeout_count}/{MAX_TIMEOUT_REINITS}), "
                        f"force-relaunching {package}"
                    )
                    try:
                        collector.adb.force_stop(package)
                        collector.adb.launch_app(package)
                    except Exception as _e:
                        logger.error(f"Timeout reinit launch failed: {_e}")
                    collector.server.clear_signal_queue()
                    state.timeout_count = 0
                    state.last_ui_tree = None
                    state.last_action = None
                    time.sleep(3.0)
                    continue
                # A timeout means no screenshot/XML arrived. A runtime
                # permission dialog (permissioncontroller) emits no a11y events,
                # so it only shows up here — grant it ("While using the app")
                # via adb so exploration continues instead of relaunching past
                # an ungranted permission.
                if _try_grant_permission_via_adb(collector, state):
                    state.timeout_count = 0
                    continue
                # If we drifted out of the target app (e.g. a system role screen
                # that emits no accessibility events and can't be closed via
                # force_stop), relaunch to escape it. Otherwise nudge a static
                # in-app screen with a tap.
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
                if _handle_external_app(collector, state, package, result[2]):
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
            with contextlib.suppress(Exception):
                collector.explorer.recover(package)


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
        if state.is_first_screen or _back_would_exit(state):
            logger.warning(
                f"Step {state.step}: {MAX_NO_CHANGE_RETRIES} "
                f"no-change retries, on first/root screen — relaunching instead of back"
            )
            relaunch_app_fallback(collector.adb, package)
        else:
            logger.warning(
                f"Step {state.step}: {MAX_NO_CHANGE_RETRIES} "
                f"no-change retries, pressing back"
            )
            _mark_if_back_exited(state, safe_press_back(
                collector.adb, collector.explorer, package
            ))
        collector.server.clear_signal_queue()
        state.no_change_retries = 0
        collector.explorer.clear_excluded()
        state.last_action = None
        state.last_ui_tree = None
        time.sleep(collector.action_delay)
        return False

    if state.last_ui_tree is not None and len(state.last_ui_tree) > 0:
        if state.last_raw_xml:
            collector.explorer.set_screen_context(state.last_raw_xml, package=package)
        action = collector.explorer.select_action(
            state.last_ui_tree, state.step, is_first_screen=state.is_first_screen,
            page_id=state.current_page_id, is_root_screen=_back_would_exit(state),
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
        if state.is_first_screen or _back_would_exit(state):
            logger.info(
                f"Step {state.step}: no UI tree, on first/root screen — relaunching instead of back"
            )
            relaunch_app_fallback(collector.adb, package)
        else:
            _mark_if_back_exited(state, safe_press_back(
                collector.adb, collector.explorer, package
            ))
        state.no_change_retries = 0
        collector.explorer.clear_excluded()
        state.last_action = None
        time.sleep(collector.action_delay)
    return False


def _handle_external_app(
    collector: Collector,
    state: CollectionState,
    package: str,
    payload: dict | None = None,
) -> bool:
    """Handle an external_app signal. Returns True if session should end.

    Whenever recovery actually relaunches the target app (an open_app), record
    it once per excursion for open_app learning. The launch is *not* a
    navigation transition: ``state.last_action`` is cleared so the next xml
    frame draws no live page-graph edge, the explorer cleared its routing
    record inside ``return_to_app``/``recover``, and the logged event carries
    ``transition: false`` so the offline rebuild and converter skip it.
    """
    # Back-exit learning (D4): if the drift is specifically to the launcher and
    # the last action was a deliberate Back, the current page back-exits the
    # app — remember it so Back is suppressed there next time. tap-driven trips
    # and non-launcher drifts (gms/store) are never marked.
    detected = (payload or {}).get("detected_package", "")
    if (
        is_launcher(detected)
        and state.last_action is not None
        and state.last_action.action_type == "press_back"
        and state.current_page_id is not None
    ):
        state.back_exit_page_ids.add(state.current_page_id)
        logger.warning(
            f"Step {state.step}: back from page {state.current_page_id} exited "
            f"to launcher {detected} — suppressing back on this page"
        )

    state.external_app_count += 1
    logger.warning(
        f"Step {state.step}: external app detected "
        f"({state.external_app_count}/{MAX_EXTERNAL_APP_RETRIES})"
    )
    if state.external_app_count >= MAX_EXTERNAL_APP_RETRIES:
        if state.step - state.last_external_reinit_step >= REINIT_FORGIVE_STEPS:
            state.reinit_external_count = 0
        state.reinit_external_count += 1
        state.last_external_reinit_step = state.step
        if state.reinit_external_count > MAX_EXTERNAL_REINITS:
            logger.error(
                f"External app reinit exhausted "
                f"({MAX_EXTERNAL_REINITS}/{MAX_EXTERNAL_REINITS}), "
                f"ending session"
            )
            return True
        logger.warning(
            f"Step {state.step}: external app reinit "
            f"({state.reinit_external_count}/{MAX_EXTERNAL_REINITS}), "
            f"force-relaunching {package}"
        )
        try:
            launched = collector.explorer.recover(package)
        except Exception as e:
            logger.error(f"External reinit failed: {e}")
            launched = False
        _record_open_app(collector, state, package, payload, launched)
        collector.server.clear_signal_queue()
        state.external_app_count = 0
        state.last_action = None
        state.last_ui_tree = None
        time.sleep(collector.action_delay)
        return False
    launched = False
    try:
        if state.external_app_count <= 3:
            launched = collector.explorer.return_to_app(package)
        else:
            launched = collector.explorer.recover(package)
    except Exception as e:
        logger.error(f"Recovery attempt failed: {e}")
    _record_open_app(collector, state, package, payload, launched)
    collector.server.clear_signal_queue()
    state.last_action = None
    state.last_ui_tree = None
    time.sleep(collector.action_delay)
    return False


def _record_open_app(
    collector: Collector,
    state: CollectionState,
    package: str,
    payload: dict | None,
    launched: bool,
) -> None:
    """Log an open_app once per excursion when recovery relaunched the app."""
    if not launched or state.open_app_logged:
        return
    app_name = collector._app_names.get(package, "")
    from_package = payload.get("detected_package") if payload else None
    collector.writer.log_open_app(
        package, app_name, step=state.step, from_package=from_package
    )
    state.open_app_logged = True
    logger.info(f"Step {state.step}: recorded open_app for {package}")


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


# Positive grant buttons in priority order. "While using the app" wins so the
# grant persists for the app's foreground lifetime (policy: always grant runtime
# permission prompts with "While using the app", never "Only this time"/deny).
_GRANT_KEYWORDS: tuple[str, ...] = (
    "while using", "앱 사용 중에만", "사용 중에만",
    "allow", "허용", "ok", "확인", "yes",
)
# Never tap a button whose label contains one of these, even if a grant keyword
# is also a substring (e.g. "Don't allow" contains "allow"; "Only this time" is
# a one-shot grant we don't want for collection).
_DENY_TOKENS: tuple[str, ...] = (
    "don't", "don’t", "deny", "거부", "취소", "cancel", "only this time", "이번만",
)


def _try_grant_permission_via_adb(
    collector: Collector, state: CollectionState,
) -> bool:
    """Grant a permission dialog the push loop can't see, via adb.

    permissioncontroller's ``GrantPermissionsActivity`` emits no accessibility
    events, so when it pops up the server only observes a *signal timeout* — the
    event-driven ``_handle_permission_dialog`` never runs and the dialog blocks
    exploration. Here, on a timeout, we poll the foreground via adb; if it is a
    permission dialog we dump the UI (adb-side ``uiautomator``, not the
    accessibility tree) and tap a grant button, always preferring **"While using
    the app"**. Returns True if a button was tapped.
    """
    try:
        top = collector.adb.get_current_package()
    except Exception:
        return False
    if not is_permission_dialog(top):
        return False

    try:
        collector.adb.shell("uiautomator dump /sdcard/_mc_perm.xml")
        raw = collector.adb.shell("cat /sdcard/_mc_perm.xml")
    except Exception:
        return False
    start = raw.find("<?xml")
    if start == -1:
        start = raw.find("<hierarchy")
    if start == -1:
        return False
    try:
        clickable = UITree.from_xml_string(raw[start:]).get_clickable_elements()
    except Exception:
        return False

    # Scan clickable buttons only — the dialog title/message also contains
    # "Allow <app> to …" but is not clickable, so scanning all nodes would tap
    # the title. Priority keeps "While using the app" ahead of a plain "Allow";
    # the deny guard prevents tapping "Don't allow"/"Only this time".
    target = None
    for kw in _GRANT_KEYWORDS:
        for el in clickable:
            label = f"{el.text} {el.content_desc}".lower()
            if kw in label and not any(tok in label for tok in _DENY_TOKENS):
                target = el
                break
        if target is not None:
            break
    if target is None:
        return False

    cx, cy = target.center
    collector.adb.tap(cx, cy)
    logger.info(
        f"Step {state.step}: permission dialog on {top} (no a11y event) — "
        f"tapped '{target.display_name}' via adb"
    )
    collector.server.clear_signal_queue()
    state.last_action = None
    state.last_ui_tree = None
    time.sleep(collector.action_delay)
    return True


def _resolve_coverage_activity(collector: Collector, activity_name: str) -> str:
    """Resolve the activity name to record for coverage.

    The AccessibilityService reports ``activity_name`` as a generic View class
    (e.g. ``.../android.view.ViewGroup``) on many frames, which never matches a
    declared activity and freezes coverage. Discriminator: if the a11y value is
    already a declared activity, trust it (no adb call — avoids adding
    topResumedActivity race noise on frames a11y already got right); otherwise
    resolve the real foreground activity via adb.

    This value is used ONLY for the coverage tracker. The a11y ``activity_name``
    still keys the matcher / page_graph / save_observation / event, so
    structural dedup is untouched.
    """
    tracker = collector._activity_tracker
    if tracker is not None and tracker.is_declared(activity_name):
        return activity_name
    return collector.adb.get_current_activity()


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
        return True

    # Permission / install grant dialog: act on it (grant > dismiss) instead of
    # burning steps skipping it as stale XML — otherwise we loop here forever.
    if is_permission_dialog(top_package):
        _handle_permission_dialog(collector, state, xml_str)
        return True

    # Drifted into another system screen we cannot drive: relaunch the target.
    if top_package and is_system_screen(top_package) and top_package != package:
        logger.warning(
            f"Step {state.step}: in system screen {top_package}, "
            f"returning to {package}"
        )
        collector.explorer.return_to_app(package)
        collector.server.clear_signal_queue()
        return True

    if top_package and top_package != package:
        logger.info(
            f"Step {state.step}: stale XML from {top_package} "
            f"(expected {package}), skipping"
        )
        # Drop the foreign signal so a stuck background app can't spin this path.
        collector.server.clear_signal_queue()
        return True

    if not activity_name:
        activity_name = collector.adb.get_current_activity()

    if collector._activity_tracker is not None:
        # Coverage records the REAL foreground activity: a11y often reports a
        # generic View class (ViewGroup) that never matches a declared activity.
        # Matcher/page_graph/save/event keep the a11y `activity_name` above.
        coverage_activity = _resolve_coverage_activity(collector, activity_name)
        # Key coverage rows by the frame_index this screen will be allocated
        # below (next_frame_index() reads/advances the same step_count value),
        # so the offline page-graph rebuild's CSV fallback joins on the same
        # index events use. `step` is a loop counter and no longer matches
        # frame indices.
        entry = collector._activity_tracker.record(
            coverage_activity, collector.writer.step_count
        )
        logger.debug(
            f"Activity coverage: {entry['coverage']:.2%} "
            f"({entry['unique_visited']}/{entry['total_activities']})"
        )

    previous_page_id = state.current_page_id
    pages_before = len(state.page_graph.nodes)

    # Cost attribution covers every LLM consumer this step (element extraction
    # in the matcher + input-text generation); set the step before the matcher
    # makes any call.
    if collector._llm_client is not None:
        collector._llm_client.set_step(state.step)

    # Page identity. With a ScreenMatcher, element-set matching decides the page
    # (and feeds the explorer's same-function compression); without one, fall
    # back to the structural-fingerprint identity (byte-for-byte legacy path).
    # Alongside page identity, resolve OBSERVATION identity: which of the
    # page's stored visual states (if any) this screen matches — is_new_observation
    # gates whether the save block below writes new observation files at all.
    state.current_screen_match = None
    page_key: str | None = None
    observation_num = 0
    is_new_observation = False
    if collector._screen_matcher is not None:
        encoded_xml, _ = encode_with_bounds(xml_str)
        # Pass the in-memory screenshot bytes so the matcher's Stage-0 luminance
        # prefilter can dedup a near-pixel-identical screen with no LLM call. The
        # bytes are still held here (saved/cleared only further below); None when
        # no screenshot arrived → the prefilter degrades gracefully.
        match = collector._screen_matcher.match(
            xml_str, encoded_xml, activity_name, screenshot=collector._latest_screenshot
        )
        # A pending match is a loading/splash (or empty-extract) frame the matcher
        # declined to register: keep current_screen_match=None (page_key stays
        # None too) so the save block below persists nothing and no page node is
        # created, leaving current_page_id at its prior value. The empty-UI guard
        # below then waits / relaunches.
        if not match.pending:
            state.current_screen_match = match
            state.current_page_id = state.page_graph.get_or_create_page_by_match(
                match, activity_name, xml_str, state.step,
            )
            state.page_graph.record_observation(
                state.current_page_id, match.is_new_observation,
            )
            collector.explorer.set_match_context(match.page_key, match.families)
            page_key = match.page_key
            observation_num = match.observation_num
            is_new_observation = match.is_new_observation
    else:
        state.current_page_id = state.page_graph.get_or_create_page(
            activity_name, xml_str, state.step,
        )
        # No anchor/pixel comparator on this path — every frame is a new
        # observation (a location/shape unification with the matcher path's
        # pages/{page_key}/{obs}/ layout, not a new dedup behavior).
        observation_num = state.page_graph.next_observation_num(state.current_page_id)
        is_new_observation = True
        page_key = str(state.current_page_id)
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
            if state.is_first_screen or _back_would_exit(state):
                relaunch_app_fallback(collector.adb, package)
            else:
                _mark_if_back_exited(state, safe_press_back(
                    collector.adb, collector.explorer, package
                ))
            collector.server.clear_signal_queue()
            collector.explorer.clear_excluded()
            state.same_page_count = 0
            state.last_action = None
            state.last_ui_tree = None
            time.sleep(collector.action_delay)
            return True

    # Allocate this frame's join key for events.jsonl BEFORE the save block —
    # unconditionally, pending or not, so activity_coverage.csv's step_count
    # keying (read above, before match()) stays exactly aligned to "frames
    # processed so far" as it was before this refactor.
    frame_index = collector.writer.next_frame_index()

    # Persist a new observation's files ONLY when this screen isn't a reuse of
    # one already on disk (page_key is None for a pending/declined frame — see
    # above). Best-effort: never break collection on a write failure.
    if page_key is not None and is_new_observation:
        try:
            collector.writer.save_observation(
                page_key, observation_num, collector._latest_screenshot, xml_str,
                match=state.current_screen_match, activity=activity_name,
            )
        except Exception as e:
            logger.warning(f"Step {state.step}: save_observation failed ({e})")
        if (
            state.current_screen_match is not None
            and state.current_screen_match.is_new_page
            and collector._screen_matcher is not None
        ):
            try:
                knowledge = collector._screen_matcher.get_page_knowledge(page_key)
                if knowledge is not None:
                    collector.writer.save_page_knowledge(page_key, knowledge)
            except Exception as e:
                logger.warning(f"Step {state.step}: save_page_knowledge failed ({e})")
    collector._latest_screenshot = None

    ui_tree = UITree.from_xml_string(xml_str)
    # A tree with nodes but NO interactable element (e.g. a React-Native screen
    # whose accessibility subtree is a single non-actionable container, or a
    # transient blank-after-BACK frame) is just as useless as an empty one: the
    # explorer would only blind-tap a random coordinate that fires no event and
    # cascades into signal timeouts. Treat "not actionable" the same as empty so
    # we wait for load / relaunch instead.
    if len(ui_tree) == 0 or not ui_tree.get_interactable_elements():
        state.empty_ui_retries += 1
        if state.empty_ui_retries <= MAX_EMPTY_UI_RETRIES:
            logger.info(
                f"Step {state.step}: no UI elements, waiting for app load "
                f"({state.empty_ui_retries}/{MAX_EMPTY_UI_RETRIES})"
            )
            time.sleep(1.0)
            state.last_ui_tree = None
            state.last_action = None
            return True
        state.empty_ui_retries = 0
        if state.is_first_screen or _back_would_exit(state):
            logger.warning(
                f"Step {state.step}: no UI elements, on first/root screen — relaunching instead of back"
            )
            relaunch_app_fallback(collector.adb, package)
        else:
            logger.warning(
                f"Step {state.step}: no UI elements, pressing back"
            )
            _mark_if_back_exited(state, safe_press_back(
                collector.adb, collector.explorer, package
            ))
        state.last_ui_tree = None
        state.last_action = None
        return True

    state.empty_ui_retries = 0
    # Back on a real in-app interactive frame: the external excursion (if any)
    # is over, so the next drift starts a fresh excursion that records its own
    # open_app. (The external-app *counter* below is deliberately stickier — it
    # only resets on a brand-new page to break external↔return loops on a known
    # page — but open_app dedup must clear here or a return to a known page
    # would suppress the next excursion's open_app.)
    state.open_app_logged = False
    # Only reset the external-app counter on genuine progress (a brand-new page).
    # Resetting on every in-app frame let an external↔return loop run forever
    # because each return landed on an already-known page.
    if discovered_new_page:
        state.external_app_count = 0

    collector.explorer.set_screen_context(xml_str, activity_name, package)
    action = collector.explorer.select_action(
        ui_tree, state.step, is_first_screen=state.is_first_screen,
        page_id=state.current_page_id, is_root_screen=_back_would_exit(state),
    )
    logger.info(
        f"Step {state.step}: {action.action_type} "
        f"(element_index={action.element_index})"
    )

    collector.explorer.execute_action(action)
    state.total_actions += 1
    # A real action ran: this is the only path that advances `step` and the only
    # one that clears the idle backstop.
    state.idle_iterations = 0

    collector.server.clear_signal_queue()

    state.last_action = action
    state.last_ui_tree = ui_tree
    state.last_raw_xml = xml_str

    event = action.to_dict()
    event["step"] = state.step
    event["activity_name"] = activity_name
    # frame_index (allocated above, before the save block) orders events;
    # page_key/observation_num are the join key to the actual screen files —
    # data/{package}/pages/{page_key}/{observation_num}/ — on EVERY event,
    # new observation or reused. `step` is a loop-counter label only.
    event["frame_index"] = frame_index
    event["page_key"] = page_key
    event["observation_num"] = observation_num
    collector.writer.log_event(event)

    time.sleep(collector.action_delay)
    state.step += 1
    return False
