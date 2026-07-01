"""Session lifecycle: init/resume at start, finalize at end."""

from __future__ import annotations

import os
import shutil
from typing import TYPE_CHECKING

from loguru import logger

from monkey_collector.catalog_activities import ActivityCatalog
from monkey_collector.domain.page_graph import PageGraph, build_graph_from_session
from monkey_collector.pipeline.screen_matching.rehydrate import (
    rehydrate_screen_matcher,
)

if TYPE_CHECKING:
    from monkey_collector.pipeline.collection_loop import CollectionState
    from monkey_collector.pipeline.collector import Collector


def _resolve_declared_activities(
    collector: Collector, package: str,
) -> tuple[list[str], bool]:
    """Resolve the declared-activity ground truth for a package.

    Primary: ``catalog/activities.json`` (static, fixed across sessions and
    devices). Fallback: ``adb dumpsys package`` when the catalog file is
    missing/corrupt or the package is not registered (logs a single warning).

    Returns:
        ``(activities, allow_dynamic_total)`` — catalog hit fixes the
        denominator (``False``); fallback preserves legacy dynamic
        expansion (``True``).
    """
    catalog = ActivityCatalog.instance()
    if catalog.is_loaded():
        from_catalog = catalog.get_declared(package)
        if from_catalog is not None:
            logger.info(
                f"Activity ground truth from catalog: {len(from_catalog)} "
                f"activities for {package}"
            )
            return from_catalog, False
        logger.warning(
            f"Activity catalog miss for {package}; falling back to dumpsys"
        )
    return collector.adb.get_declared_activities(package), True


def wait_for_connection(collector: Collector, timeout_seconds: int = 120) -> bool:
    """Block until a device connects or timeout."""
    import time

    logger.info(
        "Waiting for device to connect (ensure CollectorService is running)..."
    )
    for _ in range(timeout_seconds):
        time.sleep(1)
        if collector.server.is_client_connected():
            logger.info("Device connected")
            return True
    logger.error(f"Device did not connect within {timeout_seconds} seconds")
    return False


def receive_target_package(collector: Collector, package: str | None) -> str | None:
    """Receive or confirm the target package from the client."""
    if package is None:
        logger.info("Waiting for target package from client...")
        pkg = collector.server.wait_for_package(timeout=30.0)
        if pkg is None:
            logger.error("No package received from client")
            return None
        return pkg

    received = collector.server.wait_for_package(timeout=5.0)
    if received is None:
        # No package ACK means the client never ran startCollection for this
        # START — the device crashed/restarted or the START hit a stale socket
        # during the session handoff. Do NOT fall back to the server-requested
        # package: that masks a dead session and burns the whole step budget on
        # signal timeouts. Abort so the caller skips to the next app cleanly.
        logger.error(
            f"Client did not acknowledge START for {package} within 5s "
            f"(device may have crashed or START hit a stale socket); "
            f"aborting session instead of collecting blind"
        )
        return None
    return received


def init_or_resume_session(
    collector: Collector,
    package: str,
) -> tuple[str, int, bool]:
    """Initialize a new session or resume an existing one for the package.

    Returns (session_id, resume_step, is_resumed).
    """
    collector.server.on_external_app = (
        lambda payload: collector.writer.log_external_app(payload)
    )

    existing = (
        None if collector._new_session
        else collector.writer.find_existing_session(package)
    )

    if existing:
        session_id = existing
        resume_step = collector.writer.resume_session(session_id)
        if collector._activity_tracker is not None:
            total_activities, allow_dynamic = _resolve_declared_activities(
                collector, package,
            )
            collector._activity_tracker.resume(
                collector.writer.runtime_session_dir, total_activities, package,
                allow_dynamic_total=allow_dynamic,
            )
        if collector._cost_tracker is not None:
            collector._cost_tracker.resume(collector.writer.runtime_session_dir)
        logger.info(f"Resuming session: {session_id} from step {resume_step}")
        return session_id, resume_step, True

    session_id = package
    if collector._new_session:
        # Two roots now — a "fresh start" must wipe both, or a surviving
        # data/{package}/ half would immediately rehydrate stale page
        # knowledge right back into what's supposed to be a new session.
        existing_data_dir = os.path.join(collector.writer.data_dir, session_id)
        existing_runtime_dir = os.path.join(collector.writer.runtime_dir, session_id)
        for existing_dir in (existing_data_dir, existing_runtime_dir):
            if os.path.isdir(existing_dir):
                shutil.rmtree(existing_dir)
                logger.info(f"Removed existing session directory: {existing_dir}")
    collector.writer.init_session(session_id, package)
    if collector._activity_tracker is not None:
        total_activities, allow_dynamic = _resolve_declared_activities(
            collector, package,
        )
        collector._activity_tracker.initialize(
            collector.writer.runtime_session_dir, total_activities, package,
            allow_dynamic_total=allow_dynamic,
        )
    if collector._cost_tracker is not None:
        collector._cost_tracker.initialize(collector.writer.runtime_session_dir)
    return session_id, 0, False


def rehydrate_session(collector: Collector, state: CollectionState) -> None:
    """Rebuild page/observation knowledge from disk after a resume.

    Two halves, both required for the fix to be complete:

    1. Loads the durable ``page_graph.json`` back into ``state.page_graph``
       (a fresh ``CollectionState`` always starts with an empty one, resumed
       or not). Without this, ``finalize_session``'s "prefer the live graph"
       rule would silently overwrite ``page_graph.json`` with only the pages
       touched during THIS resumed run, discarding history for every page not
       revisited before the session ends again — the same class of bug as the
       ``ScreenMatcher`` reset gap below, one layer up.
    2. When a ``ScreenMatcher`` is active, rehydrates its registry /
       structural exact-match cache from ``data/{package}/pages/``.

    Call only when resuming (a fresh session's ``data/`` tree is empty, so
    calling this unconditionally would just be silent no-op log noise).
    """
    graph_path = os.path.join(collector.writer.data_session_dir, "page_graph.json")
    if os.path.isfile(graph_path):
        try:
            state.page_graph = PageGraph.load(graph_path)
            logger.info(f"page_graph rehydrated: {len(state.page_graph.nodes)} pages")
        except Exception as e:
            logger.warning(f"page_graph rehydration failed, starting empty ({e})")

    if collector._screen_matcher is not None:
        try:
            rehydrate_screen_matcher(collector._screen_matcher, collector.writer)
        except Exception as e:
            logger.warning(f"screen_matcher rehydration failed, starting empty ({e})")


def finalize_session(collector: Collector, session_id: str) -> None:
    """Finalize: notify app, save session, persist page graph, visualize.

    The live page graph (``state.page_graph``, carrying ``page_key`` /
    ``element_names`` when a ScreenMatcher is active) built during the loop is
    persisted as-is whenever it has content. Only a session with no live graph
    at all falls back to rebuilding one post-hoc from the session via the
    structural fingerprint — the same path the offline ``page-map`` command uses.
    """
    collector.server.send_session_end()
    collector.writer.finalize_session()

    # The live graph (state.page_graph) is populated inline during the loop in
    # both matcher and non-matcher modes (get_or_create_page_by_match /
    # get_or_create_page), so there's no remaining reason to gate this on
    # screen_matcher being active — prefer it whenever it has content, falling
    # back to the offline structural rebuild only for a session with no live
    # graph at all (e.g. this call site invoked without a loop having run).
    live_graph = getattr(collector, "_live_page_graph", None)
    if live_graph is not None and live_graph.nodes:
        graph = live_graph
    else:
        graph = build_graph_from_session(collector.writer.data_session_dir)

    if graph.nodes:
        graph_data = graph.to_dict()
        graph_data["metadata"]["session_id"] = session_id
        collector.writer.save_page_graph(graph_data)
        try:
            from monkey_collector.export.graph_visualizer import visualize_session
            visualize_session(
                collector.writer.data_session_dir, open_browser=False,
            )
        except Exception as e:
            logger.warning(f"Page map visualization failed: {e}")
