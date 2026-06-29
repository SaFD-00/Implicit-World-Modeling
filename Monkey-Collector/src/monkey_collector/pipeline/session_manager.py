"""Session lifecycle: init/resume at start, finalize at end."""

from __future__ import annotations

import os
import shutil
from typing import TYPE_CHECKING

from loguru import logger

from monkey_collector.catalog_activities import ActivityCatalog
from monkey_collector.domain.page_graph import build_graph_from_session

if TYPE_CHECKING:
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
) -> tuple[str, int]:
    """Initialize a new session or resume an existing one for the package.

    Returns (session_id, resume_step).
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
                collector.writer.session_dir, total_activities, package,
                allow_dynamic_total=allow_dynamic,
            )
        if collector._cost_tracker is not None:
            collector._cost_tracker.resume(collector.writer.session_dir)
        logger.info(f"Resuming session: {session_id} from step {resume_step}")
        return session_id, resume_step

    session_id = package
    if collector._new_session:
        existing_dir = os.path.join(collector.writer.base_dir, session_id)
        if os.path.isdir(existing_dir):
            shutil.rmtree(existing_dir)
            logger.info(f"Removed existing session directory: {existing_dir}")
    collector.writer.init_session(session_id, package)
    if collector._activity_tracker is not None:
        total_activities, allow_dynamic = _resolve_declared_activities(
            collector, package,
        )
        collector._activity_tracker.initialize(
            collector.writer.session_dir, total_activities, package,
            allow_dynamic_total=allow_dynamic,
        )
    if collector._cost_tracker is not None:
        collector._cost_tracker.initialize(collector.writer.session_dir)
    return session_id, 0


def finalize_session(collector: Collector, session_id: str) -> None:
    """Finalize: notify app, save session, persist page graph, visualize.

    With a live ScreenMatcher the element-set page graph (carrying ``page_key`` /
    ``element_names``) built during the loop is persisted as-is. Without one
    (degrade path) the graph is rebuilt post-hoc from the session via the
    structural fingerprint — the same path the offline ``page-map`` command uses.
    """
    collector.server.send_session_end()
    collector.writer.finalize_session()

    live_graph = getattr(collector, "_live_page_graph", None)
    if collector._screen_matcher is not None and live_graph is not None and live_graph.nodes:
        graph = live_graph
    else:
        graph = build_graph_from_session(collector.writer.session_dir)

    if graph.nodes:
        graph_data = graph.to_dict()
        graph_data["metadata"]["session_id"] = session_id
        collector.writer.save_page_graph(graph_data)
        try:
            from monkey_collector.export.graph_visualizer import visualize_session
            visualize_session(
                collector.writer.session_dir, open_browser=False,
            )
        except Exception as e:
            logger.warning(f"Page map visualization failed: {e}")
