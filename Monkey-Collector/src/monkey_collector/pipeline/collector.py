"""Collector facade: wire TCP server, explorer, storage, and drive sessions.

Flow (server-driven):
  1. Start TCP server and wait for client (Android CollectorService) to connect.
  2. Send ``{"type": "START", "package": pkg}`` to the client.
  3. Client replies with a ``P`` message carrying the same package name.
  4. Server launches the app via ADB; the AccessibilityService picks it up and
     begins streaming ``S`` (screenshot) + ``X`` (XML) signals.
  5. Loop: parse XML → select action → execute via ADB → save payload.
  6. Finalize session; optionally loop to the next package with
     ``run_queue([...])``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from monkey_collector.adb import AdbClient
from monkey_collector.domain.activity_coverage import ActivityCoverageTracker
from monkey_collector.domain.cost_tracker import CostTracker
from monkey_collector.pipeline.collection_loop import CollectionState, run_collection_loop
from monkey_collector.pipeline.exploration import Explorer
from monkey_collector.pipeline.session_manager import (
    finalize_session,
    init_or_resume_session,
    receive_target_package,
    wait_for_connection,
)
from monkey_collector.pipeline.text_generator import TextGenerator
from monkey_collector.storage import DataWriter
from monkey_collector.tcp_server import CollectionServer

if TYPE_CHECKING:
    from monkey_collector.llm.client import LLMClient
    from monkey_collector.llm.screen_grouper import ScreenGrouper


class Collector:
    """Orchestrates server-driven data collection on a single device."""

    def __init__(
        self,
        adb: AdbClient,
        explorer: Explorer,
        server: CollectionServer,
        writer: DataWriter,
        max_steps: int = 100,
        action_delay: float = 1.0,
        xml_timeout: float = 25.0,
        activity_coverage_tracker: ActivityCoverageTracker | None = None,
        cost_tracker: CostTracker | None = None,
        text_generator: TextGenerator | None = None,
        llm_client: LLMClient | None = None,
        screen_grouper: ScreenGrouper | None = None,
        new_session: bool = False,
    ):
        self.adb = adb
        self.explorer = explorer
        self.server = server
        self.writer = writer
        self.max_steps = max_steps
        self.action_delay = action_delay
        self.xml_timeout = xml_timeout
        self._latest_screenshot: bytes | None = None
        self._activity_tracker = activity_coverage_tracker
        self._cost_tracker = cost_tracker
        self._text_generator = text_generator
        self._llm_client = llm_client
        self._screen_grouper = screen_grouper
        self._new_session = new_session

    def run(self, package: str) -> str:
        """Run a single server-driven collection session.

        Starts and stops the TCP server internally; suitable for one-off
        single-app runs.
        """
        self.server.on_screenshot = self._on_screenshot
        self.server.start()
        try:
            session_id = self._run_session(package)
        finally:
            self.server.stop()
        return session_id

    def run_queue(self, packages: list[str]) -> list[str]:
        """Sequentially collect every package in *packages* on one device.

        The TCP server stays up for the whole run; between packages the
        client is expected to reconnect (triggered by our ``SESSION_END``
        handshake inside ``finalize_session``).
        """
        session_ids: list[str] = []
        self.server.on_screenshot = self._on_screenshot
        self.server.start()

        try:
            for idx, pkg in enumerate(packages, start=1):
                logger.info(
                    f"=== Session {idx}/{len(packages)} — package={pkg} ==="
                )
                if idx > 1:
                    self.server.reset_for_new_session()
                self.explorer.clear_excluded()
                self._latest_screenshot = None

                try:
                    session_id = self._run_session(pkg)
                except KeyboardInterrupt:
                    logger.info("Interrupted during session")
                    break

                if session_id:
                    session_ids.append(session_id)
                    logger.info(
                        f"Session {idx} complete: {session_id} "
                        f"({len(session_ids)}/{len(packages)} done)"
                    )
                else:
                    logger.warning(f"Session {idx} ended without result for {pkg}")
        except KeyboardInterrupt:
            logger.info("Shutting down server...")
        finally:
            self.server.stop()

        return session_ids

    def _run_session(self, package: str) -> str:
        """Run a single collection session for *package* (server must be started).

        The server drives the handshake: it sends ``START`` once the client is
        connected, waits for the client's ``P`` reply, then launches the app
        via ADB before handing off to ``run_collection_loop``.
        """
        if not wait_for_connection(self):
            return ""

        if not self.server.send_start(package):
            logger.error(f"Failed to send START command for {package}")
            return ""

        pkg = receive_target_package(self, package)
        if pkg is None:
            logger.error(f"Client did not acknowledge START for {package}")
            return ""
        if pkg != package:
            logger.warning(
                f"Client reported package={pkg} but server requested {package}; "
                f"using server value"
            )
        package = package  # trust the server-driven value
        logger.info(f"Target package: {package}")

        try:
            self.adb.force_stop(package)
        except Exception as e:
            logger.debug(f"force_stop({package}) ignored: {e}")
        self.adb.launch_app(package)

        session_id, resume_step = init_or_resume_session(self, package)
        logger.info(f"Starting session: {session_id}")
        logger.info(f"Target app: {package}, max_steps: {self.max_steps}")

        state = CollectionState(
            step=resume_step,
            max_step=resume_step + self.max_steps,
        )

        # Each app session explores in isolation — drop the previous session's
        # transition graph / coverage so cross-app memory cannot leak.
        self.explorer.reset()

        try:
            run_collection_loop(self, state, package)
        finally:
            finalize_session(self, session_id)

        logger.info(
            f"Session complete: {session_id} | "
            f"steps={self.writer.step_count}, actions={state.total_actions}"
        )
        return session_id

    def _on_screenshot(self, image_data: bytes):
        """Callback: store latest screenshot for saving with next XML."""
        self._latest_screenshot = image_data
