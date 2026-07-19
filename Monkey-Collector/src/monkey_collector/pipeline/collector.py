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
    rehydrate_session,
    wait_for_connection,
)
from monkey_collector.pipeline.text_generator import TextGenerator
from monkey_collector.storage import DataWriter
from monkey_collector.tcp_server import CollectionServer

if TYPE_CHECKING:
    from monkey_collector.domain.page_graph import PageGraph
    from monkey_collector.llm.client import LLMClient
    from monkey_collector.pipeline.screen_matching.screen_matcher import ScreenMatcher


class Collector:
    """Orchestrates server-driven data collection on a single device."""

    def __init__(
        self,
        adb: AdbClient,
        explorer: Explorer,
        server: CollectionServer,
        writer: DataWriter,
        max_steps: int = 1500,
        action_delay: float = 1.0,
        xml_timeout: float = 12.0,
        poke_delay: float = 1.5,
        budget_mode: str = "steps",
        max_duration_sec: int = 0,
        max_action_repeats: int = 8,
        max_steps_without_new_page: int = 98,
        activity_coverage_tracker: ActivityCoverageTracker | None = None,
        cost_tracker: CostTracker | None = None,
        text_generator: TextGenerator | None = None,
        llm_client: LLMClient | None = None,
        screen_matcher: ScreenMatcher | None = None,
        new_session: bool = False,
        app_contexts: dict[str, str] | None = None,
        app_names: dict[str, str] | None = None,
    ):
        self.adb = adb
        self.explorer = explorer
        self.server = server
        self.writer = writer
        self.max_steps = max_steps
        self.action_delay = action_delay
        self.xml_timeout = xml_timeout
        # Silence inside one signal wait before the server pokes the client with
        # CAPTURE; carved out of xml_timeout, so the total wait is unchanged.
        # 0/negative or >= xml_timeout disables poking.
        self.poke_delay = poke_delay
        self.budget_mode = budget_mode
        self.max_duration_sec = max_duration_sec
        # D2/D3 loop guards (repeat-action circuit breaker + plateau early-stop).
        # Defaulted so existing tests that build a Collector without them still
        # get the product guard values; 0 or negative disables each guard.
        self.max_action_repeats = max_action_repeats
        self.max_steps_without_new_page = max_steps_without_new_page
        self._latest_screenshot: bytes | None = None
        self._activity_tracker = activity_coverage_tracker
        self._cost_tracker = cost_tracker
        self._text_generator = text_generator
        self._llm_client = llm_client
        self._screen_matcher = screen_matcher
        self._new_session = new_session
        self._app_contexts = app_contexts or {}
        # package -> human-readable app name (apps.csv), used to label open_app
        # events on external recovery. Falls back to "" when unresolved.
        self._app_names = app_names or {}
        # The live element-set page graph of the current session (set per
        # session), persisted by finalize when a ScreenMatcher is active.
        self._live_page_graph: PageGraph | None = None

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

        # Ground LLM input-text generation in the current app's domain. Set every
        # session (package id as fallback) so the shared generator never leaks the
        # previous app's description into this one.
        if self._text_generator is not None:
            self._text_generator.set_app_context(
                self._app_contexts.get(package, package)
            )

        try:
            self.adb.force_stop(package)
        except Exception as e:
            logger.debug(f"force_stop({package}) ignored: {e}")
        self.adb.launch_app(package)

        session_id, resume_step, is_resumed = init_or_resume_session(self, package)
        logger.info(f"Starting session: {session_id}")
        logger.info(f"Target app: {package}, max_steps: {self.max_steps}")

        state = CollectionState(
            step=resume_step,
            max_step=resume_step + self.max_steps,
            budget_mode=self.budget_mode,
            max_duration_sec=self.max_duration_sec,
        )

        # Each app session explores in isolation — drop the previous session's
        # transition graph / coverage and page knowledge so cross-app memory
        # cannot leak.
        self.explorer.reset()
        if self._screen_matcher is not None:
            self._screen_matcher.reset()

        # Resume rehydrates state.page_graph + the ScreenMatcher's knowledge
        # from the durable data/{package}/pages/ tree — a fresh session's tree
        # is empty, so this only matters (and is only called) on resume.
        if is_resumed:
            rehydrate_session(self, state)

        # Expose the live (element-set) page graph to finalize. It mutates in
        # place during the loop, so this reference captures the final graph —
        # AFTER rehydration, since rehydrate_session may have replaced it.
        self._live_page_graph = state.page_graph

        try:
            run_collection_loop(self, state, package)
        finally:
            finalize_session(self, session_id)
            # Structural effect-log diagnostic — always on, no knob. Emitted
            # before the next session's reset() discards this Memory. The
            # getattr/hasattr guard is for explorer doubles that carry no
            # Memory, NOT a feature switch.
            memory = getattr(self.explorer, "_memory", None)
            if memory is not None and hasattr(memory, "log_effect_summary"):
                memory.log_effect_summary()

        logger.info(
            f"Session complete: {session_id} | "
            f"steps={self.writer.step_count}, actions={state.total_actions}"
        )
        return session_id

    def _on_screenshot(self, image_data: bytes):
        """Callback: store latest screenshot for saving with next XML."""
        self._latest_screenshot = image_data
