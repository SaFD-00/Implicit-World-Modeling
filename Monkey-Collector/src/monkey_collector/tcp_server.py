"""TCP server for receiving data from Android AccessibilityService app."""

import json
import socket
import threading
from collections.abc import Callable
from queue import Empty, Queue

from loguru import logger

BUFFER_SIZE = 65536


class CollectionServer:
    """TCP server that receives screenshots and XML from the Android app.

    Protocol (App → Server):
      S + size_line + binary_data   = Screenshot
      X + top_pkg + activity_name + target_pkg + is_first_screen("0"/"1") + size_line + xml_data = XML hierarchy
      E + json_line                 = External app detection
      N                             = No visual change detected
      F                             = Session finish

    Protocol (Server → App):
      action_json + \\r\\n           = Action command
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 12345,
        on_screenshot: Callable | None = None,
        on_xml: Callable | None = None,
        on_external_app: Callable | None = None,
        on_finish: Callable | None = None,
    ):
        self.host = host
        self.port = port
        self.on_screenshot = on_screenshot
        self.on_xml = on_xml
        self.on_external_app = on_external_app
        self.on_finish = on_finish
        self._server_socket: socket.socket | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._client: socket.socket | None = None
        # Screen-signal synchronization (consumed by the collection loop)
        self._xml_event = threading.Event()
        self._latest_xml: str | None = None
        self._latest_xml_meta: dict | None = None
        # Package name from client
        self._package_event = threading.Event()
        self._target_package: str | None = None
        # Signal queue for change detection (XML or no-change)
        self._signal_queue: Queue = Queue()

    def start(self):
        """Start the server in a background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(f"Collection server started on {self.host}:{self.port}")

    def stop(self):
        """Stop the server."""
        self._running = False
        if self._client:
            try:
                self._client.close()
            except OSError:
                pass
        if self._server_socket:
            try:
                self._server_socket.close()
            except OSError:
                pass
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Collection server stopped")

    def reset_for_new_session(self):
        """Reset per-session state (queues, events) for the next collection run.

        Leaves ``self._client`` intact: the previous session's SESSION_END/F
        handshake already closed the old socket, and the Android service
        auto-reconnects with a fresh socket before this is called. Closing
        that fresh socket here would force a second reconnect that the client
        does not perform, causing the next ``wait_for_connection`` to time out.
        """
        self._package_event.clear()
        self._target_package = None
        self._xml_event.clear()
        self._latest_xml = None
        self._latest_xml_meta = None
        self.clear_signal_queue()
        logger.debug("Server state reset for new session")

    def is_client_connected(self) -> bool:
        return self._client is not None

    def send_action(self, action: dict) -> bool:
        """Send an action command to the connected App."""
        if not self._client:
            logger.warning("No client connected, cannot send action")
            return False
        try:
            data = json.dumps(action, ensure_ascii=False) + "\r\n"
            self._client.sendall(data.encode("utf-8"))
            return True
        except (OSError, BrokenPipeError) as e:
            logger.error(f"Failed to send action: {e}")
            return False

    def send_session_end(self) -> bool:
        """Send SESSION_END control signal to the connected app.

        Tells the app to stop the current collection session.
        """
        return self.send_action({"type": "SESSION_END"})

    def send_start(self, package: str) -> bool:
        """Tell the connected client to begin collecting *package*.

        The client is expected to start its collection pipeline and reply with
        a ``P`` message carrying the same package name.
        """
        return self.send_action({"type": "START", "package": package})

    def wait_for_xml(
        self, timeout: float = 25.0
    ) -> tuple[str, dict] | None:
        """Block until the next XML is received from the Android app.

        Returns (xml_string, meta_dict) or None on timeout.
        """
        self._xml_event.clear()
        if self._xml_event.wait(timeout):
            meta = self._latest_xml_meta or {}
            return self._latest_xml, meta
        return None

    def wait_for_change_signal(
        self, timeout: float = 25.0
    ) -> tuple[str, str | None, dict | None] | None:
        """Block until XML or no-change signal is received.

        Returns:
            ("xml", xml_string, meta_dict) on screen change
            ("no_change", None, None) on no visual change
            None on timeout
        """
        try:
            return self._signal_queue.get(timeout=timeout)
        except Empty:
            return None

    def get_latest_signal(
        self, timeout: float = 25.0
    ) -> tuple[str, str | None, dict | None] | None:
        """Collapse consecutive stale ``xml`` signals, but never drop control.

        Draining collapses only consecutive ``xml`` frames to the latest one
        (an intermediate screen the loop would have skipped anyway). Control
        signals (``external_app`` / ``finish`` / ``no_change``) carry loop
        semantics the collector must act on, so a control signal stops the
        drain and is returned immediately; anything queued behind it is left in
        place for the next call. If the queue is empty, blocks up to `timeout`
        for a new signal.

        Thread-safe: the receiver thread concurrently ``put``s onto the same
        ``Queue``; ``get_nowait`` / ``get`` are the only reads.
        """
        latest_xml = None
        while True:
            try:
                signal = self._signal_queue.get_nowait()
            except Empty:
                break
            if signal[0] == "xml":
                latest_xml = signal  # collapse consecutive stale frames
                continue
            # Control signal — never silently dropped. Return it now and leave
            # any signals queued behind it for the next drain.
            return signal

        if latest_xml is not None:
            return latest_xml

        try:
            return self._signal_queue.get(timeout=timeout)
        except Empty:
            return None

    def clear_signal_queue(self):
        """Discard all queued signals."""
        discarded = 0
        while True:
            try:
                self._signal_queue.get_nowait()
                discarded += 1
            except Empty:
                break
        if discarded:
            logger.debug(f"Cleared {discarded} stale signals from queue")

    def _run(self):
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.settimeout(1.0)
        self._server_socket.bind((self.host, self.port))
        self._server_socket.listen(1)
        logger.info(f"Listening on {self.host}:{self.port}")

        while self._running:
            try:
                client, addr = self._server_socket.accept()
                self._client = client
                logger.info(f"Client connected from {addr[0]}:{addr[1]}")
                self._handle_client(client)
            except TimeoutError:
                continue
            except OSError:
                break

    def _handle_client(self, client: socket.socket):
        client.settimeout(60.0)
        try:
            while self._running:
                try:
                    msg_type = client.recv(1)
                except TimeoutError:
                    continue

                if not msg_type:
                    logger.info("Client disconnected")
                    self._signal_queue.put(("finish", None, None))
                    break

                msg_type = msg_type.decode("ascii")

                if msg_type == "P":
                    self._handle_package_name(client)
                elif msg_type == "S":
                    self._handle_screenshot(client)
                elif msg_type == "X":
                    self._handle_xml(client)
                elif msg_type == "N":
                    self._signal_queue.put(("no_change", None, None))
                    logger.debug("Received no-change signal from client")
                elif msg_type == "E":
                    self._handle_external_app(client)
                elif msg_type == "F":
                    logger.info("Received finish signal from client")
                    self._signal_queue.put(("finish", None, None))
                    if self.on_finish:
                        self.on_finish()
                    break
                else:
                    logger.warning(f"Unknown message type: {msg_type!r}")
        except (ConnectionResetError, BrokenPipeError) as e:
            logger.warning(f"Client disconnected: {e}")
        except TimeoutError:
            logger.warning("Client connection timed out")
        finally:
            client.close()
            self._client = None

    def _recv_text_line(self, client: socket.socket) -> str:
        original_timeout = client.gettimeout()
        client.settimeout(30.0)
        try:
            data = b""
            while not data.endswith(b"\n"):
                chunk = client.recv(1)
                if not chunk:
                    break
                data += chunk
            return data.decode("utf-8").strip()
        except TimeoutError:
            logger.warning("Timeout receiving text line")
            raise
        finally:
            client.settimeout(original_timeout)

    def _recv_binary(self, client: socket.socket) -> bytes:
        size_str = self._recv_text_line(client)
        file_size = int(size_str)
        data = b""
        remaining = file_size
        while remaining > 0:
            chunk = client.recv(min(remaining, BUFFER_SIZE))
            if not chunk:
                break
            data += chunk
            remaining -= len(chunk)
        return data

    def wait_for_package(self, timeout: float = 120.0) -> str | None:
        """Block until the client sends target package name via P message."""
        if self._package_event.wait(timeout):
            return self._target_package
        return None

    def _handle_package_name(self, client: socket.socket):
        """Receive target package name from client."""
        package_name = self._recv_text_line(client)
        self._target_package = package_name
        self._package_event.set()
        logger.info(f"Target package received: {package_name}")

    def _handle_screenshot(self, client: socket.socket):
        image_data = self._recv_binary(client)
        if self.on_screenshot:
            self.on_screenshot(image_data)
        logger.debug(f"Received screenshot: {len(image_data)} bytes")

    def _handle_xml(self, client: socket.socket):
        top_package = self._recv_text_line(client)
        activity_name = self._recv_text_line(client)
        target_package = self._recv_text_line(client)
        is_first_screen_str = self._recv_text_line(client)
        xml_data = self._recv_binary(client)
        raw_xml = xml_data.decode("utf-8").strip()
        raw_xml = raw_xml.replace('class=""', 'class="unknown"')

        is_first_screen = is_first_screen_str == "1"

        if self.on_xml:
            self.on_xml(raw_xml, top_package, target_package)

        self._latest_xml = raw_xml
        self._latest_xml_meta = {
            "top_package": top_package,
            "activity_name": activity_name,
            "target_package": target_package,
            "is_first_screen": is_first_screen,
        }
        self._xml_event.set()
        self._signal_queue.put(("xml", raw_xml, self._latest_xml_meta))
        logger.debug(
            f"Received XML: top={top_package}, activity={activity_name}, "
            f"target={target_package}, size={len(raw_xml)} bytes"
        )

    def _handle_external_app(self, client: socket.socket):
        payload_str = self._recv_text_line(client)
        try:
            payload = json.loads(payload_str)
        except json.JSONDecodeError:
            payload = {"raw": payload_str}

        if self.on_external_app:
            self.on_external_app(payload)
        self._signal_queue.put(("external_app", None, payload))
        logger.warning(f"External app detected: {payload}")
