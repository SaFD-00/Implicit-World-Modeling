"""Tests for monkey_collector.server — CollectionServer TCP protocol and signal queue."""

import json
import socket
import threading
from unittest.mock import MagicMock

import pytest

from monkey_collector.tcp_server import CollectionServer


@pytest.fixture
def srv():
    """CollectionServer instance (not started)."""
    return CollectionServer(host="127.0.0.1", port=0)


class TestSendAction:
    def test_no_client(self, srv):
        assert srv.send_action({"action_type": "tap"}) is False

    def test_success(self, srv):
        mock_socket = MagicMock()
        srv._client = mock_socket
        result = srv.send_action({"action_type": "tap", "x": 100})
        assert result is True
        mock_socket.sendall.assert_called_once()
        data = mock_socket.sendall.call_args[0][0]
        assert data.endswith(b"\r\n")
        decoded = json.loads(data.decode("utf-8").strip())
        assert decoded["action_type"] == "tap"

    def test_broken_pipe(self, srv):
        mock_socket = MagicMock()
        mock_socket.sendall.side_effect = BrokenPipeError("broken")
        srv._client = mock_socket
        assert srv.send_action({"action_type": "tap"}) is False


class TestSignalQueue:
    def test_clear(self, srv):
        srv._signal_queue.put(("xml", "data1", {}))
        srv._signal_queue.put(("no_change", None, None))
        srv._signal_queue.put(("xml", "data2", {}))
        srv.clear_signal_queue()
        assert srv._signal_queue.empty()

    def test_get_latest_drains_stale(self, srv):
        srv._signal_queue.put(("xml", "stale1", {"step": 1}))
        srv._signal_queue.put(("xml", "stale2", {"step": 2}))
        srv._signal_queue.put(("xml", "latest", {"step": 3}))
        result = srv.get_latest_signal(timeout=1.0)
        assert result is not None
        assert result[1] == "latest"
        assert result[2]["step"] == 3

    def test_get_latest_blocks_on_empty(self, srv):
        # Put a signal from another thread after a delay
        def put_later():
            import time
            time.sleep(0.1)
            srv._signal_queue.put(("xml", "delayed", {}))

        t = threading.Thread(target=put_later)
        t.start()
        result = srv.get_latest_signal(timeout=2.0)
        t.join()
        assert result is not None
        assert result[1] == "delayed"

    def test_get_latest_timeout(self, srv):
        result = srv.get_latest_signal(timeout=0.1)
        assert result is None

    def test_control_signal_after_stale_xml_returned(self, srv):
        # [xml, xml, external_app] → the control signal must not be dropped.
        srv._signal_queue.put(("xml", "stale1", {}))
        srv._signal_queue.put(("xml", "stale2", {}))
        srv._signal_queue.put(("external_app", None, {"pkg": "com.other"}))
        result = srv.get_latest_signal(timeout=1.0)
        assert result[0] == "external_app"
        assert result[2] == {"pkg": "com.other"}

    def test_leading_control_signal_preserves_trailing_xml(self, srv):
        # [external_app, xml, xml] → return external_app immediately and LEAVE
        # the two xml frames in the queue for the next call. This is the exact
        # property the fix guarantees: control signals are never overwritten.
        srv._signal_queue.put(("external_app", None, {"pkg": "com.other"}))
        srv._signal_queue.put(("xml", "later1", {}))
        srv._signal_queue.put(("xml", "later2", {}))
        result = srv.get_latest_signal(timeout=1.0)
        assert result[0] == "external_app"
        assert not srv._signal_queue.empty()
        # Next call collapses the two trailing xml to the latest.
        nxt = srv.get_latest_signal(timeout=1.0)
        assert nxt[0] == "xml"
        assert nxt[1] == "later2"

    def test_lone_finish_never_dropped(self, srv):
        srv._signal_queue.put(("finish", None, None))
        result = srv.get_latest_signal(timeout=1.0)
        assert result[0] == "finish"

    def test_no_change_stops_drain_and_preserves_trailing(self, srv):
        # [xml, no_change, xml] → stop at no_change, leave the trailing xml.
        srv._signal_queue.put(("xml", "stale", {}))
        srv._signal_queue.put(("no_change", None, None))
        srv._signal_queue.put(("xml", "after", {}))
        result = srv.get_latest_signal(timeout=1.0)
        assert result[0] == "no_change"
        assert not srv._signal_queue.empty()

    def test_all_xml_collapses_to_latest(self, srv):
        # [xml, xml, xml] → the last xml (pure stale-frame collapse path).
        srv._signal_queue.put(("xml", "a", {}))
        srv._signal_queue.put(("xml", "b", {}))
        srv._signal_queue.put(("xml", "c", {}))
        result = srv.get_latest_signal(timeout=1.0)
        assert result[0] == "xml"
        assert result[1] == "c"


class TestWaitForChangeSignal:
    def test_xml(self, srv):
        srv._signal_queue.put(("xml", "<xml/>", {"top_package": "com.test"}))
        result = srv.wait_for_change_signal(timeout=1.0)
        assert result is not None
        assert result[0] == "xml"

    def test_no_change(self, srv):
        srv._signal_queue.put(("no_change", None, None))
        result = srv.wait_for_change_signal(timeout=1.0)
        assert result[0] == "no_change"

    def test_timeout(self, srv):
        result = srv.wait_for_change_signal(timeout=0.1)
        assert result is None


class TestSendSessionEnd:
    def test_sends_session_end(self, srv):
        mock_socket = MagicMock()
        srv._client = mock_socket
        result = srv.send_session_end()
        assert result is True
        data = mock_socket.sendall.call_args[0][0]
        decoded = json.loads(data.decode("utf-8").strip())
        assert decoded["type"] == "SESSION_END"

    def test_no_client(self, srv):
        assert srv.send_session_end() is False


class TestResetForNewSession:
    def test_clears_state(self, srv):
        sentinel_client = MagicMock()
        srv._target_package = "com.test.app"
        srv._package_event.set()
        srv._latest_xml = "<xml/>"
        srv._latest_xml_meta = {"key": "val"}
        srv._client = sentinel_client
        srv._signal_queue.put(("xml", "data", {}))

        srv.reset_for_new_session()

        assert srv._target_package is None
        assert not srv._package_event.is_set()
        assert srv._latest_xml is None
        assert srv._latest_xml_meta is None
        assert srv._signal_queue.empty()
        assert srv._client is sentinel_client
        sentinel_client.close.assert_not_called()

    def test_preserves_client_socket(self, srv):
        # Per ARCHITECTURE handshake: reset must not touch the fresh socket
        # the client has already reconnected with. Closing it here would
        # force a second reconnect that the Android service does not
        # perform, causing the next wait_for_connection to time out.
        mock_socket = MagicMock()
        srv._client = mock_socket

        srv.reset_for_new_session()

        mock_socket.close.assert_not_called()
        assert srv._client is mock_socket


class TestIsClientConnected:
    def test_not_connected(self, srv):
        assert srv.is_client_connected() is False

    def test_connected(self, srv):
        srv._client = MagicMock()
        assert srv.is_client_connected() is True


class TestWaitForPackage:
    def test_receives_package(self, srv):
        def set_package():
            import time
            time.sleep(0.1)
            srv._target_package = "com.test.app"
            srv._package_event.set()

        t = threading.Thread(target=set_package)
        t.start()
        result = srv.wait_for_package(timeout=2.0)
        t.join()
        assert result == "com.test.app"

    def test_timeout(self, srv):
        result = srv.wait_for_package(timeout=0.1)
        assert result is None


class TestProtocolViaSocketpair:
    """Integration test: feed binary protocol through a real socket pair."""

    def test_xml_message(self, srv):
        """Send an X message through socketpair and verify signal queue."""
        client_sock, server_sock = socket.socketpair()
        try:
            srv._client = server_sock
            srv._running = True

            # Simulate X message in a thread
            def handle():
                srv._handle_xml(server_sock)

            # Build X message data
            top_pkg = b"com.test.app\n"
            activity_name = b"com.test.app/.MainActivity\n"
            target_pkg = b"com.test.app\n"
            is_first = b"0\n"
            xml_data = b"<hierarchy><node /></hierarchy>"
            size_line = f"{len(xml_data)}\n".encode()

            # Send from client side
            client_sock.sendall(top_pkg + activity_name + target_pkg + is_first + size_line + xml_data)

            t = threading.Thread(target=handle)
            t.start()
            t.join(timeout=5)

            # Check signal queue
            result = srv._signal_queue.get(timeout=2.0)
            assert result[0] == "xml"
            assert "<hierarchy>" in result[1]
            assert result[2]["top_package"] == "com.test.app"
            assert result[2]["activity_name"] == "com.test.app/.MainActivity"
            assert result[2]["is_first_screen"] is False
        finally:
            client_sock.close()
            server_sock.close()

    def test_screenshot_callback(self, srv):
        """Send an S message and verify the callback fires."""
        client_sock, server_sock = socket.socketpair()
        received = []
        srv.on_screenshot = lambda data: received.append(data)

        try:
            srv._client = server_sock
            srv._running = True

            fake_image = b"\x89PNG_fake_image_data"
            size_line = f"{len(fake_image)}\n".encode()

            client_sock.sendall(size_line + fake_image)

            def handle():
                srv._handle_screenshot(server_sock)

            t = threading.Thread(target=handle)
            t.start()
            t.join(timeout=5)

            assert len(received) == 1
            assert received[0] == fake_image
        finally:
            client_sock.close()
            server_sock.close()

    def test_package_message(self, srv):
        """P message sets target package via _handle_package_name."""
        client_sock, server_sock = socket.socketpair()
        try:
            package_line = b"com.test.app\n"
            client_sock.sendall(package_line)

            def handle():
                srv._handle_package_name(server_sock)

            t = threading.Thread(target=handle)
            t.start()
            t.join(timeout=5)

            result = srv.wait_for_package(timeout=1.0)
            assert result == "com.test.app"
        finally:
            client_sock.close()
            server_sock.close()

    def test_external_app_valid_json(self, srv):
        """E message with valid JSON → signal queue + callback."""
        client_sock, server_sock = socket.socketpair()
        received = []
        srv.on_external_app = lambda p: received.append(p)
        try:
            payload = json.dumps({"detected_package": "com.other"}) + "\n"
            client_sock.sendall(payload.encode("utf-8"))

            def handle():
                srv._handle_external_app(server_sock)

            t = threading.Thread(target=handle)
            t.start()
            t.join(timeout=5)

            signal = srv._signal_queue.get(timeout=1.0)
            assert signal[0] == "external_app"
            assert signal[2]["detected_package"] == "com.other"
            assert len(received) == 1
        finally:
            client_sock.close()
            server_sock.close()

    def test_external_app_invalid_json(self, srv):
        """E message with non-JSON → payload wraps in {"raw": ...}."""
        client_sock, server_sock = socket.socketpair()
        try:
            client_sock.sendall(b"not-valid-json\n")

            def handle():
                srv._handle_external_app(server_sock)

            t = threading.Thread(target=handle)
            t.start()
            t.join(timeout=5)

            signal = srv._signal_queue.get(timeout=1.0)
            assert signal[0] == "external_app"
            assert signal[2] == {"raw": "not-valid-json"}
        finally:
            client_sock.close()
            server_sock.close()

    def test_no_change_and_finish(self, srv):
        """N and F messages produce correct signals via _handle_client."""
        client_sock, server_sock = socket.socketpair()
        try:
            srv._running = True
            # Send N then F
            client_sock.sendall(b"NF")

            def handle():
                srv._handle_client(server_sock)

            t = threading.Thread(target=handle)
            t.start()
            t.join(timeout=5)

            sig1 = srv._signal_queue.get(timeout=1.0)
            assert sig1[0] == "no_change"
            sig2 = srv._signal_queue.get(timeout=1.0)
            assert sig2[0] == "finish"
        finally:
            client_sock.close()
            server_sock.close()


class TestStartStopLifecycle:
    def test_start_and_stop(self):
        srv = CollectionServer(host="127.0.0.1", port=0)
        # Use port 0 to let OS assign a free port
        srv.port = 0
        # We can't easily test with port=0 since _run binds directly
        # Instead, use a high random port
        import random
        srv.port = random.randint(40000, 50000)
        srv.start()
        assert srv._running is True
        assert srv._thread is not None
        srv.stop()
        assert srv._running is False


class TestWaitForXml:
    def test_success(self, srv):
        """wait_for_xml returns (xml_str, meta) when xml_event is set."""
        def set_xml():
            import time
            time.sleep(0.1)
            srv._latest_xml = "<hierarchy />"
            srv._latest_xml_meta = {"top_package": "com.test"}
            srv._xml_event.set()

        t = threading.Thread(target=set_xml)
        t.start()
        result = srv.wait_for_xml(timeout=2.0)
        t.join()
        assert result is not None
        assert result[0] == "<hierarchy />"
        assert result[1]["top_package"] == "com.test"

    def test_timeout(self, srv):
        result = srv.wait_for_xml(timeout=0.1)
        assert result is None


class TestClientDisconnection:
    def test_disconnect_puts_finish(self, srv):
        """Client disconnect (empty recv) puts finish signal."""
        client_sock, server_sock = socket.socketpair()
        try:
            srv._running = True
            # Close client immediately → server recv returns b""
            client_sock.close()

            def handle():
                srv._handle_client(server_sock)

            t = threading.Thread(target=handle)
            t.start()
            t.join(timeout=5)

            signal = srv._signal_queue.get(timeout=1.0)
            assert signal[0] == "finish"
        finally:
            server_sock.close()
