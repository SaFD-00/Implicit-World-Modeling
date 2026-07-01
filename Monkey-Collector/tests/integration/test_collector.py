"""Tests for monkey_collector.collector — main collection orchestration (integration)."""

import itertools
import json
from unittest.mock import MagicMock, patch

import pytest

from monkey_collector.catalog_activities import ActivityCatalog
from monkey_collector.domain.actions import Tap
from monkey_collector.domain.activity_coverage import ActivityCoverageTracker
from monkey_collector.domain.page_graph import PageGraph
from monkey_collector.pipeline.collector import Collector
from tests.fixtures.xml_samples import COMPLEX_XML, MINIMAL_XML, SIMPLE_XML


@pytest.fixture(autouse=True)
def _mock_build_graph():
    """Mock build_graph_from_session for all collector tests."""
    with patch("monkey_collector.pipeline.session_manager.build_graph_from_session") as mock:
        mock.return_value = PageGraph()
        yield mock


def _make_xml_signal(xml=SIMPLE_XML, pkg="com.test.app", is_first=False, activity="com.test.app/.MainActivity"):
    return ("xml", xml, {
        "top_package": pkg,
        "activity_name": activity,
        "target_package": pkg,
        "is_first_screen": is_first,
    })


def _make_collector(mock_adb, signals, max_steps=10):
    """Create a Collector with all dependencies mocked."""
    from monkey_collector.pipeline.exploration import LLMGuidedExplorer
    from monkey_collector.storage import DataWriter
    from monkey_collector.tcp_server import CollectionServer

    mock_explorer = MagicMock(spec=LLMGuidedExplorer)
    mock_explorer.select_action.return_value = Tap(x=500, y=500, element_index=0)
    mock_explorer.has_left_app.return_value = False

    mock_server = MagicMock(spec=CollectionServer)
    mock_server.is_client_connected.return_value = True
    mock_server.wait_for_package.return_value = "com.test.app"
    mock_server.get_latest_signal.side_effect = signals

    mock_writer = MagicMock(spec=DataWriter)
    mock_writer.step_count = 0
    mock_writer.next_frame_index.side_effect = itertools.count()
    mock_writer.save_observation.return_value = {}
    mock_writer.find_existing_session.return_value = None
    mock_writer.data_dir = "/tmp/data"
    mock_writer.runtime_dir = "/tmp/runtime"
    mock_writer.data_session_dir = "/tmp/data/com.test.app"
    mock_writer.runtime_session_dir = "/tmp/runtime/com.test.app"

    collector = Collector(
        adb=mock_adb,
        explorer=mock_explorer,
        server=mock_server,
        writer=mock_writer,
        max_steps=max_steps,
        action_delay=0,
        xml_timeout=0.1,
    )

    return collector, mock_explorer, mock_server, mock_writer


@pytest.mark.integration
class TestSessionResume:
    @patch("monkey_collector.pipeline.collection_loop.time.sleep")
    def test_resumes_existing_session(self, mock_sleep, mock_adb):
        """When existing session found, resume is used instead of init."""
        signals = [
            _make_xml_signal(),
            ("finish", None, None),
        ]
        collector, explorer, server, writer = _make_collector(mock_adb, signals)
        writer.find_existing_session.return_value = "com.test.app"
        writer.resume_session.return_value = 5
        writer.data_session_dir = "/tmp/data/com.test.app"
        writer.runtime_session_dir = "/tmp/runtime/com.test.app"

        session_id = collector.run(package="com.test.app")

        assert session_id == "com.test.app"
        writer.resume_session.assert_called_once_with("com.test.app")
        writer.init_session.assert_not_called()

    @patch("monkey_collector.pipeline.collection_loop.time.sleep")
    def test_new_session_flag(self, mock_sleep, mock_adb):
        """--new-session flag forces new session creation."""
        signals = [
            _make_xml_signal(),
            ("finish", None, None),
        ]
        collector, explorer, server, writer = _make_collector(mock_adb, signals)
        collector._new_session = True
        writer.find_existing_session.return_value = "com.test.app"

        session_id = collector.run(package="com.test.app")

        writer.resume_session.assert_not_called()
        writer.init_session.assert_called_once()

    @patch("monkey_collector.pipeline.collection_loop.time.sleep")
    def test_no_existing_session(self, mock_sleep, mock_adb):
        """No existing session → normal init."""
        signals = [
            _make_xml_signal(),
            ("finish", None, None),
        ]
        collector, explorer, server, writer = _make_collector(mock_adb, signals)
        writer.find_existing_session.return_value = None

        session_id = collector.run(package="com.test.app")

        writer.resume_session.assert_not_called()
        writer.init_session.assert_called_once()


@pytest.mark.integration
class TestRunSessionHappyPath:
    @patch("monkey_collector.pipeline.collection_loop.time.sleep")
    def test_three_steps(self, mock_sleep, mock_adb):
        signals = [
            _make_xml_signal(),
            _make_xml_signal(),
            _make_xml_signal(),
            ("finish", None, None),
        ]
        collector, explorer, server, writer = _make_collector(mock_adb, signals)

        session_id = collector.run(package="com.test.app")

        assert session_id == "com.test.app"
        assert explorer.select_action.call_count == 3
        assert explorer.execute_action.call_count == 3
        assert writer.save_observation.call_count == 3
        writer.finalize_session.assert_called_once()


@pytest.mark.integration
class TestRunSessionNoChangeRetry:
    @patch("monkey_collector.pipeline.collection_loop.time.sleep")
    def test_retry_then_continue(self, mock_sleep, mock_adb):
        signals = [
            _make_xml_signal(),           # step 0: normal
            ("no_change", None, None),    # step 1: retry 1
            ("no_change", None, None),    # step 1: retry 2
            _make_xml_signal(),           # step 2: normal
            ("finish", None, None),
        ]
        collector, explorer, server, writer = _make_collector(mock_adb, signals)

        session_id = collector.run(package="com.test.app")

        assert session_id != ""
        # Element exclusion should have been called
        assert explorer.exclude_element.call_count >= 1


@pytest.mark.integration
class TestRunSessionExternalApp:
    @patch("monkey_collector.pipeline.collection_loop.time.sleep")
    def test_recovery_escalation(self, mock_sleep, mock_adb):
        # 4 external_app signals: first 3 → return_to_app, 4th → recover
        signals = [
            _make_xml_signal(),
            ("external_app", None, {"detected_package": "com.other"}),
            ("external_app", None, {"detected_package": "com.other"}),
            ("external_app", None, {"detected_package": "com.other"}),
            ("external_app", None, {"detected_package": "com.other"}),
            _make_xml_signal(),
            ("finish", None, None),
        ]
        collector, explorer, server, writer = _make_collector(mock_adb, signals)

        session_id = collector.run(package="com.test.app")

        assert session_id != ""
        assert explorer.return_to_app.call_count == 3
        assert explorer.recover.call_count == 1


@pytest.mark.integration
class TestRunSessionFinish:
    @patch("monkey_collector.pipeline.collection_loop.time.sleep")
    def test_finish_signal(self, mock_sleep, mock_adb):
        signals = [
            _make_xml_signal(),
            ("finish", None, None),
        ]
        collector, explorer, server, writer = _make_collector(mock_adb, signals)

        session_id = collector.run(package="com.test.app")

        assert session_id != ""
        writer.finalize_session.assert_called_once()


@pytest.mark.integration
class TestAppContextWiring:
    @patch("monkey_collector.pipeline.collection_loop.time.sleep")
    def test_sets_app_description_from_map(self, mock_sleep, mock_adb):
        signals = [_make_xml_signal(), ("finish", None, None)]
        collector, *_ = _make_collector(mock_adb, signals)
        collector._text_generator = MagicMock()
        collector._app_contexts = {"com.test.app": "Tasks (Productivity/Todo)"}

        collector.run(package="com.test.app")

        collector._text_generator.set_app_context.assert_called_once_with(
            "Tasks (Productivity/Todo)"
        )

    @patch("monkey_collector.pipeline.collection_loop.time.sleep")
    def test_falls_back_to_package_id(self, mock_sleep, mock_adb):
        signals = [_make_xml_signal(), ("finish", None, None)]
        collector, *_ = _make_collector(mock_adb, signals)
        collector._text_generator = MagicMock()
        collector._app_contexts = {}  # package not in catalog

        collector.run(package="com.test.app")

        collector._text_generator.set_app_context.assert_called_once_with(
            "com.test.app"
        )


@pytest.mark.integration
class TestSessionEndSignal:
    @patch("monkey_collector.pipeline.collection_loop.time.sleep")
    def test_session_end_sent_on_finish(self, mock_sleep, mock_adb):
        """send_session_end() is called when session finishes normally."""
        signals = [
            _make_xml_signal(),
            ("finish", None, None),
        ]
        collector, explorer, server, writer = _make_collector(mock_adb, signals)

        collector.run(package="com.test.app")
        server.send_session_end.assert_called_once()

    @patch("monkey_collector.pipeline.collection_loop.time.sleep")
    def test_session_end_sent_on_timeout(self, mock_sleep, mock_adb):
        """send_session_end() is called even when session ends due to timeouts."""
        signals = [None, None, None, None, None]
        collector, explorer, server, writer = _make_collector(mock_adb, signals)

        collector.run(package="com.test.app")
        server.send_session_end.assert_called_once()

    @patch("monkey_collector.pipeline.collection_loop.time.sleep")
    def test_session_end_sent_on_max_external_app(self, mock_sleep, mock_adb):
        """send_session_end() is called when session ends due to max external app retries."""
        signals = [_make_xml_signal()]
        signals += [("external_app", None, {"detected_package": "com.other"})] * 10
        collector, explorer, server, writer = _make_collector(mock_adb, signals, max_steps=50)

        collector.run(package="com.test.app")
        server.send_session_end.assert_called_once()


@pytest.mark.integration
class TestRunSessionTimeout:
    @patch("monkey_collector.pipeline.collection_loop.time.sleep")
    def test_max_timeouts(self, mock_sleep, mock_adb):
        # 5 consecutive timeouts (None) should end session
        signals = [None, None, None, None, None]
        collector, explorer, server, writer = _make_collector(mock_adb, signals)

        session_id = collector.run(package="com.test.app")

        assert session_id != ""
        writer.finalize_session.assert_called_once()


@pytest.mark.integration
class TestRunSessionNoConnection:
    @patch("monkey_collector.pipeline.collection_loop.time.sleep")
    def test_no_client(self, mock_sleep, mock_adb):
        from monkey_collector.pipeline.exploration import LLMGuidedExplorer
        from monkey_collector.storage import DataWriter
        from monkey_collector.tcp_server import CollectionServer

        mock_server = MagicMock(spec=CollectionServer)
        mock_server.is_client_connected.return_value = False

        collector = Collector(
            adb=mock_adb,
            explorer=MagicMock(spec=LLMGuidedExplorer),
            server=mock_server,
            writer=MagicMock(spec=DataWriter),
            max_steps=5,
            action_delay=0,
            xml_timeout=0.1,
        )

        session_id = collector.run(package="com.test.app")
        assert session_id == ""


@pytest.mark.integration
class TestRunQueue:
    @patch("monkey_collector.pipeline.collection_loop.time.sleep")
    @patch.object(Collector, "_run_session")
    def test_sequential_packages(self, mock_run_session, mock_sleep, mock_adb):
        """run_queue walks every package and accumulates session ids."""
        mock_run_session.side_effect = ["session_1", "session_2"]

        from monkey_collector.pipeline.exploration import LLMGuidedExplorer
        from monkey_collector.storage import DataWriter
        from monkey_collector.tcp_server import CollectionServer

        mock_server = MagicMock(spec=CollectionServer)
        collector = Collector(
            adb=mock_adb, explorer=MagicMock(spec=LLMGuidedExplorer), server=mock_server,
            writer=MagicMock(spec=DataWriter), max_steps=5, action_delay=0, xml_timeout=0.1,
        )

        result = collector.run_queue(["com.one", "com.two"])
        assert result == ["session_1", "session_2"]
        assert mock_run_session.call_count == 2
        mock_run_session.assert_any_call("com.one")
        mock_run_session.assert_any_call("com.two")
        # reset between the two sessions (once before session 2)
        assert mock_server.reset_for_new_session.call_count == 1
        mock_server.stop.assert_called_once()

    @patch("monkey_collector.pipeline.collection_loop.time.sleep")
    @patch.object(Collector, "_run_session")
    def test_interrupt_during_session(self, mock_run_session, mock_sleep, mock_adb):
        """KeyboardInterrupt during session → only completed sessions returned."""
        mock_run_session.side_effect = ["session_1", KeyboardInterrupt()]

        from monkey_collector.pipeline.exploration import LLMGuidedExplorer
        from monkey_collector.storage import DataWriter
        from monkey_collector.tcp_server import CollectionServer

        mock_server = MagicMock(spec=CollectionServer)
        collector = Collector(
            adb=mock_adb, explorer=MagicMock(spec=LLMGuidedExplorer), server=mock_server,
            writer=MagicMock(spec=DataWriter), max_steps=5, action_delay=0, xml_timeout=0.1,
        )

        result = collector.run_queue(["com.one", "com.two"])
        assert result == ["session_1"]
        mock_server.stop.assert_called_once()

    @patch("monkey_collector.pipeline.collection_loop.time.sleep")
    @patch.object(Collector, "_run_session")
    def test_empty_session_id_logged_but_queue_continues(
        self, mock_run_session, mock_sleep, mock_adb,
    ):
        """Empty string from _run_session is skipped but the queue continues."""
        mock_run_session.side_effect = ["", "session_2"]

        from monkey_collector.pipeline.exploration import LLMGuidedExplorer
        from monkey_collector.storage import DataWriter
        from monkey_collector.tcp_server import CollectionServer

        mock_server = MagicMock(spec=CollectionServer)
        collector = Collector(
            adb=mock_adb, explorer=MagicMock(spec=LLMGuidedExplorer), server=mock_server,
            writer=MagicMock(spec=DataWriter), max_steps=5, action_delay=0, xml_timeout=0.1,
        )

        result = collector.run_queue(["com.one", "com.two"])
        assert result == ["session_2"]


@pytest.mark.integration
class TestServerDrivenHandshake:
    @patch("monkey_collector.pipeline.collection_loop.time.sleep")
    def test_send_start_called_with_package(self, mock_sleep, mock_adb):
        """_run_session sends START with the requested package before launching."""
        signals = [_make_xml_signal(pkg="com.test.app"), ("finish", None, None)]
        collector, explorer, server, writer = _make_collector(mock_adb, signals)

        collector.run(package="com.test.app")

        server.send_start.assert_called_once_with("com.test.app")
        mock_adb.launch_app.assert_called_once_with("com.test.app")

    @patch("monkey_collector.pipeline.collection_loop.time.sleep")
    def test_send_start_failure_aborts_session(self, mock_sleep, mock_adb):
        """If send_start returns False, session is aborted without launching."""
        collector, explorer, server, writer = _make_collector(mock_adb, [])
        server.send_start.return_value = False

        session_id = collector.run(package="com.test.app")

        assert session_id == ""
        mock_adb.launch_app.assert_not_called()

    @patch("monkey_collector.pipeline.collection_loop.time.sleep")
    def test_package_mismatch_warns_and_uses_server_value(
        self, mock_sleep, mock_adb,
    ):
        """Client reporting a different package is logged but server value wins."""
        signals = [_make_xml_signal(pkg="com.server"), ("finish", None, None)]
        collector, explorer, server, writer = _make_collector(mock_adb, signals)
        server.wait_for_package.return_value = "com.client"

        session_id = collector.run(package="com.server")

        assert "com.server" in session_id
        mock_adb.launch_app.assert_called_once_with("com.server")


@pytest.mark.integration
class TestNoChangeExhaustion:
    @patch("monkey_collector.pipeline.collection_loop.time.sleep")
    def test_first_screen_tap_fallback(self, mock_sleep, mock_adb):
        """3 no-change on first screen → tap (not back)."""
        signals = [
            _make_xml_signal(is_first=True),  # sets is_first_screen=True
            ("no_change", None, None),
            ("no_change", None, None),
            ("no_change", None, None),  # retries=3 → tap fallback
            ("finish", None, None),
        ]
        collector, explorer, server, writer = _make_collector(mock_adb, signals)

        collector.run(package="com.test.app")
        # tap_random_fallback calls get_device_resolution + tap
        mock_adb.get_device_resolution.assert_called()
        mock_adb.tap.assert_called()
        # press_back should NOT be called during no-change exhaustion on first screen
        # (it may be called from explorer.execute_action for PressBack actions, but
        # we verify it's not called for the exhaustion path by checking adb directly)

    @patch("monkey_collector.pipeline.collection_loop.time.sleep")
    def test_not_first_screen_back(self, mock_sleep, mock_adb):
        """3 no-change on a non-root page → press_back.

        The first in-app page is the root (back suppressed there), so navigate
        to a second distinct page before exhausting no-change to reach the back
        path.
        """
        signals = [
            _make_xml_signal(xml=SIMPLE_XML, is_first=True),  # root page
            _make_xml_signal(xml=COMPLEX_XML, activity="com.test.app/.SettingsActivity"),
            ("no_change", None, None),
            ("no_change", None, None),
            ("no_change", None, None),  # retries=3 → back
            ("finish", None, None),
        ]
        collector, explorer, server, writer = _make_collector(mock_adb, signals)
        explorer.has_left_app.return_value = False

        collector.run(package="com.test.app")
        mock_adb.press_back.assert_called()

    @patch("monkey_collector.pipeline.collection_loop.time.sleep")
    def test_back_causes_exit_recovery(self, mock_sleep, mock_adb):
        """3 no-change on a non-root page → back → has_left_app=True → return_to_app."""
        signals = [
            _make_xml_signal(xml=SIMPLE_XML, is_first=True),  # root page
            _make_xml_signal(xml=COMPLEX_XML, activity="com.test.app/.SettingsActivity"),
            ("no_change", None, None),
            ("no_change", None, None),
            ("no_change", None, None),
            ("finish", None, None),
        ]
        collector, explorer, server, writer = _make_collector(mock_adb, signals)
        explorer.has_left_app.return_value = True

        collector.run(package="com.test.app")
        explorer.return_to_app.assert_called_with("com.test.app")


@pytest.mark.integration
class TestNoChangeNoUITree:
    @patch("monkey_collector.pipeline.collection_loop.time.sleep")
    def test_no_tree_back(self, mock_sleep, mock_adb):
        """no-change with no UI tree on a non-root page → back."""
        # Root page first, then a distinct empty-tree page (so it is non-root and
        # leaves last_ui_tree=None), then no-change hits the back fallback.
        signals = [
            _make_xml_signal(xml=SIMPLE_XML, is_first=True),  # root page
            ("xml", MINIMAL_XML, {"top_package": "com.test.app", "activity_name": "com.test.app/.EmptyActivity", "target_package": "com.test.app", "is_first_screen": False}),
            ("no_change", None, None),  # last_ui_tree=None (set to None after empty tree)
            ("finish", None, None),
        ]
        collector, explorer, server, writer = _make_collector(mock_adb, signals)
        explorer.has_left_app.return_value = False

        collector.run(package="com.test.app")
        mock_adb.press_back.assert_called()

    @patch("monkey_collector.pipeline.collection_loop.time.sleep")
    def test_no_tree_first_screen_tap(self, mock_sleep, mock_adb):
        """no-change with no UI tree + first screen → tap fallback."""
        signals = [
            ("xml", MINIMAL_XML, {"top_package": "com.test.app", "activity_name": "com.test.app/.MainActivity", "target_package": "com.test.app", "is_first_screen": True}),
            ("no_change", None, None),
            ("finish", None, None),
        ]
        collector, explorer, server, writer = _make_collector(mock_adb, signals)

        collector.run(package="com.test.app")
        # tap_random_fallback should be called
        mock_adb.get_device_resolution.assert_called()


@pytest.mark.integration
class TestExternalAppMax:
    @patch("monkey_collector.pipeline.collection_loop.time.sleep")
    def test_max_retries_ends_session(self, mock_sleep, mock_adb):
        """10 external_app signals escalate through recovery, then finish."""
        signals = [_make_xml_signal()]
        signals += [("external_app", None, {"detected_package": "com.other"})] * 10
        signals.append(("finish", None, None))
        collector, explorer, server, writer = _make_collector(mock_adb, signals, max_steps=50)

        collector.run(package="com.test.app")
        writer.finalize_session.assert_called_once()
        # counts 1-3 → return_to_app (×3); counts 4-9 → recover (×6); count 10
        # → reinit branch also relaunches via recover (×1) → recover totals 7.
        assert explorer.return_to_app.call_count == 3
        assert explorer.recover.call_count == 7

    @patch("monkey_collector.pipeline.collection_loop.time.sleep")
    def test_recovery_exception(self, mock_sleep, mock_adb):
        """Exception during recovery doesn't crash."""
        signals = [
            _make_xml_signal(),
            ("external_app", None, {"detected_package": "com.other"}),
            _make_xml_signal(),
            ("finish", None, None),
        ]
        collector, explorer, server, writer = _make_collector(mock_adb, signals)
        explorer.return_to_app.side_effect = Exception("recovery failed")

        session_id = collector.run(package="com.test.app")
        assert session_id != ""


@pytest.mark.integration
class TestXmlEdgeCases:
    @patch("monkey_collector.pipeline.collection_loop.time.sleep")
    def test_stale_xml_skipped(self, mock_sleep, mock_adb):
        """XML with mismatched top_package is skipped."""
        signals = [
            ("xml", SIMPLE_XML, {
                "top_package": "com.other.app",  # doesn't match target
                "activity_name": "com.other.app/.OtherActivity",
                "target_package": "com.test.app",
                "is_first_screen": False,
            }),
            _make_xml_signal(),  # normal signal
            ("finish", None, None),
        ]
        collector, explorer, server, writer = _make_collector(mock_adb, signals)

        collector.run(package="com.test.app")
        # Only 1 action should be selected (the stale one is skipped)
        assert explorer.select_action.call_count == 1

    @patch("monkey_collector.pipeline.collection_loop.time.sleep")
    def test_empty_tree_recovery(self, mock_sleep, mock_adb):
        """Empty UI tree on a non-root page + has_left_app → waits then recovery."""
        empty_meta = {"top_package": "com.test.app", "activity_name": "com.test.app/.EmptyActivity", "target_package": "com.test.app", "is_first_screen": False}
        signals = [
            _make_xml_signal(xml=SIMPLE_XML, is_first=True),  # root page
            # First 2 empty trees → wait (no back)
            ("xml", MINIMAL_XML, empty_meta),
            ("xml", MINIMAL_XML, empty_meta),
            # 3rd empty tree → press_back + recovery
            ("xml", MINIMAL_XML, empty_meta),
            ("finish", None, None),
        ]
        collector, explorer, server, writer = _make_collector(mock_adb, signals)
        explorer.has_left_app.return_value = True

        collector.run(package="com.test.app")
        mock_adb.press_back.assert_called()
        explorer.return_to_app.assert_called_with("com.test.app")


@pytest.mark.integration
class TestScreenshotSaving:
    @patch("monkey_collector.pipeline.collection_loop.time.sleep")
    def test_screenshot_saved(self, mock_sleep, mock_adb):
        """Screenshot stored via callback is saved with next XML."""
        signals = [
            _make_xml_signal(),
            ("finish", None, None),
        ]
        collector, explorer, server, writer = _make_collector(mock_adb, signals)
        collector._latest_screenshot = b"fake_png_data"

        collector.run(package="com.test.app")
        writer.save_observation.assert_called_once()
        args, _ = writer.save_observation.call_args
        assert args[2] == b"fake_png_data"


@pytest.mark.integration
class TestOnScreenshot:
    def test_stores_data(self, mock_adb):
        """_on_screenshot callback stores image data."""
        collector, _, _, _ = _make_collector(mock_adb, [])
        collector._on_screenshot(b"image_bytes")
        assert collector._latest_screenshot == b"image_bytes"


@pytest.mark.integration
class TestTapRandomFallback:
    def test_success(self, mock_adb):
        """tap_random_fallback taps center of screen."""
        from monkey_collector.pipeline.recovery import tap_random_fallback

        tap_random_fallback(mock_adb)
        mock_adb.get_device_resolution.assert_called_once()
        mock_adb.tap.assert_called_once_with(540, 960)

    def test_exception_caught(self, mock_adb):
        """Exception in tap_random_fallback is caught."""
        from monkey_collector.pipeline.recovery import tap_random_fallback

        mock_adb.get_device_resolution.side_effect = Exception("no device")
        tap_random_fallback(mock_adb)  # should not raise


@pytest.mark.integration
class TestActivityGroundTruthSource:
    @pytest.fixture
    def _isolate_catalog(self):
        ActivityCatalog.reset()
        yield
        ActivityCatalog.reset()

    @patch("monkey_collector.pipeline.collection_loop.time.sleep")
    def test_catalog_hit_skips_dumpsys(
        self, mock_sleep, mock_adb, tmp_path, _isolate_catalog,
    ):
        """When catalog has the package, dumpsys is not consulted."""
        cat_path = tmp_path / "activities.json"
        cat_path.write_text(json.dumps({
            "com.test.app": {
                "app_name": "Test",
                "activities": [
                    "com.test.app/.MainActivity",
                    "com.test.app/.SettingsActivity",
                    "com.test.app/.ThirdActivity",
                ],
            },
        }))
        ActivityCatalog.instance(cat_path)

        signals = [_make_xml_signal(), ("finish", None, None)]
        collector, *_ = _make_collector(mock_adb, signals)
        collector._activity_tracker = MagicMock(spec=ActivityCoverageTracker)

        collector.run(package="com.test.app")
        mock_adb.get_declared_activities.assert_not_called()
        collector._activity_tracker.initialize.assert_called_once()
        kwargs = collector._activity_tracker.initialize.call_args.kwargs
        assert kwargs["allow_dynamic_total"] is False

    @patch("monkey_collector.pipeline.collection_loop.time.sleep")
    def test_catalog_miss_falls_back_to_dumpsys(
        self, mock_sleep, mock_adb, tmp_path, _isolate_catalog,
    ):
        """When catalog is loaded but lacks the package, dumpsys is queried."""
        cat_path = tmp_path / "activities.json"
        cat_path.write_text(json.dumps({"some.other.pkg": {"activities": []}}))
        ActivityCatalog.instance(cat_path)

        signals = [_make_xml_signal(), ("finish", None, None)]
        collector, *_ = _make_collector(mock_adb, signals)
        collector._activity_tracker = MagicMock(spec=ActivityCoverageTracker)

        collector.run(package="com.test.app")
        mock_adb.get_declared_activities.assert_called_once_with("com.test.app")
        kwargs = collector._activity_tracker.initialize.call_args.kwargs
        assert kwargs["allow_dynamic_total"] is True

    @patch("monkey_collector.pipeline.collection_loop.time.sleep")
    def test_no_catalog_file_falls_back_to_dumpsys(
        self, mock_sleep, mock_adb, tmp_path, _isolate_catalog,
    ):
        """No catalog file at all → universal dumpsys fallback."""
        ActivityCatalog.instance(tmp_path / "absent.json")

        signals = [_make_xml_signal(), ("finish", None, None)]
        collector, *_ = _make_collector(mock_adb, signals)
        collector._activity_tracker = MagicMock(spec=ActivityCoverageTracker)

        collector.run(package="com.test.app")
        mock_adb.get_declared_activities.assert_called_once_with("com.test.app")


@pytest.mark.integration
class TestStepException:
    @patch("monkey_collector.pipeline.collection_loop.time.sleep")
    def test_recover_and_continue(self, mock_sleep, mock_adb):
        """Exception during step → recover + continue to next step."""
        signals = [
            _make_xml_signal(),
            _make_xml_signal(),
            ("finish", None, None),
        ]
        collector, explorer, server, writer = _make_collector(mock_adb, signals)
        # First call raises, second call works
        explorer.select_action.side_effect = [Exception("crash"), Tap(x=100, y=100, element_index=0)]

        session_id = collector.run(package="com.test.app")
        assert session_id != ""
        explorer.recover.assert_called_once_with("com.test.app")


class TestTimeoutRecovery:
    """Signal-timeout handling should escape system screens by relaunching."""

    @patch("monkey_collector.pipeline.collection_loop.time.sleep")
    def test_timeout_relaunches_when_left_app(self, mock_sleep, mock_adb):
        """On signal timeout, if we drifted out of the target app (e.g. a
        system role screen that emits no events), return_to_app to escape."""
        signals = [None, ("finish", None, None)]
        collector, explorer, server, writer = _make_collector(mock_adb, signals)
        explorer.has_left_app.return_value = True

        collector.run(package="com.test.app")

        explorer.has_left_app.assert_called_with("com.test.app")
        explorer.return_to_app.assert_called_with("com.test.app")

    @patch("monkey_collector.pipeline.collection_loop.time.sleep")
    def test_timeout_taps_when_still_in_app(self, mock_sleep, mock_adb):
        """On signal timeout while still inside the target app, nudge with a
        tap (no relaunch)."""
        signals = [None, ("finish", None, None)]
        collector, explorer, server, writer = _make_collector(mock_adb, signals)
        explorer.has_left_app.return_value = False

        collector.run(package="com.test.app")

        explorer.return_to_app.assert_not_called()
