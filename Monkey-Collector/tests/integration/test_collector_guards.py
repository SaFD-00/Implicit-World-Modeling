"""Integration tests for collection-loop guards: keyboard dismissal, permission
dialog handling, and the progress-based external-app counter."""

from unittest.mock import patch

import pytest

from monkey_collector.domain.page_graph import PageGraph
from tests.fixtures.xml_samples import COMPLEX_XML, SIMPLE_XML
from tests.integration.test_collector import _make_collector, _make_xml_signal

_PERMISSION_XML = (
    '<hierarchy>'
    '<node class="android.widget.Button" '
    'resource-id="com.android.permissioncontroller:id/permission_allow_button" '
    'text="Allow" clickable="true" bounds="[0,0][200,100]"/>'
    '</hierarchy>'
)


@pytest.fixture(autouse=True)
def _mock_build_graph():
    with patch(
        "monkey_collector.pipeline.session_manager.build_graph_from_session"
    ) as mock:
        mock.return_value = PageGraph()
        yield mock


@pytest.mark.integration
class TestKeyboardGuard:
    @patch("monkey_collector.pipeline.collection_loop.time.sleep")
    def test_keyboard_signal_dismissed_not_recorded(self, _sleep, mock_adb):
        signals = [
            _make_xml_signal(activity="com.test.app/...SoftInputWindow"),
            ("finish", None, None),
        ]
        collector, explorer, server, writer = _make_collector(mock_adb, signals)

        collector.run(package="com.test.app")

        # keyboard is dismissed via back and not treated as a page/step content
        mock_adb.press_back.assert_called()
        explorer.select_action.assert_not_called()
        writer.save_xml.assert_not_called()


@pytest.mark.integration
class TestPermissionGuard:
    @patch("monkey_collector.pipeline.collection_loop.time.sleep")
    def test_permission_dialog_taps_allow(self, _sleep, mock_adb):
        signals = [
            ("xml", _PERMISSION_XML, {
                "top_package": "com.google.android.permissioncontroller",
                "activity_name": "com.android.permissioncontroller/.GrantPermissionsActivity",
                "target_package": "com.test.app",
                "is_first_screen": False,
            }),
            ("finish", None, None),
        ]
        collector, explorer, server, writer = _make_collector(mock_adb, signals)

        collector.run(package="com.test.app")

        # the Allow button (center of [0,0][200,100]) is tapped; no normal
        # exploration / save happens for the dialog frame
        mock_adb.tap.assert_called_with(100, 50)
        explorer.select_action.assert_not_called()
        writer.save_xml.assert_not_called()


@pytest.mark.integration
class TestExternalCounterAccumulates:
    @patch("monkey_collector.pipeline.collection_loop.time.sleep")
    def test_external_loop_on_known_page_ends_session(self, _sleep, mock_adb):
        # One in-app frame (new page), then 10 external detections each followed
        # by a return to the SAME (known) page. Because the counter only resets
        # on a brand-new page, the 10th external must end the session.
        ext = ("external_app", None, {"detected_package": "com.other"})
        signals = [_make_xml_signal()]
        for _ in range(9):
            signals.append(ext)
            signals.append(_make_xml_signal())
        signals.append(ext)  # 10th external → MAX_EXTERNAL_APP_RETRIES → end
        signals.append(("finish", None, None))  # safety net

        collector, explorer, server, writer = _make_collector(
            mock_adb, signals, max_steps=50,
        )

        session_id = collector.run(package="com.test.app")

        assert session_id != ""
        # counts 1-3 → return_to_app (×3); counts 4-9 → recover (×6); the 10th
        # (count==MAX) enters the reinit branch which also relaunches via
        # recover (×1) before resetting the counter, so recover totals 7. The
        # session then ends on the trailing finish signal.
        assert explorer.return_to_app.call_count == 3
        assert explorer.recover.call_count == 7
        writer.finalize_session.assert_called_once()


@pytest.mark.integration
class TestOpenAppRecording:
    @patch("monkey_collector.pipeline.collection_loop.time.sleep")
    def test_open_app_logged_once_per_excursion(self, _sleep, mock_adb):
        # Two excursions, each = several externals in a row then a return to an
        # in-app frame. open_app must be logged exactly once per excursion
        # (dedup within an excursion; the in-app frame between excursions
        # re-arms the dedup flag).
        ext = ("external_app", None, {"detected_package": "com.other"})
        signals = [
            _make_xml_signal(),
            ext, ext, ext,            # excursion 1
            _make_xml_signal(),       # back in-app → re-arm
            ext, ext,                 # excursion 2
            _make_xml_signal(),
            ("finish", None, None),
        ]
        collector, explorer, server, writer = _make_collector(
            mock_adb, signals, max_steps=50,
        )
        explorer.return_to_app.return_value = True  # an actual relaunch

        collector.run(package="com.test.app")

        assert writer.log_open_app.call_count == 2
        # Always the target package, never the external one.
        for call in writer.log_open_app.call_args_list:
            assert call.args[0] == "com.test.app"

    @patch("monkey_collector.pipeline.collection_loop.time.sleep")
    def test_no_open_app_when_back_suffices(self, _sleep, mock_adb):
        # return_to_app returns False (a single Back landed back in the app, no
        # relaunch) → nothing to record.
        ext = ("external_app", None, {"detected_package": "com.other"})
        signals = [
            _make_xml_signal(),
            ext,
            _make_xml_signal(),
            ("finish", None, None),
        ]
        collector, explorer, server, writer = _make_collector(
            mock_adb, signals, max_steps=50,
        )
        explorer.return_to_app.return_value = False

        collector.run(package="com.test.app")

        writer.log_open_app.assert_not_called()

    @patch("monkey_collector.pipeline.collection_loop.time.sleep")
    def test_external_recovery_draws_no_navigation_edge(self, _sleep, mock_adb):
        # A→B is a normal navigation edge. After B the device drifts external and
        # recovery lands back on A. That return must NOT add a B→A edge: external
        # recovery is not navigation. Only the A→B edge survives.
        ext = ("external_app", None, {"detected_package": "com.other"})
        signals = [
            _make_xml_signal(xml=SIMPLE_XML),
            _make_xml_signal(xml=COMPLEX_XML),   # edge A→B
            ext,                                 # drift + recovery (clears last_action)
            _make_xml_signal(xml=SIMPLE_XML),    # back on A — must not add B→A
            ("finish", None, None),
        ]
        collector, explorer, server, writer = _make_collector(
            mock_adb, signals, max_steps=50,
        )
        explorer.return_to_app.return_value = True

        collector.run(package="com.test.app")

        edges = collector._live_page_graph.edges
        assert len(edges) == 1
        assert edges[0].action_type == "tap"
