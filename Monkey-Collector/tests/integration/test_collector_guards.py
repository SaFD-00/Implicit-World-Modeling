"""Integration tests for collection-loop guards: keyboard dismissal, permission
dialog handling, and the progress-based external-app counter."""

from unittest.mock import patch

import pytest

from monkey_collector.domain.page_graph import PageGraph
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
        # counts 1-3 → return_to_app, 4-9 → recover, 10th → immediate end
        assert explorer.return_to_app.call_count == 3
        assert explorer.recover.call_count == 6
        writer.finalize_session.assert_called_once()
