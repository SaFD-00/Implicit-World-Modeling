"""Tests for the CAPTURE poke (F2) and the no_change permission guard.

The client pushes a frame only when the accessibility service observes a
change, so a settled screen that fired no event stays silent for the whole
signal-timeout window even though it has a frame to give. The loop now slices
that window: after ``poke_delay`` of silence it sends a CAPTURE and keeps
waiting, up to MAX_POKES_PER_WAIT times.

The invariant under test: the slices sum to exactly ``xml_timeout``, so a
client that never answers still times out at the same moment it always did —
the timeout counter and the MAX_SIGNAL_TIMEOUTS escalation are untouched.
Timing is asserted on the ``timeout`` argument sequence handed to
``get_latest_signal``, never on the wall clock.

Harness reuses the collector builder from the integration collector tests, as
tests/unit/test_signal_timeout_escalation.py does.
"""

from unittest.mock import patch

import pytest

from monkey_collector.pipeline.collection_loop import CollectionState, run_collection_loop
from monkey_collector.pipeline.recovery import MAX_POKES_PER_WAIT
from tests.integration.test_collector import _make_collector, _make_xml_signal

XML_TIMEOUT = 1.0
POKE_DELAY = 0.05

# uiautomator dump of a runtime permission dialog: the grant button is the only
# clickable node, and the (non-clickable) title also contains "Allow" — the scan
# must not tap the title.
PERMISSION_DUMP = """\
<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" text="Allow Test App to access photos?" resource-id=""
        class="android.widget.TextView" content-desc="" checkable="false"
        checked="false" clickable="false" enabled="true" focusable="false"
        focused="false" scrollable="false" long-clickable="false"
        password="false" selected="false" bounds="[100,600][980,700]"
        package="com.google.android.permissioncontroller"
        visible-to-user="true" important="true" />
  <node index="1" text="While using the app" resource-id=""
        class="android.widget.Button" content-desc="" checkable="false"
        checked="false" clickable="true" enabled="true" focusable="true"
        focused="false" scrollable="false" long-clickable="false"
        password="false" selected="false" bounds="[100,800][980,900]"
        package="com.google.android.permissioncontroller"
        visible-to-user="true" important="true" />
</hierarchy>
"""


def _run_loop(mock_adb, signals, *, poke_delay=POKE_DELAY, xml_timeout=XML_TIMEOUT):
    collector, explorer, server, writer = _make_collector(mock_adb, signals, max_steps=100)
    collector.xml_timeout = xml_timeout
    collector.poke_delay = poke_delay
    state = CollectionState(step=0, max_step=100)
    with patch("monkey_collector.pipeline.collection_loop.time.sleep"):
        run_collection_loop(collector, state, "com.test.app")
    return state, collector, explorer, server


def _timeouts(server):
    """The `timeout` argument handed to each get_latest_signal call, in order."""
    return [call.kwargs["timeout"] for call in server.get_latest_signal.call_args_list]


class TestPokeFires:
    def test_poke_on_silence_then_signal_arrives(self, mock_adb):
        # Silent for one poke_delay slice, then the poked client answers: the
        # signal is consumed normally and NO timeout is counted.
        signals = [None, _make_xml_signal(), ("finish", None, None)]
        state, _, _, server = _run_loop(mock_adb, signals)

        assert server.send_capture_request.call_count == 1
        assert state.timeout_count == 0
        # First wait: two slices of poke_delay (poke fired between them).
        assert _timeouts(server)[:2] == [POKE_DELAY, POKE_DELAY]

    @patch("monkey_collector.pipeline.collection_loop.nudge_static_screen")
    def test_total_silence_pokes_twice_and_counts_one_timeout(self, _nudge, mock_adb):
        # A client that never answers: exactly MAX_POKES_PER_WAIT pokes, and the
        # wait still ends at exactly xml_timeout → one timeout, no escalation.
        signals = [None, None, None, ("finish", None, None)]
        state, _, _, server = _run_loop(mock_adb, signals)

        assert server.send_capture_request.call_count == MAX_POKES_PER_WAIT
        assert state.timeout_count == 1
        assert mock_adb.force_stop.call_count == 0

        # The three slices of the silent wait sum to exactly xml_timeout.
        slices = _timeouts(server)[:3]
        assert slices == pytest.approx(
            [POKE_DELAY, POKE_DELAY, XML_TIMEOUT - 2 * POKE_DELAY]
        )
        assert sum(slices) == pytest.approx(XML_TIMEOUT)


class TestPokeDisabled:
    @patch("monkey_collector.pipeline.collection_loop.nudge_static_screen")
    def test_zero_delay_is_bit_identical_to_legacy(self, _nudge, mock_adb):
        # poke_delay <= 0 disables poking: one full-timeout wait, as before.
        signals = [None, ("finish", None, None)]
        state, _, _, server = _run_loop(mock_adb, signals, poke_delay=0.0)

        assert server.send_capture_request.call_count == 0
        assert state.timeout_count == 1
        assert _timeouts(server)[0] == XML_TIMEOUT

    @patch("monkey_collector.pipeline.collection_loop.nudge_static_screen")
    def test_delay_at_or_above_timeout_disables_poking(self, _nudge, mock_adb):
        # No room to poke inside the window → legacy single wait.
        signals = [None, ("finish", None, None)]
        state, _, _, server = _run_loop(
            mock_adb, signals, poke_delay=XML_TIMEOUT, xml_timeout=XML_TIMEOUT
        )

        assert server.send_capture_request.call_count == 0
        assert state.timeout_count == 1
        assert _timeouts(server)[0] == XML_TIMEOUT


class TestNoChangePermissionGuard:
    """A poked client answers fast, so an ungranted permission dialog now
    surfaces as no_change instead of the signal timeout where the adb grant
    lives. _handle_no_change must grant it rather than blame the last element.
    """

    def test_permission_dialog_is_granted_not_excluded(self, mock_adb):
        mock_adb.get_current_package.return_value = (
            "com.google.android.permissioncontroller"
        )
        mock_adb.shell.return_value = PERMISSION_DUMP

        # xml (sets last_action) → no_change (dialog was covering the screen).
        signals = [
            _make_xml_signal(),
            ("no_change", None, None),
            ("finish", None, None),
        ]
        state, _, explorer, _ = _run_loop(mock_adb, signals)

        # Granted via the dialog's own button — center of [100,800][980,900].
        mock_adb.tap.assert_called_once_with(540, 850)
        # The element did nothing because a dialog covered it: do not blame it.
        explorer.exclude_element.assert_not_called()
        assert state.no_change_retries == 0

    def test_non_dialog_no_change_keeps_the_legacy_path(self, mock_adb):
        # Default mock_adb foreground is the target app (not a dialog).
        signals = [
            _make_xml_signal(),
            ("no_change", None, None),
            ("finish", None, None),
        ]
        state, _, explorer, _ = _run_loop(mock_adb, signals)

        explorer.exclude_element.assert_called_once_with(0)
        assert state.no_change_retries == 1
