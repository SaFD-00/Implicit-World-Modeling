"""Tests for signal-timeout escalation in run_collection_loop.

MAX_SIGNAL_TIMEOUTS (3) consecutive signal timeouts escalate to a
force-relaunch (nudge on timeouts 1-2, relaunch on the 3rd); an xml frame in
between resets the timeout counter; and MAX_TIMEOUT_REINITS caps total reinits
on a permanently dead session.

Harness reuses the collector builder from the integration collector tests but
drives ``run_collection_loop`` directly (bypassing ``_run_session``) so the
only ``force_stop`` calls come from the reinit path, not session startup.
"""

from unittest.mock import patch

from monkey_collector.pipeline.collection_loop import CollectionState, run_collection_loop
from monkey_collector.pipeline.recovery import MAX_TIMEOUT_REINITS
from tests.integration.test_collector import _make_collector, _make_xml_signal


def _run_loop(mock_adb, signals, max_steps=100):
    collector, explorer, server, writer = _make_collector(
        mock_adb, signals, max_steps=max_steps
    )
    state = CollectionState(step=0, max_step=max_steps)
    with patch("monkey_collector.pipeline.collection_loop.time.sleep"):
        run_collection_loop(collector, state, "com.test.app")
    return state, collector, explorer, server, writer


class TestSignalTimeoutEscalation:
    @patch("monkey_collector.pipeline.collection_loop.nudge_static_screen")
    def test_escalates_on_third_timeout(self, mock_nudge, mock_adb):
        # Timeouts 1-2 nudge the static screen; the 3rd escalates to a
        # force-relaunch. has_left_app=False + a non-permission foreground keep
        # both earlier timeouts on the nudge path.
        signals = [None, None, None, ("finish", None, None)]
        _run_loop(mock_adb, signals)

        assert mock_adb.force_stop.call_count == 1
        assert mock_adb.launch_app.call_count == 1
        assert mock_nudge.call_count == 2

    @patch("monkey_collector.pipeline.collection_loop.nudge_static_screen")
    def test_timeout_count_resets_on_xml(self, mock_nudge, mock_adb):
        # Two timeouts, an xml frame (resets the counter), then two more
        # timeouts never reach the escalation threshold: no force-relaunch.
        signals = [
            None, None,
            _make_xml_signal(),
            None, None,
            ("finish", None, None),
        ]
        _run_loop(mock_adb, signals)

        assert mock_adb.force_stop.call_count == 0
        assert mock_adb.launch_app.call_count == 0

    @patch("monkey_collector.pipeline.collection_loop.nudge_static_screen")
    def test_reinit_exhaustion_ends_session(self, mock_nudge, mock_adb):
        # A permanently dead session (never emits xml) escalates once every 3
        # timeouts. With no forward progress the reinit budget is never
        # forgiven, so after MAX_TIMEOUT_REINITS relaunches the next escalation
        # ends the session BEFORE relaunching — force_stop tops out at the
        # budget, not one more.
        signals = [None] * (3 * (MAX_TIMEOUT_REINITS + 1))
        _run_loop(mock_adb, signals, max_steps=500)

        assert mock_adb.force_stop.call_count == MAX_TIMEOUT_REINITS
