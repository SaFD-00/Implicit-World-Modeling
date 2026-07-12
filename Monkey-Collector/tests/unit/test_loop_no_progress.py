"""Tests for the no-progress loop guards (R2 volume-not-diversity).

D2 — repeat-action circuit breaker: the same (page_key, action_type,
element_index) executed max_action_repeats times on a page, with no new page
in between, makes the next attempt break out via back/relaunch instead of
executing. The escape does not advance `step` and clears the counters. A new
page forgives (resets) the counters.

D3 — plateau early-stop: after max_steps_without_new_page real-action steps with
no new page the loop clean-stops (state.no_progress_stop). New pages keep
resetting the tally; a non-positive threshold disables the guard.

Both guards default to product values on the Collector; these tests inject
tight thresholds so a short fake signal stream exercises the boundary.
"""

from unittest.mock import patch

from monkey_collector.domain.actions import Tap
from monkey_collector.pipeline.collection_loop import CollectionState, run_collection_loop
from tests.fixtures.xml_samples import COMPLEX_XML, SIMPLE_XML
from tests.integration.test_collector import _make_collector, _make_xml_signal


def _run(collector, max_step=100):
    state = CollectionState(step=0, max_step=max_step)
    with patch("monkey_collector.pipeline.collection_loop.time.sleep"), \
            patch("time.sleep"):
        run_collection_loop(collector, state, "com.test.app")
    return state


def _settings(**kw):
    return _make_xml_signal(xml=COMPLEX_XML, activity="com.test.app/.SettingsActivity", **kw)


def _third(**kw):
    return _make_xml_signal(xml=SIMPLE_XML, activity="com.test.app/.ThirdActivity", **kw)


class TestD2RepeatActionBreaker:
    def test_repeat_escapes_without_advancing_step(self, mock_adb):
        # Root A, then page B repeated. With max_action_repeats=3, B executes the
        # same Tap 3 times (steps advance), and the 4th attempt breaks out via
        # back (B is a non-root, unmarked page) WITHOUT executing or advancing.
        signals = [
            _make_xml_signal(xml=SIMPLE_XML, is_first=True),  # root A → step 1
            _settings(),  # B new → step 2
            _settings(),  # B → step 3
            _settings(),  # B → step 4 (countB now == 3)
            _settings(),  # B → 4th attempt: ESCAPE (no execute, no step)
            ("finish", None, None),
        ]
        collector, explorer, server, writer = _make_collector(mock_adb, signals)
        collector.max_action_repeats = 3
        collector.max_steps_without_new_page = 0  # isolate D2
        explorer.has_left_app.return_value = False

        state = _run(collector)

        # Root(1) + 3 B executes = 4 executed actions; the 5th frame escaped.
        assert explorer.execute_action.call_count == 4
        assert explorer.select_action.call_count == 5  # select runs before the check
        assert state.step == 4  # escape did not advance step
        mock_adb.press_back.assert_called()  # broke out via back (B non-root)
        assert state.action_repeat_counts == {}  # counters cleared on escape

    def test_new_page_forgives_repeat_counter(self, mock_adb):
        # B reaches the repeat threshold, then a NEW page C interposes and clears
        # the counters. Returning to B starts B's count fresh, so no escape fires
        # — proving discovered_new_page resets D2 (acceptance #4).
        signals = [
            _make_xml_signal(xml=SIMPLE_XML, is_first=True),  # root A → step 1
            _settings(),  # B new → step 2, countB=1
            _settings(),  # B → step 3, countB=2
            _settings(),  # B → step 4, countB=3 (at threshold)
            _third(),     # C new → step 5, counters cleared
            _settings(),  # B again → countB restarts at 1, executes, step 6
            ("finish", None, None),
        ]
        collector, explorer, server, writer = _make_collector(mock_adb, signals)
        collector.max_action_repeats = 3
        collector.max_steps_without_new_page = 0  # isolate D2
        explorer.has_left_app.return_value = False

        state = _run(collector)

        # Every frame executed (6 actions); no escape, so no back/relaunch.
        assert explorer.execute_action.call_count == 6
        assert state.step == 6
        mock_adb.press_back.assert_not_called()
        mock_adb.force_stop.assert_not_called()

    def test_disabled_when_threshold_non_positive(self, mock_adb):
        # max_action_repeats=0 disables the breaker: many identical actions on
        # the same page never escape.
        signals = [_make_xml_signal(xml=SIMPLE_XML, is_first=True)]
        signals += [_settings()] * 8
        signals.append(("finish", None, None))
        collector, explorer, server, writer = _make_collector(mock_adb, signals)
        collector.max_action_repeats = 0
        collector.max_steps_without_new_page = 0
        explorer.has_left_app.return_value = False

        state = _run(collector)

        assert explorer.execute_action.call_count == 9  # all executed
        mock_adb.press_back.assert_not_called()
        assert state.no_progress_stop is False


class TestD3PlateauEarlyStop:
    def test_plateau_clean_stops(self, mock_adb):
        # Root A then the same page A forever. With max_steps_without_new_page=3
        # and D2 disabled, the loop clean-stops after 3 real steps with no new
        # page (root step resets the tally to 0 that same step).
        signals = [_make_xml_signal(xml=SIMPLE_XML, is_first=True)]
        signals += [_make_xml_signal(xml=SIMPLE_XML)] * 10  # same page A
        signals.append(("finish", None, None))
        collector, explorer, server, writer = _make_collector(mock_adb, signals)
        collector.max_action_repeats = 0  # isolate D3
        collector.max_steps_without_new_page = 3
        explorer.has_left_app.return_value = False

        state = _run(collector)

        # step1 (A, new, tally→0), then 3 same-page steps (tally 1,2,3) → stop.
        assert state.no_progress_stop is True
        assert state.step == 4
        # Stopped before consuming the finish signal (clean early-stop).
        assert server.get_latest_signal.call_count == 4

    def test_new_pages_keep_resetting_plateau(self, mock_adb):
        # A stream of DISTINCT pages never plateaus: each new page resets the
        # tally, so the session runs to the finish signal.
        signals = [
            _make_xml_signal(xml=SIMPLE_XML, is_first=True, activity="com.test.app/.A0"),
            _make_xml_signal(xml=SIMPLE_XML, activity="com.test.app/.A1"),
            _make_xml_signal(xml=SIMPLE_XML, activity="com.test.app/.A2"),
            _make_xml_signal(xml=SIMPLE_XML, activity="com.test.app/.A3"),
            _make_xml_signal(xml=SIMPLE_XML, activity="com.test.app/.A4"),
            ("finish", None, None),
        ]
        collector, explorer, server, writer = _make_collector(mock_adb, signals)
        collector.max_action_repeats = 0
        collector.max_steps_without_new_page = 3
        explorer.has_left_app.return_value = False

        state = _run(collector)

        assert state.no_progress_stop is False
        assert state.step == 5

    def test_disabled_when_threshold_non_positive(self, mock_adb):
        # max_steps_without_new_page=0 disables the plateau stop: a long run on
        # one page ends only via the finish signal.
        signals = [_make_xml_signal(xml=SIMPLE_XML, is_first=True)]
        signals += [_make_xml_signal(xml=SIMPLE_XML)] * 10
        signals.append(("finish", None, None))
        collector, explorer, server, writer = _make_collector(mock_adb, signals)
        collector.max_action_repeats = 0
        collector.max_steps_without_new_page = 0
        explorer.has_left_app.return_value = False

        state = _run(collector)

        assert state.no_progress_stop is False
        assert state.step == 11  # all 11 xml frames executed, then finish
