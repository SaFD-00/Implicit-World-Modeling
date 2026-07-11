"""Tests for back-exit page learning (P1-3).

A ``press_back`` that drifts the app specifically to the launcher marks the
page it backed from (``CollectionState.back_exit_page_ids``); Back is then
suppressed on that page. tap-driven launcher trips and non-launcher drifts
(gms/store) are never marked (D4). Also covers ``safe_press_back``'s bool
return, which the second marking route depends on.
"""

from unittest.mock import MagicMock, patch

from monkey_collector.domain.actions import PressBack
from monkey_collector.pipeline.collection_loop import (
    CollectionState,
    _back_would_exit,
    run_collection_loop,
)
from tests.fixtures.xml_samples import COMPLEX_XML, SIMPLE_XML
from tests.integration.test_collector import _make_collector, _make_xml_signal

LAUNCHER = "com.google.android.apps.nexuslauncher"


def _run(mock_adb, signals, select_return=None, max_steps=100):
    collector, explorer, server, writer = _make_collector(
        mock_adb, signals, max_steps=max_steps
    )
    if select_return is not None:
        explorer.select_action.return_value = select_return
    state = CollectionState(step=0, max_step=max_steps)
    with patch("monkey_collector.pipeline.collection_loop.time.sleep"):
        run_collection_loop(collector, state, "com.test.app")
    return state, collector, explorer


class TestBackWouldExit:
    def test_root_page_would_exit(self):
        state = CollectionState()
        state.current_page_id = 0
        state.root_page_id = 0
        assert _back_would_exit(state) is True

    def test_non_root_unmarked_would_not_exit(self):
        state = CollectionState()
        state.current_page_id = 5
        state.root_page_id = 0
        assert _back_would_exit(state) is False

    def test_marked_page_would_exit(self):
        state = CollectionState()
        state.current_page_id = 5
        state.root_page_id = 0
        state.back_exit_page_ids.add(5)
        assert _back_would_exit(state) is True


class TestExternalDriftMarking:
    def test_back_exit_to_launcher_marks_page(self, mock_adb):
        # An in-app frame (last action = Back), then an external_app drift to
        # the launcher → the current page is marked as back-exiting.
        signals = [
            _make_xml_signal(),
            ("external_app", None, {"detected_package": LAUNCHER}),
            ("finish", None, None),
        ]
        state, *_ = _run(mock_adb, signals, select_return=PressBack())

        assert state.current_page_id in state.back_exit_page_ids
        assert len(state.back_exit_page_ids) == 1

    def test_non_launcher_external_does_not_mark(self, mock_adb):
        # A drift to a gms surface (system, not launcher) after a Back must NOT
        # mark the page — that is a bounce, not a true back-exit to home.
        signals = [
            _make_xml_signal(),
            ("external_app", None, {"detected_package": "com.google.android.gms"}),
            ("finish", None, None),
        ]
        state, *_ = _run(mock_adb, signals, select_return=PressBack())

        assert state.back_exit_page_ids == set()

    def test_tap_exit_does_not_mark(self, mock_adb):
        # Last action was a tap (default select_action returns Tap). Even a
        # launcher drift must not mark the page: back-exit learning is
        # press_back-only (D4).
        signals = [
            _make_xml_signal(),
            ("external_app", None, {"detected_package": LAUNCHER}),
            ("finish", None, None),
        ]
        state, *_ = _run(mock_adb, signals)  # default Tap

        assert state.back_exit_page_ids == set()


class TestMarkedPageSuppressesBack:
    def test_marked_page_suppresses_back(self, mock_adb):
        # Root (A) → a distinct non-root page (B); B back-exits to launcher and
        # is marked; returning to B and exhausting no-change must relaunch
        # (force_stop) instead of pressing Back.
        signals = [
            _make_xml_signal(xml=SIMPLE_XML, is_first=True),                       # root A
            _make_xml_signal(xml=COMPLEX_XML, activity="com.test.app/.SettingsActivity"),  # page B
            ("external_app", None, {"detected_package": LAUNCHER}),                # mark B
            _make_xml_signal(xml=COMPLEX_XML, activity="com.test.app/.SettingsActivity"),  # back on B
            ("no_change", None, None),
            ("no_change", None, None),
            ("no_change", None, None),  # retries=3 → suppressed back → relaunch
            ("finish", None, None),
        ]
        state, collector, explorer = _run(
            mock_adb, signals, select_return=PressBack()
        )

        # B was learned as a back-exit page, and exhaustion relaunched rather
        # than pressing back: the only press_back source in this harness is
        # safe_press_back (execute_action is mocked), so its absence is
        # load-bearing.
        assert len(state.back_exit_page_ids) == 1
        assert mock_adb.force_stop.call_count == 1
        mock_adb.press_back.assert_not_called()


class TestSafePressBackReturn:
    def test_safe_press_back_returns_exit_flag(self, mock_adb):
        from monkey_collector.pipeline.recovery import safe_press_back

        explorer = MagicMock()
        with patch("time.sleep"):
            explorer.has_left_app.return_value = True
            assert safe_press_back(mock_adb, explorer, "com.test.app") is True
            explorer.return_to_app.assert_called_once_with("com.test.app")

            explorer.has_left_app.return_value = False
            assert safe_press_back(mock_adb, explorer, "com.test.app") is False
