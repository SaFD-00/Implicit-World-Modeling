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
    MAX_KEYBOARD_ESCAPES,
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


KEYBOARD = "com.test.app/com.android.inputmethodservice.SoftInputWindow"


class TestKeyboardBackExit:
    """A keyboard on a back-exit page is dismissed via ESC, not a Back that
    would drop the app to the launcher (P1-3 blind spot). The first dismissal
    that confirms a launcher exit marks the page (D-B1); later keyboards there
    use ESC, capped by MAX_KEYBOARD_ESCAPES."""

    def test_keyboard_back_exit_marks_page(self, mock_adb):
        # Non-root page B, then a keyboard whose Back leaves the app to the
        # launcher (has_left_app=True) → B is marked as back-exiting.
        signals = [
            _make_xml_signal(xml=SIMPLE_XML, is_first=True),  # root A
            _make_xml_signal(xml=COMPLEX_XML, activity="com.test.app/.SettingsActivity"),  # B
            _make_xml_signal(activity=KEYBOARD),              # keyboard on B
            ("finish", None, None),
        ]
        collector, explorer, server, writer = _make_collector(mock_adb, signals)
        explorer.has_left_app.return_value = True
        state = CollectionState(step=0, max_step=100)
        with patch("monkey_collector.pipeline.collection_loop.time.sleep"), \
                patch("time.sleep"):
            run_collection_loop(collector, state, "com.test.app")

        assert state.current_page_id in state.back_exit_page_ids
        mock_adb.press_back.assert_called()

    def test_keyboard_dismiss_in_app_does_not_mark(self, mock_adb):
        # Keyboard dismissed by Back that stays in-app (has_left_app=False) → no
        # mark (D4 preserved), but the dismissal still happens via Back.
        signals = [
            _make_xml_signal(xml=SIMPLE_XML, is_first=True),  # root A
            _make_xml_signal(xml=COMPLEX_XML, activity="com.test.app/.SettingsActivity"),  # B
            _make_xml_signal(activity=KEYBOARD),
            ("finish", None, None),
        ]
        collector, explorer, server, writer = _make_collector(mock_adb, signals)
        explorer.has_left_app.return_value = False
        state = CollectionState(step=0, max_step=100)
        with patch("monkey_collector.pipeline.collection_loop.time.sleep"), \
                patch("time.sleep"):
            run_collection_loop(collector, state, "com.test.app")

        assert state.back_exit_page_ids == set()
        mock_adb.press_back.assert_called()

    def test_marked_page_keyboard_uses_escape(self, mock_adb):
        # First keyboard-back marks B (has_left_app True); the second keyboard on
        # the now-marked B is dismissed via ESC (hide_keyboard), no extra Back.
        signals = [
            _make_xml_signal(xml=SIMPLE_XML, is_first=True),  # root A
            _make_xml_signal(xml=COMPLEX_XML, activity="com.test.app/.SettingsActivity"),  # B
            _make_xml_signal(activity=KEYBOARD),  # keyboard1 → mark B, Back
            _make_xml_signal(activity=KEYBOARD),  # keyboard2 → ESC
            ("finish", None, None),
        ]
        collector, explorer, server, writer = _make_collector(mock_adb, signals)
        explorer.has_left_app.side_effect = [True, False]
        state = CollectionState(step=0, max_step=100)
        with patch("monkey_collector.pipeline.collection_loop.time.sleep"), \
                patch("time.sleep"):
            run_collection_loop(collector, state, "com.test.app")

        mock_adb.hide_keyboard.assert_called_once()
        assert mock_adb.press_back.call_count == 1  # only the marking Back

    def test_escape_exhaustion_falls_back_to_back(self, mock_adb):
        # Three keyboards on the root (already a back-exit page): ESC twice, then
        # the exhausted third dismissal falls back to a real Back.
        signals = [
            _make_xml_signal(xml=SIMPLE_XML, is_first=True),  # root A
            _make_xml_signal(activity=KEYBOARD),  # ESC 1
            _make_xml_signal(activity=KEYBOARD),  # ESC 2
            _make_xml_signal(activity=KEYBOARD),  # exhausted → Back
            ("finish", None, None),
        ]
        collector, explorer, server, writer = _make_collector(mock_adb, signals)
        explorer.has_left_app.return_value = False
        state = CollectionState(step=0, max_step=100)
        with patch("monkey_collector.pipeline.collection_loop.time.sleep"), \
                patch("time.sleep"):
            run_collection_loop(collector, state, "com.test.app")

        assert mock_adb.hide_keyboard.call_count == 2
        assert mock_adb.press_back.call_count == 1

    def test_esc_verify_marker_logged_once_per_attempt(self, mock_adb):
        # W1: each ESC dismissal logs a `keyboard ESC verify:` marker exactly
        # once, with the attempt count from the (unchanged) escalation
        # counter. Measurement only — must not alter ESC/Back call counts.
        import io

        from loguru import logger

        signals = [
            _make_xml_signal(xml=SIMPLE_XML, is_first=True),  # root A
            _make_xml_signal(activity=KEYBOARD),  # ESC 1
            _make_xml_signal(activity=KEYBOARD),  # ESC 2
            _make_xml_signal(activity=KEYBOARD),  # exhausted -> Back
            ("finish", None, None),
        ]
        collector, explorer, server, writer = _make_collector(mock_adb, signals)
        explorer.has_left_app.return_value = False
        mock_adb.is_keyboard_shown.return_value = True
        state = CollectionState(step=0, max_step=100)

        sink = io.StringIO()
        logger.enable("monkey_collector")
        handler_id = logger.add(sink, format="{message}", level="INFO")
        try:
            with patch("monkey_collector.pipeline.collection_loop.time.sleep"), \
                    patch("time.sleep"):
                run_collection_loop(collector, state, "com.test.app")
        finally:
            logger.remove(handler_id)
            logger.disable("monkey_collector")

        output = sink.getvalue()
        markers = [line for line in output.splitlines() if "keyboard ESC verify:" in line]
        assert len(markers) == 2
        assert markers[0] == (
            f"keyboard ESC verify: still_shown=True (attempt 1/{MAX_KEYBOARD_ESCAPES})"
        )
        assert markers[1] == (
            f"keyboard ESC verify: still_shown=True (attempt 2/{MAX_KEYBOARD_ESCAPES})"
        )
        # Escalation semantics unchanged: ESC twice, exhausted 3rd falls to Back.
        assert mock_adb.hide_keyboard.call_count == 2
        assert mock_adb.press_back.call_count == 1

    def test_keyboard_frames_do_not_advance_step(self, mock_adb):
        # Keyboard overlays are not pages: step must not advance across them.
        signals = [
            _make_xml_signal(activity=KEYBOARD),
            _make_xml_signal(activity=KEYBOARD),
            ("finish", None, None),
        ]
        collector, explorer, server, writer = _make_collector(mock_adb, signals)
        explorer.has_left_app.return_value = False
        state = CollectionState(step=0, max_step=100)
        with patch("monkey_collector.pipeline.collection_loop.time.sleep"), \
                patch("time.sleep"):
            run_collection_loop(collector, state, "com.test.app")

        assert state.step == 0


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
