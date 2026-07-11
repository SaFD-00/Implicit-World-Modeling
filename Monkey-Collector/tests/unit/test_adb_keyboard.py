"""Device-free tests for AdbClient._parse_keyboard_shown.

Fixtures are verbatim excerpts from a live ``adb -s emulator-5556 shell
dumpsys input_method`` run (Pixel6-2, API 33, google_apis image), captured
2026-07-11:

  hidden: "mShowRequested=false mShowExplicitlyRequested=false
            mShowForced=false mInputShown=false"
  shown:  "mShowRequested=true mShowExplicitlyRequested=false
            mShowForced=false mInputShown=true"

Call the staticmethod directly — AdbClient.__init__ resolves an emulator
serial and must never run in a unit test.
"""

from monkey_collector.adb import AdbClient

_parse = AdbClient._parse_keyboard_shown

LIVE_HIDDEN = (
    "  mShowRequested=false mShowExplicitlyRequested=false mShowForced=false "
    "mInputShown=false"
)

LIVE_SHOWN = (
    "  mShowRequested=true mShowExplicitlyRequested=false mShowForced=false "
    "mInputShown=true"
)


def test_live_hidden_fixture():
    assert _parse(LIVE_HIDDEN) is False


def test_live_shown_fixture():
    assert _parse(LIVE_SHOWN) is True


def test_alternate_field_name_shown():
    # Some OEM/API variants report isInputViewShown instead of mInputShown.
    assert _parse("isInputViewShown=true") is True


def test_alternate_field_name_hidden():
    assert _parse("isInputViewShown=false") is False


def test_no_match_defaults_to_hidden():
    assert _parse("") is False
    assert _parse("garbage output with no matching field") is False
