"""Tests for SemanticState: identity hashing, element extraction, signatures."""

from monkey_collector.pipeline.exploration.state import (
    SCROLL,
    SELECT,
    SET_TEXT,
    TOUCH,
    SemanticState,
)

ACTIVITY = "com.test.app/.MainActivity"
PACKAGE = "com.test.app"


def _state(xml: str, activity: str = ACTIVITY) -> SemanticState:
    return SemanticState.from_screen(xml, activity, PACKAGE)


# ── identity hashing ──


def test_same_xml_yields_same_identity(simple_xml):
    a = _state(simple_xml)
    b = _state(simple_xml)
    assert a.state_str == b.state_str
    assert a.structure_str == b.structure_str


def test_text_change_keeps_structure_but_changes_state(simple_xml):
    original = _state(simple_xml)
    changed = _state(simple_xml.replace("Item title", "A different label"))
    assert original.structure_str == changed.structure_str
    assert original.state_str != changed.state_str


def test_different_activity_changes_identity(simple_xml):
    main = _state(simple_xml, "com.test.app/.MainActivity")
    other = _state(simple_xml, "com.test.app/.OtherActivity")
    assert main.structure_str != other.structure_str
    assert main.state_str != other.state_str


# ── in-app gate ──


def test_is_in_app_matches_package(simple_xml):
    assert _state(simple_xml, "com.test.app/.MainActivity").is_in_app()
    assert not _state(simple_xml, "com.other.app/.Main").is_in_app()


# ── element extraction & action mapping ──


def test_minimal_xml_has_no_actionable_elements(minimal_xml):
    assert _state(minimal_xml).elements == ()


def test_button_supports_touch_and_long_touch(simple_xml):
    button = _state(simple_xml).find_by_signature("button::Search")
    assert button is not None
    assert button.allowed_actions == (TOUCH, "long_touch")


def test_edit_text_supports_set_text(simple_xml):
    field = _state(simple_xml).find_by_signature("input:text:Search field")
    assert field is not None
    assert field.allowed_actions == (SET_TEXT,)


def test_checkbox_supports_select(complex_xml):
    toggle = _state(complex_xml, "com.test.app/.SettingsActivity").find_by_signature(
        "input:checkbox:Dark mode toggle"
    )
    assert toggle is not None
    assert toggle.allowed_actions == (SELECT,)


def test_scroll_container_recovered_with_negative_index(simple_xml):
    state = _state(simple_xml)
    scrolls = [e for e in state.elements if SCROLL in e.allowed_actions]
    assert len(scrolls) == 1
    assert scrolls[0].index < 0  # outside the encoded grouping index space


def test_encoded_elements_use_nonnegative_indices(simple_xml):
    state = _state(simple_xml)
    encoded = [e for e in state.elements if SCROLL not in e.allowed_actions]
    assert encoded  # there are tappable/text elements
    assert all(e.index >= 0 for e in encoded)


# ── signature stability ──


def test_signature_is_coordinate_free(simple_xml):
    """Moving an element's bounds must not change its signature."""
    original = _state(simple_xml)
    moved = _state(simple_xml.replace("[900,24][1056,144]", "[100,500][256,620]"))
    orig_btn = original.find_by_signature("button::Search")
    moved_btn = moved.find_by_signature("button::Search")
    assert orig_btn is not None and moved_btn is not None
    assert orig_btn.signature == moved_btn.signature
    assert orig_btn.center != moved_btn.center  # coordinates did move


def test_actionable_elements_excludes_plain_text(simple_xml):
    state = _state(simple_xml)
    # "Item title" is a <p> with no actions and must not appear as actionable.
    assert all("Item title" not in e.desc for e in state.actionable_elements())
