"""Tests for ActionMapper: semantic action vocabulary → domain Action."""

from monkey_collector.domain.actions import InputText, LongPress, Swipe, Tap
from monkey_collector.pipeline.exploration.action_mapper import ActionMapper
from monkey_collector.pipeline.exploration.state import (
    LONG_TOUCH,
    SCROLL,
    SELECT,
    SET_TEXT,
    TOUCH,
    SemanticElement,
    SemanticState,
)

ACTIVITY = "com.test.app/.MainActivity"
PACKAGE = "com.test.app"


def _element(action: str, index: int = 3, center=(100, 200)) -> SemanticElement:
    return SemanticElement(
        index=index,
        signature=f"x:{action}",
        allowed_actions=(action,),
        center=center,
        desc="<x></x>",
    )


def test_touch_maps_to_tap():
    action = ActionMapper().to_domain_action(TOUCH, _element(TOUCH), "")
    assert isinstance(action, Tap)
    assert (action.x, action.y) == (100, 200)
    assert action.element_index == 3


def test_select_maps_to_tap():
    action = ActionMapper().to_domain_action(SELECT, _element(SELECT), "")
    assert isinstance(action, Tap)


def test_long_touch_maps_to_long_press():
    action = ActionMapper().to_domain_action(LONG_TOUCH, _element(LONG_TOUCH), "")
    assert isinstance(action, LongPress)
    assert (action.x, action.y) == (100, 200)


def test_scroll_maps_to_vertical_swipe():
    action = ActionMapper().to_domain_action(SCROLL, _element(SCROLL, index=-1), "")
    assert isinstance(action, Swipe)
    assert action.x1 == action.x2 == 100  # vertical
    assert action.y1 > action.y2  # swipes upward
    assert action.element_index == -1  # negative index clamps to -1


def test_set_text_without_generator_is_empty():
    action = ActionMapper().to_domain_action(SET_TEXT, _element(SET_TEXT), "")
    assert isinstance(action, InputText)
    assert action.text == ""


def test_set_text_uses_generator_with_matched_editable(simple_xml):
    """set_text resolves the editable UIElement by center and feeds the generator."""

    class StubGenerator:
        def __init__(self):
            self.seen = None

        def generate(self, element, raw_xml):
            self.seen = element
            return "typed value"

    state = SemanticState.from_screen(simple_xml, ACTIVITY, PACKAGE)
    field = state.find_by_signature("input:text:Search field")
    assert field is not None

    gen = StubGenerator()
    action = ActionMapper(text_generator=gen).to_domain_action(SET_TEXT, field, simple_xml)
    assert isinstance(action, InputText)
    assert action.text == "typed value"
    # the generator received the matching editable element (Search field)
    assert gen.seen is not None
    assert gen.seen.content_desc == "Search field"
