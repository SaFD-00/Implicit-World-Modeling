"""Translate the engine's semantic actions into concrete device actions.

The exploration engine reasons in a small action vocabulary
(``touch``/``select``/``long_touch``/``set_text``/``scroll``) over
:class:`SemanticElement`. ``ActionMapper`` turns one such (action_type, element)
pair into a :class:`~monkey_collector.domain.actions.Action` ready for ADB
execution, delegating input-text generation to the shared ``TextGenerator``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from monkey_collector.domain.actions import (
    Action,
    InputText,
    LongPress,
    Swipe,
    Tap,
)
from monkey_collector.pipeline.exploration.state import (
    LONG_TOUCH,
    SCROLL,
    SELECT,
    SET_TEXT,
    TOUCH,
    SemanticElement,
)
from monkey_collector.xml.ui_tree import UITree

if TYPE_CHECKING:
    from monkey_collector.pipeline.text_generator import TextGenerator

# Vertical swipe distance (px) above/below the element center for a scroll.
_SCROLL_SPAN = 200
_SCROLL_DURATION_MS = 300


class ActionMapper:
    """Maps semantic (action_type, element) pairs to domain actions."""

    def __init__(self, text_generator: TextGenerator | None = None):
        self._text_generator = text_generator

    def to_domain_action(
        self,
        action_type: str,
        element: SemanticElement,
        raw_xml: str,
    ) -> Action:
        """Build the concrete :class:`Action` for *action_type* on *element*."""
        x, y = element.center
        element_index = element.index if element.index >= 0 else -1

        if action_type in (TOUCH, SELECT):
            return Tap(x=x, y=y, element_index=element_index)
        if action_type == LONG_TOUCH:
            return LongPress(x=x, y=y, element_index=element_index)
        if action_type == SET_TEXT:
            text = self._generate_text(element, raw_xml)
            return InputText(text=text, x=x, y=y, element_index=element_index)
        if action_type == SCROLL:
            return Swipe(
                x1=x,
                y1=y + _SCROLL_SPAN,
                x2=x,
                y2=y - _SCROLL_SPAN,
                duration_ms=_SCROLL_DURATION_MS,
                element_index=element_index,
            )
        # Unknown action types fall back to a plain tap on the element.
        return Tap(x=x, y=y, element_index=element_index)

    # -- text generation ------------------------------------------------------

    def _generate_text(self, element: SemanticElement, raw_xml: str) -> str:
        """Generate input text for a set_text action via the shared generator.

        Resolves the matching editable :class:`UIElement` (by center) so the
        generator gets the field's resource id / description as context; on any
        miss it degrades to an empty string rather than failing the action.
        """
        if self._text_generator is None:
            return ""
        ui_element = self._match_editable(element, raw_xml)
        if ui_element is None:
            return ""
        return self._text_generator.generate(ui_element, raw_xml)

    @staticmethod
    def _match_editable(element: SemanticElement, raw_xml: str):
        editables = UITree.from_xml_string(raw_xml).get_editable_elements()
        if not editables:
            return None
        for candidate in editables:
            if candidate.center == element.center:
                return candidate
        return editables[0]
