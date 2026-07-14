"""Tests for interactable-element detection on encoded XML."""

from monkey_collector.pipeline.screen_matching.ui_attributes import (
    extract_interactable_indexes,
)

# Encoded-XML schema: button/input tags, aria-label/alt/text/type/value attrs.
SCREEN = (
    '<div index="0">'
    '<button index="1" aria-label="Add">Add</button>'
    '<div index="2"><button index="3" aria-label="Search">Search</button></div>'
    '<input index="4" type="text" value="hello" />'
    '<input index="5" type="checkbox" aria-label="Wifi" />'
    '<p index="6">Just text</p>'
    "</div>"
)


def test_extract_interactable_indexes_only_button_input():
    # buttons 1,3 + inputs 4,5 — not the div/p containers.
    assert extract_interactable_indexes(SCREEN) == [1, 3, 4, 5]


def test_extract_interactable_indexes_root_interactable_counted():
    # A one-control screen can collapse to a bare <button> root; root-inclusive
    # iteration must still count it (else it would be mis-flagged as empty).
    assert extract_interactable_indexes('<button index="0">Go</button>') == [0]


def test_extract_interactable_indexes_malformed_xml_is_empty():
    assert extract_interactable_indexes("<not valid!!!") == []


def test_extract_interactable_indexes_no_interactables_is_empty():
    assert extract_interactable_indexes('<div index="0"><p index="1">hi</p></div>') == []
