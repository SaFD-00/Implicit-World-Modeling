"""Tests for element fingerprinting + structural matching primitives."""

import xml.etree.ElementTree as ET

from monkey_collector.pipeline.screen_matching.ui_attributes import (
    UIAttributes,
    extract_interactable_indexes,
    find_matching_node,
    get_ui_key_attrib,
    text_blind_requirements,
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


def _tree(xml=SCREEN):
    return ET.fromstring(xml)


def test_extract_interactable_indexes_only_button_input():
    # buttons 1,3 + inputs 4,5 — not the div/p containers.
    assert extract_interactable_indexes(SCREEN) == [1, 3, 4, 5]


def test_get_ui_key_attrib_self_parent():
    attrs = get_ui_key_attrib(3, SCREEN)
    assert attrs["self"]["tag"] == "button"
    assert attrs["self"]["aria-label"] == "Search"
    assert attrs["self"]["text"] == "Search"
    # parent is the wrapping div
    assert attrs["parent"]["tag"] == "div"


def test_get_ui_key_attrib_missing_index():
    attrs = get_ui_key_attrib(99, SCREEN)
    assert attrs == {"self": {}, "parent": {}, "children": []}


def test_find_matching_node_by_aria_label():
    ui = UIAttributes.from_attrib_dict(get_ui_key_attrib(1, SCREEN))
    matched = [n.attrib["index"] for n in find_matching_node(_tree(), ui.to_dict())]
    assert matched == ["1"]


def test_text_blind_ignores_dynamic_text():
    # Fingerprint a button whose label changes between captures.
    a = '<div index="0"><button index="1" aria-label="Cart">3 items</button></div>'
    b = '<div index="0"><button index="1" aria-label="Cart">5 items</button></div>'
    ui = UIAttributes.from_attrib_dict(get_ui_key_attrib(1, a))
    # Exact (text-sensitive) match fails across the text change ...
    assert not find_matching_node(ET.fromstring(b), ui.to_dict())
    # ... but the text-blind copy matches (aria-label still gates).
    blinded = text_blind_requirements(ui.to_dict())
    assert [n.attrib["index"] for n in find_matching_node(ET.fromstring(b), blinded)] == ["1"]


def test_text_blind_does_not_mutate_stored_fingerprint():
    ui = UIAttributes.from_attrib_dict(get_ui_key_attrib(1, SCREEN))
    before = ui.to_dict()["self"]["text"]
    text_blind_requirements(ui.to_dict())
    assert ui.to_dict()["self"]["text"] == before  # deepcopy, not in-place


def test_all_match_requires_every_anchor():
    # A two-anchor element is supported only if BOTH anchors are present.
    add = UIAttributes.from_attrib_dict(get_ui_key_attrib(1, SCREEN))
    search = UIAttributes.from_attrib_dict(get_ui_key_attrib(3, SCREEN))
    partial = '<div index="0"><button index="1" aria-label="Add">Add</button></div>'
    tree = ET.fromstring(partial)
    found_add = find_matching_node(tree, text_blind_requirements(add.to_dict()))
    found_search = find_matching_node(tree, text_blind_requirements(search.to_dict()))
    assert found_add and not found_search  # only one of two anchors present
