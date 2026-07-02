"""Tests for the encoded-XML → element-line serializer and set criteria."""

from monkey_collector.pipeline.screen_matching.element_lines import (
    element_diff_count,
    element_jaccard,
    serialize_element_lines,
)


def test_drops_index_and_bounds():
    # Same screen, different index/bounds on the button → identical line set,
    # so two dumps of one screen compare equal (the whole point of dropping them).
    a = '<div index="0"><button index="1" aria-label="Add">Add</button></div>'
    b = '<div index="9"><button index="7" bounds="[0,0][1,1]" aria-label="Add">Add</button></div>'
    assert serialize_element_lines(a) == serialize_element_lines(b)
    assert serialize_element_lines(a) == ['<button aria-label="Add">Add</button>']


def test_non_leaf_container_excluded_scroll_container_included():
    xml = (
        '<div index="0">'
        '<div index="1" data-scroll="true"><button index="2" aria-label="X">X</button></div>'
        "</div>"
    )
    lines = serialize_element_lines(xml)
    # root div (non-leaf, no scroll) excluded; scroll div included though non-leaf;
    # button leaf included.
    assert lines == ['<div data-scroll="true"></div>', '<button aria-label="X">X</button>']


def test_attribute_fixed_order():
    # Attributes given out of canonical order → emitted in _ATTR_ORDER
    # (aria-label, alt, type, value, ...), so the line is a stable string.
    xml = '<input index="0" value="v" type="text" aria-label="L"/>'
    assert serialize_element_lines(xml) == ['<input aria-label="L" type="text" value="v"></input>']


def test_text_newlines_stripped():
    xml = '<p index="0" aria-label="a">line1\nline2</p>'
    assert serialize_element_lines(xml) == ['<p aria-label="a">line1line2</p>']


def test_duplicates_preserved_in_document_order():
    xml = (
        '<div index="0">'
        '<button index="1" aria-label="A">A</button>'
        '<button index="2" aria-label="A">A</button>'
        "</div>"
    )
    lines = serialize_element_lines(xml)
    assert lines == ['<button aria-label="A">A</button>'] * 2  # dups kept for BM25 tf


def test_parse_failure_returns_empty():
    assert serialize_element_lines("not xml <<<") == []
    assert serialize_element_lines("") == []


def test_element_diff_count():
    assert element_diff_count({"x", "y"}, {"x", "y", "z"}) == 1
    assert element_diff_count({"a"}, {"b"}) == 2
    assert element_diff_count(set(), set()) == 0


def test_element_jaccard():
    assert element_jaccard({"x", "y"}, {"x", "y", "z"}) == 2 / 3
    assert element_jaccard({"a"}, {"b"}) == 0.0
    assert element_jaccard(set(), set()) == 0.0  # empty union → 0, no ZeroDivision
