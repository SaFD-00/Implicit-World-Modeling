"""Tests for the single-call element extractor."""

import json

from monkey_collector.llm.element_extractor import (
    ElementExtractor,
    ExtractedElement,
    _coerce_index_list,
)

SCREEN = (
    '<div index="0">'
    '<button index="1" aria-label="Add">Add</button>'
    '<button index="2" aria-label="Search">Search</button>'
    '<button index="3" aria-label="Row1">Row 1</button>'
    '<button index="4" aria-label="Row2">Row 2</button>'
    "</div>"
)


class FakeClient:
    """Records the last chat() kwargs and returns a canned response."""

    def __init__(self, response: str):
        self.response = response
        self.last_kwargs = None

    def chat(self, messages, **kwargs):
        self.last_kwargs = {"messages": messages, **kwargs}
        return self.response


def test_coerce_index_list_dedup_and_negatives():
    assert _coerce_index_list([3, "5", -1, 3, "z", 2]) == [3, 5, 2]
    assert _coerce_index_list(None) == []
    assert _coerce_index_list(7) == [7]


def test_extract_parses_both_index_lists():
    resp = json.dumps(
        {
            "elements": [
                {
                    "name": "open_row",
                    "description": "open a row",
                    "parameters": {"which": "?"},
                    "element_index": [3, 4],
                    "key_element_index": [3],
                }
            ]
        }
    )
    client = FakeClient(resp)
    out = ElementExtractor(client).extract(SCREEN)
    assert len(out) == 1
    el = out[0]
    assert el.name == "open_row"
    assert el.element_index == [3, 4]
    assert el.key_element_index == [3]
    # description/parameters survive extraction (they ride through to elements.json)
    assert el.description == "open a row"
    assert el.parameters == {"which": "?"}
    # cost attribution + json mode requested
    assert client.last_kwargs["agent"] == "element_extractor"
    assert client.last_kwargs["response_format"] == {"type": "json_object"}


def test_key_falls_back_to_lowest_on_screen_member():
    resp = json.dumps(
        {"elements": [{"name": "open_row", "element_index": [4, 3], "key_element_index": []}]}
    )
    out = ElementExtractor(FakeClient(resp)).extract(SCREEN)
    assert out[0].key_element_index == [3]  # min on-screen family member


def test_key_strays_outside_family_are_dropped():
    resp = json.dumps(
        {"elements": [{"name": "x", "element_index": [3], "key_element_index": [3, 99]}]}
    )
    out = ElementExtractor(FakeClient(resp)).extract(SCREEN)
    assert out[0].key_element_index == [3]


def test_known_elements_excluded():
    resp = json.dumps(
        {
            "elements": [
                {"name": "open_add", "element_index": [1], "key_element_index": [1]},
                {"name": "open_new", "element_index": [2], "key_element_index": [2]},
            ]
        }
    )
    known = [ExtractedElement(name="open_add", element_index=[1], key_element_index=[1])]
    out = ElementExtractor(FakeClient(resp)).extract(SCREEN, known_elements=known)
    assert [e.name for e in out] == ["open_new"]


def test_code_fenced_json_tolerated():
    resp = '```json\n{"elements": [{"name": "x", "element_index": [1], "key_element_index": [1]}]}\n```'
    out = ElementExtractor(FakeClient(resp)).extract(SCREEN)
    assert [e.name for e in out] == ["x"]


def test_malformed_json_returns_empty():
    out = ElementExtractor(FakeClient("not json at all")).extract(SCREEN)
    assert out == []


def test_chat_exception_returns_empty():
    class Boom:
        def chat(self, *a, **k):
            raise RuntimeError("boom")

    assert ElementExtractor(Boom()).extract(SCREEN) == []


def test_empty_encoded_xml_returns_empty():
    assert ElementExtractor(FakeClient("{}")).extract("") == []
