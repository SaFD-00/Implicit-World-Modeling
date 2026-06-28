"""Tests for monkey_collector.llm.screen_grouper — screen semantic grouping."""

import json
from unittest.mock import MagicMock

from monkey_collector.llm.screen_grouper import ScreenGrouper, create_screen_grouper
from tests.fixtures.xml_samples import COMPLEX_XML, SIMPLE_XML

VALID_JSON = json.dumps(
    {
        "page_description": "Search screen",
        "elements_description": "search button, search input, list, fab",
        "groups": [
            {"indices": [2, 3], "function": "search controls"},
            {"indices": [5], "function": "single list item"},  # singleton → dropped
        ],
    }
)


def _mock_client(response, model="qwen/qwen3.7-plus"):
    client = MagicMock()
    client.model = model
    client.chat.return_value = response
    return client


class TestGroup:
    def test_parses_groups(self):
        grouper = ScreenGrouper(_mock_client(VALID_JSON))
        result = grouper.group(SIMPLE_XML)
        assert result["page_description"] == "Search screen"
        assert result["model"] == "qwen/qwen3.7-plus"
        # singleton group dropped (MIN_GROUP_SIZE == 2)
        assert len(result["groups"]) == 1
        assert result["groups"][0]["indices"] == [2, 3]
        assert result["groups"][0]["function"] == "search controls"

    def test_uses_json_response_format(self):
        client = _mock_client(VALID_JSON)
        ScreenGrouper(client).group(SIMPLE_XML)
        kwargs = client.chat.call_args.kwargs
        assert kwargs["response_format"] == {"type": "json_object"}
        assert kwargs["agent"] == "screen_grouper"

    def test_code_fenced_json(self):
        fenced = "```json\n" + VALID_JSON + "\n```"
        grouper = ScreenGrouper(_mock_client(fenced))
        result = grouper.group(SIMPLE_XML)
        assert len(result["groups"]) == 1

    def test_malformed_json_returns_empty(self):
        grouper = ScreenGrouper(_mock_client("totally not json"))
        result = grouper.group(SIMPLE_XML)
        assert result["groups"] == []
        assert result["model"] == "qwen/qwen3.7-plus"

    def test_chat_error_returns_empty(self):
        client = _mock_client(VALID_JSON)
        client.chat.side_effect = Exception("boom")
        result = ScreenGrouper(client).group(SIMPLE_XML)
        assert result["groups"] == []

    def test_empty_xml_no_call(self):
        client = _mock_client(VALID_JSON)
        result = ScreenGrouper(client).group("")
        assert result["groups"] == []
        client.chat.assert_not_called()

    def test_invalid_xml_no_call(self):
        client = _mock_client(VALID_JSON)
        result = ScreenGrouper(client).group("<not valid!!!")
        assert result["groups"] == []
        client.chat.assert_not_called()

    def test_structure_cache_reuses_result(self):
        client = _mock_client(VALID_JSON)
        grouper = ScreenGrouper(client)
        grouper.group(SIMPLE_XML)
        grouper.group(SIMPLE_XML)  # identical structure → cache hit
        assert client.chat.call_count == 1

    def test_different_structure_calls_again(self):
        client = _mock_client(VALID_JSON)
        grouper = ScreenGrouper(client)
        grouper.group(SIMPLE_XML)
        grouper.group(COMPLEX_XML)
        assert client.chat.call_count == 2

    def test_non_int_indices_filtered(self):
        payload = json.dumps(
            {
                "page_description": "p",
                "elements_description": "e",
                "groups": [{"indices": [1, "x", 2], "function": "f"}],
            }
        )
        result = ScreenGrouper(_mock_client(payload)).group(SIMPLE_XML)
        assert result["groups"][0]["indices"] == [1, 2]


class TestCreateScreenGrouper:
    def test_none_when_disabled(self):
        assert create_screen_grouper(_mock_client(VALID_JSON), enabled=False) is None

    def test_none_when_no_client(self):
        assert create_screen_grouper(None, enabled=True) is None

    def test_builds_when_enabled(self):
        grouper = create_screen_grouper(_mock_client(VALID_JSON), enabled=True)
        assert isinstance(grouper, ScreenGrouper)
