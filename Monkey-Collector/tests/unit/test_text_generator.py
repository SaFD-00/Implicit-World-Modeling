"""Tests for monkey_collector.text_generator — text generation strategies."""

import random
from unittest.mock import MagicMock

import pytest

from monkey_collector.pipeline.text_generator import (
    SAMPLE_TEXTS,
    LLMTextGenerator,
    RandomTextGenerator,
    create_text_generator,
)
from tests.conftest import make_element


@pytest.fixture
def dummy_element():
    return make_element(
        resource_id="com.test:id/search_input",
        class_name="android.widget.EditText",
        content_desc="Search field",
        text="",
    )


DUMMY_XML = '<hierarchy><node class="android.widget.EditText" bounds="[0,0][100,100]" /></hierarchy>'


def _mock_client(text="Generated text", model="qwen/qwen3.7-plus"):
    """A stand-in LLMClient whose chat() returns *text*."""
    client = MagicMock()
    client.model = model
    client.chat.return_value = text
    return client


class TestRandomTextGenerator:
    def test_returns_from_samples(self, dummy_element):
        rng = random.Random(42)
        gen = RandomTextGenerator(rng)
        for _ in range(20):
            result = gen.generate(dummy_element, DUMMY_XML)
            assert result in SAMPLE_TEXTS

    def test_deterministic_with_seed(self, dummy_element):
        results_a = [
            RandomTextGenerator(random.Random(42)).generate(dummy_element, "")
            for _ in range(5)
        ]
        results_b = [
            RandomTextGenerator(random.Random(42)).generate(dummy_element, "")
            for _ in range(5)
        ]
        assert results_a == results_b

    def test_custom_samples(self, dummy_element):
        custom = ["alpha", "beta", "gamma"]
        gen = RandomTextGenerator(random.Random(0), sample_texts=custom)
        for _ in range(20):
            assert gen.generate(dummy_element, "") in custom


class TestLLMTextGenerator:
    def test_success(self, dummy_element):
        gen = LLMTextGenerator(_mock_client("Pizza recipe"))
        result = gen.generate(dummy_element, DUMMY_XML)
        assert result == "Pizza recipe"

    def test_empty_response_fallback(self, dummy_element):
        gen = LLMTextGenerator(_mock_client(""), rng=random.Random(42))
        result = gen.generate(dummy_element, DUMMY_XML)
        assert result in SAMPLE_TEXTS

    def test_none_response_fallback(self, dummy_element):
        gen = LLMTextGenerator(_mock_client(None), rng=random.Random(42))
        result = gen.generate(dummy_element, DUMMY_XML)
        assert result in SAMPLE_TEXTS

    def test_api_error_fallback(self, dummy_element):
        client = _mock_client()
        client.chat.side_effect = Exception("API down")
        gen = LLMTextGenerator(client, rng=random.Random(42))
        result = gen.generate(dummy_element, DUMMY_XML)
        assert result in SAMPLE_TEXTS

    def test_strips_quotes(self, dummy_element):
        gen = LLMTextGenerator(_mock_client('"Hello World"'))
        result = gen.generate(dummy_element, DUMMY_XML)
        assert result == "Hello World"

    def test_empty_raw_xml(self, dummy_element):
        client = _mock_client("Search query")
        gen = LLMTextGenerator(client)
        result = gen.generate(dummy_element, "")
        assert result == "Search query"
        client.chat.assert_called_once()

    def test_sends_system_and_user_messages(self, dummy_element):
        client = _mock_client("x")
        gen = LLMTextGenerator(client)
        gen.generate(dummy_element, DUMMY_XML)

        args, kwargs = client.chat.call_args
        messages = args[0]
        assert isinstance(messages, list)
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert kwargs.get("agent") == "text_generator"
        assert kwargs.get("max_tokens") == 50


class TestCreateTextGenerator:
    def test_random_mode(self):
        gen = create_text_generator("random")
        assert isinstance(gen, RandomTextGenerator)

    def test_api_mode_with_client(self):
        gen = create_text_generator("api", llm_client=_mock_client())
        assert isinstance(gen, LLMTextGenerator)

    def test_api_mode_no_client_falls_back(self):
        gen = create_text_generator("api", llm_client=None)
        assert isinstance(gen, RandomTextGenerator)

    def test_random_mode_ignores_client(self):
        gen = create_text_generator("random", llm_client=_mock_client())
        assert isinstance(gen, RandomTextGenerator)
