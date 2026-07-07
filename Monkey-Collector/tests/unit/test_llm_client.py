"""Tests for monkey_collector.llm.client — shared OpenRouter LLM client."""

from unittest.mock import MagicMock, patch

from monkey_collector.llm.client import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    LLMClient,
    create_llm_client,
)


def _completion(content="hello", prompt_tokens=100, completion_tokens=20):
    completion = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    completion.choices = [choice]
    completion.usage.prompt_tokens = prompt_tokens
    completion.usage.completion_tokens = completion_tokens
    return completion


def _client_with(completion, **kwargs):
    """An LLMClient whose underlying OpenAI SDK returns *completion*."""
    client = LLMClient(api_key="k", model=kwargs.pop("model", "qwen/qwen3.7-plus"), **kwargs)
    oai = MagicMock()
    oai.chat.completions.create.return_value = completion
    client._client = oai
    return client, oai


class TestChat:
    def test_returns_content(self):
        client, _ = _client_with(_completion("hi there"))
        assert client.chat("prompt") == "hi there"

    def test_string_prompt_becomes_user_message(self):
        client, oai = _client_with(_completion())
        client.chat("hi")
        kwargs = oai.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == "qwen/qwen3.7-plus"
        assert kwargs["messages"] == [{"role": "user", "content": "hi"}]

    def test_message_list_passed_through(self):
        client, oai = _client_with(_completion())
        msgs = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
        ]
        client.chat(msgs)
        kwargs = oai.chat.completions.create.call_args.kwargs
        assert kwargs["messages"] == msgs

    def test_optional_kwargs_forwarded(self):
        client, oai = _client_with(_completion())
        client.chat(
            "hi", max_tokens=50, temperature=0.2, response_format={"type": "json_object"}
        )
        kwargs = oai.chat.completions.create.call_args.kwargs
        assert kwargs["max_tokens"] == 50
        assert kwargs["temperature"] == 0.2
        assert kwargs["response_format"] == {"type": "json_object"}

    def test_omits_unset_optionals(self):
        client, oai = _client_with(_completion())
        client.chat("hi")
        kwargs = oai.chat.completions.create.call_args.kwargs
        assert "max_tokens" not in kwargs
        assert "temperature" not in kwargs
        assert "response_format" not in kwargs

    def test_default_timeout_when_unset(self):
        # No per-call timeout → the shared client default flows through.
        client, oai = _client_with(_completion(), timeout=17.0)
        client.chat("hi")
        assert oai.chat.completions.create.call_args.kwargs["timeout"] == 17.0
        # No per-call max_retries → the SDK default path (no with_options).
        oai.with_options.assert_not_called()

    def test_per_call_timeout_overrides_default(self):
        # Route both the direct and with_options paths to the same create mock.
        client, oai = _client_with(_completion(), timeout=30.0)
        oai.with_options.return_value = oai
        client.chat("hi", timeout=60.0, max_retries=1)
        # Per-call timeout goes into the create() kwargs; the shared default
        # (30.0) is not mutated.
        assert oai.chat.completions.create.call_args.kwargs["timeout"] == 60.0
        assert client._timeout == 30.0
        # max_retries is applied via with_options (create() does not accept it).
        assert oai.with_options.call_args.kwargs["max_retries"] == 1

    def test_empty_choices_returns_empty_string(self):
        completion = _completion()
        completion.choices = []
        client, _ = _client_with(completion)
        assert client.chat("hi") == ""

    def test_none_content_returns_empty_string(self):
        client, _ = _client_with(_completion(content=None))
        assert client.chat("hi") == ""


class TestCostRecording:
    def test_records_with_chat_usage_names(self):
        tracker = MagicMock()
        client, _ = _client_with(
            _completion("x", prompt_tokens=100, completion_tokens=20),
            cost_tracker=tracker,
            model="m",
        )
        client.set_step(7)
        client.chat("hi", agent="text_generator")
        tracker.record.assert_called_once_with(
            model="m",
            input_tokens=100,
            output_tokens=20,
            step=7,
            agent="text_generator",
        )

    def test_no_tracker_no_error(self):
        client, _ = _client_with(_completion("x"))
        assert client.chat("hi") == "x"

    def test_no_usage_skips_record(self):
        tracker = MagicMock()
        completion = _completion("x")
        completion.usage = None
        client, _ = _client_with(completion, cost_tracker=tracker)
        client.chat("hi")
        tracker.record.assert_not_called()


class TestLazyClient:
    def test_get_client_builds_openai_with_base_url(self):
        client = LLMClient(api_key="secret", model="m", base_url="http://router")
        with patch("openai.OpenAI") as mock_openai:
            client._get_client()
            mock_openai.assert_called_once_with(base_url="http://router", api_key="secret")

    def test_client_is_lazy(self):
        client = LLMClient(api_key="k")
        assert client._client is None


class TestCreateLLMClient:
    def test_no_key_returns_none(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with patch("dotenv.load_dotenv", return_value=None):
            assert create_llm_client() is None

    def test_with_key_builds_client_with_defaults(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
        monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
        with patch("dotenv.load_dotenv", return_value=None):
            client = create_llm_client()
        assert isinstance(client, LLMClient)
        assert client.model == DEFAULT_MODEL
        assert client.base_url == DEFAULT_BASE_URL

    def test_env_overrides_model_and_base_url(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        monkeypatch.setenv("OPENROUTER_MODEL", "qwen/other-model")
        monkeypatch.setenv("OPENROUTER_BASE_URL", "http://custom")
        with patch("dotenv.load_dotenv", return_value=None):
            client = create_llm_client()
        assert client.model == "qwen/other-model"
        assert client.base_url == "http://custom"

    def test_passes_cost_tracker(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        tracker = MagicMock()
        with patch("dotenv.load_dotenv", return_value=None):
            client = create_llm_client(cost_tracker=tracker)
        assert client._cost_tracker is tracker
