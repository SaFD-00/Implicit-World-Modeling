"""Shared OpenRouter (OpenAI-compatible Chat Completions) LLM client.

A single, provider-agnostic client reused by every LLM consumer in the
collector — input text generation (``pipeline/text_generator.py``) and screen
element semantic grouping (``llm/screen_grouper.py``). Modeled on the
``GPT`` helper from the reference ``LLM-Explorer`` project, but cleaned up:
``base_url`` / ``api_key`` / ``model`` are env-driven so the provider can be
swapped without touching call sites.

OpenRouter speaks the OpenAI **Chat Completions** API (``chat.completions``),
not the Responses API — so this client uses ``client.chat.completions.create``
and reads ``usage.prompt_tokens`` / ``usage.completion_tokens`` for cost
tracking (the Responses API names them ``input_tokens`` / ``output_tokens``).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from openai import OpenAI

    from monkey_collector.domain.cost_tracker import CostTracker

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "qwen/qwen3.7-plus"
DEFAULT_TIMEOUT = 30.0


class LLMClient:
    """Thin wrapper over the OpenAI SDK pointed at OpenRouter.

    One instance is shared across consumers; each consumer passes its own
    ``agent`` label to :meth:`chat` so per-call cost is attributed correctly
    in ``cost.csv``. The current exploration step (used for cost rows) is held
    on the client via :meth:`set_step` and set once per loop iteration.
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        cost_tracker: CostTracker | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self._api_key = api_key
        self.model = model
        self.base_url = base_url
        self._cost_tracker = cost_tracker
        self._timeout = timeout
        self._client: OpenAI | None = None  # lazy-init
        self._current_step: int = 0

    def set_step(self, step: int) -> None:
        """Set the current exploration step for cost attribution."""
        self._current_step = step

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(base_url=self.base_url, api_key=self._api_key)
        return self._client

    def chat(
        self,
        messages: str | list[dict],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        response_format: dict | None = None,
        agent: str = "llm",
    ) -> str:
        """Run a single chat completion and return the message content.

        Args:
            messages: Either a single user prompt string, or a list of
                ``{"role", "content"}`` message dicts (system + user, etc.).
            max_tokens: Optional cap on output tokens.
            temperature: Optional sampling temperature.
            response_format: Optional response_format dict, e.g.
                ``{"type": "json_object"}`` (silently ignored by providers
                that don't support it).
            agent: Cost-attribution label (e.g. ``"text_generator"``,
                ``"screen_grouper"``).

        Raises:
            Any exception from the underlying SDK call — callers are expected
            to handle/fallback. This method does not swallow errors so that
            failures are visible to the consumer's own fallback logic.
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "timeout": self._timeout,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
        if response_format is not None:
            kwargs["response_format"] = response_format

        client = self._get_client()
        completion = client.chat.completions.create(**kwargs)

        self._record_cost(completion, agent)

        choices = getattr(completion, "choices", None) or []
        if not choices:
            return ""
        content = getattr(choices[0].message, "content", None)
        return content or ""

    def _record_cost(self, completion, agent: str) -> None:
        if self._cost_tracker is None:
            return
        usage = getattr(completion, "usage", None)
        if not usage:
            return
        self._cost_tracker.record(
            model=self.model,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            step=self._current_step,
            agent=agent,
        )


def create_llm_client(cost_tracker: CostTracker | None = None) -> LLMClient | None:
    """Build a shared :class:`LLMClient` from environment configuration.

    Reads (after loading ``.env`` if available):

    * ``OPENROUTER_API_KEY`` — required; without it this returns ``None`` so
      callers fall back (random text / no grouping).
    * ``OPENROUTER_BASE_URL`` — defaults to ``https://openrouter.ai/api/v1``.
    * ``OPENROUTER_MODEL`` — defaults to ``qwen/qwen3.7-plus``.

    Returns ``None`` when no API key is configured.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        logger.warning(
            "python-dotenv not installed, reading OPENROUTER_* from environment only"
        )

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        logger.warning(
            "OPENROUTER_API_KEY not set — LLM features disabled (random text, "
            "no screen grouping). Set it in .env or environment to enable."
        )
        return None

    base_url = os.environ.get("OPENROUTER_BASE_URL", DEFAULT_BASE_URL)
    model = os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
    logger.info(f"LLM client ready (provider=OpenRouter, model={model})")
    return LLMClient(
        api_key=api_key,
        model=model,
        base_url=base_url,
        cost_tracker=cost_tracker,
    )
