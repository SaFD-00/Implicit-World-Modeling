"""Text generation strategies for InputText actions.

Two strategies:
  - RandomTextGenerator: picks from a fixed sample list (legacy behavior).
  - LLMTextGenerator: asks the shared :class:`LLMClient` (OpenRouter Chat
    Completions, e.g. ``qwen/qwen3.7-plus``) for contextually appropriate text;
    falls back to random on failure.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from loguru import logger

from monkey_collector.xml.structured_parser import encode_to_html_xml
from monkey_collector.xml.ui_tree import UIElement

if TYPE_CHECKING:
    from monkey_collector.llm.client import LLMClient

# Default sample texts (same as explorer.SAMPLE_TEXTS)
SAMPLE_TEXTS = [
    "Hello World",
    "Test Note",
    "Meeting at 3pm",
    "Shopping list",
    "Important memo",
    "John Doe",
    "test@example.com",
    "12345",
    "New item",
    "Quick note",
]

SYSTEM_PROMPT = (
    "You are a mobile app tester generating realistic input text for Android UI fields.\n"
    "Given the current screen's UI structure and a target input field, generate a single "
    "realistic text value that a real user would type into this field.\n\n"
    "Rules:\n"
    "- Return ONLY the text to type, nothing else (no quotes, no explanation)\n"
    "- Make the text contextually appropriate for the field and the app screen\n"
    "- Vary your responses: use different names, addresses, search terms, etc.\n"
    "- Keep text concise (1-30 characters typically)\n"
    "- Use the field's resource ID, description, and surrounding UI context as clues\n"
    "- For search fields: generate diverse search queries relevant to the app\n"
    "- For name fields: generate realistic names\n"
    "- For email fields: generate realistic email addresses\n"
    "- For number fields: generate appropriate numbers\n"
    "- For note/memo fields: generate short realistic notes\n"
    "- For password fields: generate a test password like \"Test1234!\"\n"
    "- When an 'App under test' description is provided, tailor the text to that "
    "app's domain (e.g. product search terms for a shopping app, note content for "
    "a notes app, a task title for a to-do app)"
)

USER_PROMPT_TEMPLATE = (
    "<screen>{screen_xml}</screen>\n\n"
    "Target input field:\n"
    "- resource_id: {resource_id}\n"
    "- content_desc: {content_desc}\n"
    "- current_text: {current_text}\n"
    "- display_name: {display_name}\n\n"
    "Generate appropriate input text for this field."
)


class TextGenerator(ABC):
    """Base class for input text generation strategies."""

    @abstractmethod
    def generate(self, element: UIElement, raw_xml: str) -> str:
        """Generate text appropriate for *element* given the current screen XML."""

    def set_app_context(self, app_context: str) -> None:  # noqa: B027
        """Set a human-readable description of the app under test.

        Optional hook used by LLM-backed strategies to ground generated text in
        the current app's domain. Intentionally a concrete no-op (not abstract)
        so strategies that ignore context (e.g. random generation) need not
        override it.
        """


class RandomTextGenerator(TextGenerator):
    """Select a random text from a fixed sample list."""

    def __init__(self, rng: random.Random, sample_texts: list[str] | None = None):
        self._rng = rng
        self._sample_texts = sample_texts or SAMPLE_TEXTS

    def generate(self, element: UIElement, raw_xml: str) -> str:
        return self._rng.choice(self._sample_texts)


class LLMTextGenerator(TextGenerator):
    """Use the shared :class:`LLMClient` to generate contextual input text."""

    def __init__(
        self,
        llm_client: LLMClient,
        fallback_texts: list[str] | None = None,
        rng: random.Random | None = None,
    ):
        self._client = llm_client
        self._fallback_texts = fallback_texts or SAMPLE_TEXTS
        self._rng = rng or random.Random()
        self._app_context = ""

    def set_app_context(self, app_context: str) -> None:
        self._app_context = (app_context or "").strip()

    def generate(self, element: UIElement, raw_xml: str) -> str:
        try:
            screen_xml = encode_to_html_xml(raw_xml) if raw_xml else ""
            app_block = (
                f"App under test: {self._app_context}\n\n" if self._app_context else ""
            )
            user_msg = app_block + USER_PROMPT_TEMPLATE.format(
                screen_xml=screen_xml,
                resource_id=element.resource_id or "(none)",
                content_desc=element.content_desc or "(none)",
                current_text=element.text or "(empty)",
                display_name=element.display_name or "(unknown)",
            )

            text = self._client.chat(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=50,
                agent="text_generator",
            )
            text = (text or "").strip().strip('"').strip("'")
            if text:
                logger.debug(f"LLM generated text: {text!r}")
                return text

            logger.warning("LLM returned empty text, falling back to random")
            return self._rng.choice(self._fallback_texts)

        except Exception as e:
            logger.warning(f"LLM text generation failed ({e}), falling back to random")
            return self._rng.choice(self._fallback_texts)


def create_text_generator(
    mode: str,
    seed: int = 42,
    sample_texts: list[str] | None = None,
    llm_client: LLMClient | None = None,
) -> TextGenerator:
    """Factory: create a TextGenerator based on *mode* and client availability.

    Args:
        mode: ``"api"`` for LLM-based or ``"random"`` for hardcoded sample texts.
        seed: Random seed for reproducibility.
        sample_texts: Override default sample texts.
        llm_client: Shared LLM client. When ``mode="api"`` but no client is
            available (e.g. ``OPENROUTER_API_KEY`` unset), falls back to random.
    """
    rng = random.Random(seed)

    if mode == "random" or llm_client is None:
        if mode == "api" and llm_client is None:
            logger.warning(
                "Input mode 'api' requested but no LLM client available — "
                "falling back to random text generation."
            )
        return RandomTextGenerator(rng, sample_texts)

    logger.info(f"Using LLM text generation (model: {llm_client.model})")
    return LLMTextGenerator(
        llm_client=llm_client,
        fallback_texts=sample_texts or SAMPLE_TEXTS,
        rng=rng,
    )
