"""Shared LLM integration: a single OpenRouter client reused by all consumers."""

from monkey_collector.llm.client import LLMClient, create_llm_client
from monkey_collector.llm.element_extractor import (
    ElementExtractor,
    ExtractedElement,
    create_element_extractor,
)

__all__ = [
    "LLMClient",
    "ElementExtractor",
    "ExtractedElement",
    "create_llm_client",
    "create_element_extractor",
]
