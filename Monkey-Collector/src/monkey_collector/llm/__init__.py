"""LLM integration: a single OpenRouter client, used for input-text generation only."""

from monkey_collector.llm.client import LLMClient, create_llm_client

__all__ = [
    "LLMClient",
    "create_llm_client",
]
