"""LLM backends and the ``create_llm`` factory.

Providers (single-provider, pick one explicitly):
    groq, openai, anthropic, google/gemini
Meta backends:
    provider_manager  (Groq -> Cerebras -> Bedrock automatic fallback)
    mock              (offline, deterministic, no keys)
"""

from ragkit.core.config import LLMConfig
from ragkit.core.interfaces import BaseLLM
from ragkit.core.registry import LLM_REGISTRY
from ragkit.llms.mock import MockLLM

# Import backends so they register themselves.
from ragkit.llms.provider_manager import ProviderManager
from ragkit.llms.providers import AnthropicLLM, GoogleLLM, GroqLLM, OpenAILLM


def create_llm(config: LLMConfig | None = None) -> BaseLLM:
    """Build an LLM from config (defaults to the multi-provider manager)."""
    config = config or LLMConfig()
    return LLM_REGISTRY.create(config.backend, config=config)


def available_llms() -> list[str]:
    return LLM_REGISTRY.available()


__all__ = [
    "BaseLLM",
    "ProviderManager",
    "MockLLM",
    "GroqLLM",
    "OpenAILLM",
    "AnthropicLLM",
    "GoogleLLM",
    "create_llm",
    "available_llms",
]
