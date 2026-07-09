"""Tests for provider-first LLM selection and key resolution (offline).

These never call the network — they only verify construction, model
defaults, and API-key resolution from args / environment variables.
"""

import pytest

from ragkit.core.config import LLMConfig
from ragkit.llms import available_llms
from ragkit.llms.providers import GroqLLM, OpenAILLM


def test_registry_has_providers():
    names = available_llms()
    for expected in ("groq", "openai", "anthropic", "google", "mock", "provider_manager"):
        assert expected in names


def test_explicit_api_key_and_default_model():
    llm = GroqLLM(LLMConfig(backend="groq", api_key="gsk_test"))
    assert llm.api_key == "gsk_test"
    assert llm.model == "llama-3.3-70b-versatile"  # provider default


def test_custom_model():
    llm = OpenAILLM(LLMConfig(backend="openai", api_key="sk-test", model="gpt-4o"))
    assert llm.model == "gpt-4o"


def test_key_from_standard_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    llm = OpenAILLM(LLMConfig(backend="openai"))
    assert llm.api_key == "sk-env"


def test_groq_falls_back_to_legacy_keys_list(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEYS", "k1,k2,k3")
    llm = GroqLLM(LLMConfig(backend="groq"))
    assert llm.api_key == "k1"


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEYS", raising=False)
    with pytest.raises(ValueError):
        GroqLLM(LLMConfig(backend="groq"))
