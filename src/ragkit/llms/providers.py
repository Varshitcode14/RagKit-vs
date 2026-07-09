"""Single-provider LLM backends.

Each backend implements :class:`BaseLLM`, imports its SDK lazily, and
resolves its API key in this order:

    1. an explicit ``api_key`` passed to ``RagKit(... api_key=...)``
    2. the provider's standard environment variable (loaded from ``.env``)

Standard env var names:
    GROQ_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY

This is the provider-first API: ``RagKit(pipeline="ma_rag", llm="groq")``.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from ragkit.core.config import LLMConfig
from ragkit.core.interfaces import BaseLLM
from ragkit.core.registry import LLM_REGISTRY
from ragkit.exceptions import MissingAPIKeyError, MissingDependencyError

load_dotenv()


class _ProviderLLM(BaseLLM):
    """Common base: key resolution + config handling."""

    env_var: str = ""
    default_model: str = ""
    # extra env vars to try (e.g. the legacy comma-list GROQ_API_KEYS)
    fallback_env_vars: tuple[str, ...] = ()

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()
        self.model = self.config.model or self.default_model
        self.api_key = self._resolve_key(self.config.api_key)
        if not self.api_key:
            raise MissingAPIKeyError(
                f"No API key for '{self.config.backend}'. Pass api_key=... to "
                f"RagKit(...) or set {self.env_var} in your environment / .env file."
            )

    def _resolve_key(self, explicit: str | None) -> str:
        if explicit:
            return explicit.strip()
        val = os.getenv(self.env_var, "").strip()
        if val:
            return val
        for var in self.fallback_env_vars:
            raw = os.getenv(var, "").strip()
            if raw:
                # support comma-separated lists; take the first key
                return raw.split(",")[0].strip()
        return ""

    def _temp(self, temperature: float | None) -> float:
        return self.config.temperature if temperature is None else temperature


@LLM_REGISTRY.register("groq")
class GroqLLM(_ProviderLLM):
    env_var = "GROQ_API_KEY"
    fallback_env_vars = ("GROQ_API_KEYS",)
    default_model = "llama-3.3-70b-versatile"

    def generate(self, prompt: str, temperature: float | None = None) -> str:
        from groq import Groq

        client = Groq(api_key=self.api_key)
        resp = client.chat.completions.create(
            model=self.model,
            temperature=self._temp(temperature),
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content


@LLM_REGISTRY.register("openai")
class OpenAILLM(_ProviderLLM):
    env_var = "OPENAI_API_KEY"
    default_model = "gpt-4o-mini"

    def generate(self, prompt: str, temperature: float | None = None) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        resp = client.chat.completions.create(
            model=self.model,
            temperature=self._temp(temperature),
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content


@LLM_REGISTRY.register("anthropic")
class AnthropicLLM(_ProviderLLM):
    env_var = "ANTHROPIC_API_KEY"
    default_model = "claude-3-5-sonnet-latest"

    def generate(self, prompt: str, temperature: float | None = None) -> str:
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise MissingDependencyError("anthropic", extra="anthropic") from e
        client = anthropic.Anthropic(api_key=self.api_key)
        resp = client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=self._temp(temperature),
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in resp.content if hasattr(block, "text"))


@LLM_REGISTRY.register("google")
@LLM_REGISTRY.register("gemini")
class GoogleLLM(_ProviderLLM):
    env_var = "GOOGLE_API_KEY"
    default_model = "gemini-1.5-flash"

    def generate(self, prompt: str, temperature: float | None = None) -> str:
        try:
            import google.generativeai as genai
        except ImportError as e:  # pragma: no cover
            raise MissingDependencyError("google-generativeai", extra="google") from e
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model)
        resp = model.generate_content(
            prompt,
            generation_config={"temperature": self._temp(temperature)},
        )
        return resp.text
