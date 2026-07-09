"""Offline / test LLMs.

``MockLLM`` needs no API keys and no network. It inspects the prompt to
return plausible output for each RAGKit agent, which makes the entire
multi-agent pipeline runnable in tests and offline demos.
"""

from __future__ import annotations

from ragkit.core.interfaces import BaseLLM
from ragkit.core.registry import LLM_REGISTRY


@LLM_REGISTRY.register("mock")
class MockLLM(BaseLLM):
    """Rule-based deterministic LLM for offline use and testing."""

    def __init__(self, config=None, canned: dict[str, str] | None = None) -> None:
        self.canned = canned or {}
        self.calls: list[str] = []

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        self.calls.append(prompt)
        p = prompt.lower()

        # Explicit canned overrides win first.
        for needle, response in self.canned.items():
            if needle.lower() in p:
                return response

        # Route on the distinctive agent-role sentence in each template so
        # matches don't collide with words that appear in retrieved corpus
        # text (e.g. a document that mentions "step definer").
        if "you are the planner agent" in p:
            return '["Find the key fact needed to answer the question."]'

        if "you are the step definer" in p:
            return "retrieval query for the current step"

        if "evidence extraction agent" in p:
            return "FACT 1\nSource: DOCUMENT 1\nEvidence:\nMocked supporting evidence."

        if "you are the final answer agent" in p:
            return "mock final answer"

        if "you are the qa agent" in p:
            return "mock step answer"

        # Traditional single-pass QA system (and any other prompt).
        return "mock answer"
