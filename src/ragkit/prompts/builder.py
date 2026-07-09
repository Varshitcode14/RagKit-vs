"""Prompt building helpers.

Centralizes prompt formatting (context blocks, reasoning history) so
pipelines stay focused on control flow, and templates can be overridden.
"""

from __future__ import annotations

from collections.abc import Sequence

from ragkit.prompts import templates


class PromptBuilder:
    """Formats prompts from templates; templates can be overridden."""

    def __init__(self, overrides: dict[str, str] | None = None) -> None:
        self._t = {
            "traditional": templates.TRADITIONAL_RAG_PROMPT,
            "planner": templates.PLANNER_PROMPT,
            "step_definer": templates.STEP_DEFINER_PROMPT,
            "extractor": templates.EXTRACTOR_PROMPT,
            "qa": templates.QA_PROMPT,
            "final": templates.FINAL_ANSWER_PROMPT,
        }
        if overrides:
            self._t.update(overrides)

    def template(self, name: str) -> str:
        return self._t[name]

    # -- context / history formatting -------------------------------------
    @staticmethod
    def build_context(docs: Sequence[dict]) -> str:
        parts = []
        for i, doc in enumerate(docs, start=1):
            title = doc.get("title", f"Document {i}")
            text = doc.get("text", "")
            parts.append(f"[{i}] {title}\n{text}")
        return "\n\n".join(parts)

    @staticmethod
    def build_documents_block(docs: Sequence[dict]) -> str:
        block = ""
        for i, doc in enumerate(docs, start=1):
            block += (
                f"\nDOCUMENT {i}\n\n"
                f"Title:\n{doc.get('title', '')}\n\n"
                f"Content:\n{doc.get('text', '')}\n\n"
                f"{'-' * 60}\n\n"
            )
        return block

    @staticmethod
    def build_history(history: Sequence[dict]) -> str:
        out = ""
        for item in history:
            out += f"Step {item['step']}\nGoal: {item['goal']}\nAnswer: {item['answer']}\n\n"
        return out

    # -- convenience formatters -------------------------------------------
    def traditional(self, context: str, question: str) -> str:
        return self._t["traditional"].format(context=context, question=question)

    def planner(self, question: str) -> str:
        return self._t["planner"].format(question=question)

    def step_definer(self, question: str, step: str, history: str) -> str:
        return self._t["step_definer"].format(question=question, step=step, history=history)

    def extractor(self, goal: str, documents: str) -> str:
        return self._t["extractor"].format(goal=goal, documents=documents)

    def qa(self, goal: str, history: str, evidence: str) -> str:
        return self._t["qa"].format(goal=goal, history=history or "None", evidence=evidence)

    def final(self, question: str, history: str) -> str:
        return self._t["final"].format(question=question, history=history)
