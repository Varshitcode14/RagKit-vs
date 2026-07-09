"""MA-RAG agents.

Each agent is a small pure-ish function taking the graph state plus the
injected ``llm`` and ``prompt_builder``. This replaces the original modules
that each constructed a global ``ProviderManager()`` at import time.
"""

from __future__ import annotations

import ast
import json
import re

from ragkit.core.interfaces import BaseLLM
from ragkit.prompts.builder import PromptBuilder

_FINAL_PREFIXES = [
    "Final Answer:", "final answer:", "Answer:", "answer:",
    "The answer is", "Based on", "According to",
]


def parse_plan(response: str, question: str, max_steps: int = 3) -> list[str]:
    """Robustly parse the planner output into a list of step strings."""
    if not response:
        return [question]

    text = response.strip()
    text = re.sub(r"^```(?:json|python)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    candidate = match.group(0) if match else text

    for parser in (json.loads, ast.literal_eval):
        try:
            plan = parser(candidate)
            if isinstance(plan, list):
                steps = [str(s).strip() for s in plan if str(s).strip()]
                if steps:
                    return steps[:max_steps]
        except Exception:
            continue
    return [question]


def planner_agent(state: dict, llm: BaseLLM, pb: PromptBuilder, max_steps: int = 3) -> dict:
    response = llm.generate(pb.planner(state["question"]))
    state["plan"] = parse_plan(response, state["question"], max_steps)
    state["current_step"] = 0
    return state


def step_definer_agent(state: dict, llm: BaseLLM, pb: PromptBuilder) -> dict:
    history = ""
    if state["history"]:
        history = "PREVIOUS STEPS\n\n" + pb.build_history(state["history"])
    query = llm.generate(
        pb.step_definer(
            question=state["question"],
            step=state["current_goal"],
            history=history,
        )
    )
    state["subquery"] = query
    return state


def extractor_agent(state: dict, llm: BaseLLM, pb: PromptBuilder) -> dict:
    documents = pb.build_documents_block(state["retrieved_docs"])
    evidence = llm.generate(pb.extractor(goal=state["current_goal"], documents=documents))
    state["evidence"] = evidence
    return state


def qa_agent(state: dict, llm: BaseLLM, pb: PromptBuilder) -> str:
    history = pb.build_history(state["history"]) if state["history"] else ""
    return llm.generate(
        pb.qa(goal=state["current_goal"], history=history, evidence=state["evidence"])
    )


def final_answer_agent(state: dict, llm: BaseLLM, pb: PromptBuilder) -> dict:
    history = pb.build_history(state["history"])
    answer = llm.generate(pb.final(question=state["question"], history=history), temperature=0)
    stripped = answer.strip()
    for prefix in _FINAL_PREFIXES:
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):].strip()
            break
    state["final_answer"] = stripped
    return state
