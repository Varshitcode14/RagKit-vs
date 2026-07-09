"""MA-RAG graph state (ported from the original ``GraphState``).

Accumulator keys are declared here so LangGraph persists them across nodes
and into the final state — the fix that made retrieval metrics valid in the
original project is preserved.
"""

from __future__ import annotations

from typing import TypedDict


class GraphState(TypedDict, total=False):
    question: str
    plan: list[str]
    current_step: int
    current_goal: str
    subquery: str
    retrieved_docs: list
    evidence: str
    current_answer: str
    step_answers: list
    history: list
    final_answer: str

    # accumulators (declared so LangGraph keeps them in the output state)
    all_retrieved_titles: list
    all_retrieved_docs: list
    retrieval_time: float


def initial_state(question: str) -> dict:
    return {
        "question": question,
        "plan": [],
        "current_step": 0,
        "current_goal": "",
        "subquery": "",
        "retrieved_docs": [],
        "evidence": "",
        "step_answers": [],
        "history": [],
        "current_answer": "",
        "final_answer": "",
        "all_retrieved_titles": [],
        "all_retrieved_docs": [],
        "retrieval_time": 0.0,
    }
