"""MA-RAG LangGraph construction.

Builds the same planner -> (set_goal -> step_definer -> retriever ->
extractor -> qa -> update_history -> next_step)* -> final_answer graph as
the original project, but the graph is *constructed* with injected
components instead of relying on import-time module singletons.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from ragkit.core.interfaces import BaseLLM, BaseRetriever
from ragkit.pipelines.ma_rag import agents as A
from ragkit.pipelines.ma_rag.state import GraphState
from ragkit.prompts.builder import PromptBuilder


def _get_current_goal(state: dict):
    if state["current_step"] >= len(state["plan"]):
        return None
    return state["plan"][state["current_step"]]


def _add_step_answer(state: dict, answer: str) -> dict:
    state["step_answers"].append(answer)
    state["history"].append(
        {
            "step": state["current_step"] + 1,
            "goal": state["current_goal"],
            "answer": answer,
        }
    )
    return state


def _next_step(state: dict) -> dict:
    state["current_step"] += 1
    state["current_goal"] = ""
    state["subquery"] = ""
    state["retrieved_docs"] = []
    state["evidence"] = ""
    state["current_answer"] = ""
    return state


def build_graph(
    retriever: BaseRetriever,
    llm: BaseLLM,
    prompt_builder: PromptBuilder,
    top_k: int = 5,
    max_steps: int = 3,
) -> Callable:
    """Compile and return the MA-RAG graph."""
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as e:
        from ragkit.exceptions import MissingDependencyError

        raise MissingDependencyError("langgraph", extra="ma_rag") from e

    pb = prompt_builder

    # ── node functions (closures over injected components) ────────────
    def planner(state):
        return A.planner_agent(state, llm, pb, max_steps)

    def set_current_goal(state):
        goal = _get_current_goal(state)
        if goal is None:
            return state
        state["current_goal"] = goal
        return state

    def generate_subquery(state):
        return A.step_definer_agent(state, llm, pb)

    def retrieve_documents(state):
        start = time.perf_counter()
        docs = retriever.search(query=state["subquery"], top_k=top_k)
        elapsed = time.perf_counter() - start

        state["retrieved_docs"] = docs
        state["retrieval_time"] = state.get("retrieval_time", 0.0) + elapsed

        all_titles = state.get("all_retrieved_titles", [])
        all_docs = state.get("all_retrieved_docs", [])
        seen = set(all_titles)
        for doc in docs:
            title = doc.get("title", "")
            all_docs.append(doc)
            if title and title not in seen:
                all_titles.append(title)
                seen.add(title)
        state["all_retrieved_titles"] = all_titles
        state["all_retrieved_docs"] = all_docs
        return state

    def extract_evidence(state):
        return A.extractor_agent(state, llm, pb)

    def answer_current_step(state):
        state["current_answer"] = A.qa_agent(state, llm, pb)
        return state

    def update_history(state):
        state = _add_step_answer(state, state["current_answer"])
        state["current_answer"] = ""
        return state

    def move_to_next_step(state):
        return _next_step(state)

    def final_answer(state):
        return A.final_answer_agent(state, llm, pb)

    def route_next_step(state):
        if state["current_step"] < len(state["plan"]):
            return "set_goal"
        return "final_answer"

    # ── wire the graph ────────────────────────────────────────────────
    builder = StateGraph(GraphState)
    builder.add_node("planner", planner)
    builder.add_node("set_goal", set_current_goal)
    builder.add_node("step_definer", generate_subquery)
    builder.add_node("retriever", retrieve_documents)
    builder.add_node("extractor", extract_evidence)
    builder.add_node("qa", answer_current_step)
    builder.add_node("update_history", update_history)
    builder.add_node("next_step", move_to_next_step)
    builder.add_node("final_answer", final_answer)

    builder.set_entry_point("planner")
    builder.add_edge("planner", "set_goal")
    builder.add_edge("set_goal", "step_definer")
    builder.add_edge("step_definer", "retriever")
    builder.add_edge("retriever", "extractor")
    builder.add_edge("extractor", "qa")
    builder.add_edge("qa", "update_history")
    builder.add_edge("update_history", "next_step")
    builder.add_conditional_edges(
        "next_step",
        route_next_step,
        {"set_goal": "set_goal", "final_answer": "final_answer"},
    )
    builder.add_edge("final_answer", END)

    return builder.compile()
