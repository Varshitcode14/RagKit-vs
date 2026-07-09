"""Multi-Agent RAG pipeline.

Wraps the LangGraph graph and normalizes its raw state into a
:class:`RagResponse`, reproducing the original ``MA_RAG.pipeline.run``
semantics (titles/docs accumulated across all reasoning steps).
"""

from __future__ import annotations

import time

from ragkit.core.types import RagResponse
from ragkit.core.registry import PIPELINE_REGISTRY
from ragkit.pipelines.base import IndexedPipeline
from ragkit.pipelines.ma_rag.state import initial_state


@PIPELINE_REGISTRY.register("ma_rag")
class MARagPipeline(IndexedPipeline):
    name = "ma_rag"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._graph = None

    @property
    def graph(self):
        if self._graph is None:
            self._ensure_indexed()
            from ragkit.pipelines.ma_rag.graph import build_graph
            self._graph = build_graph(
                retriever=self.retriever,
                llm=self.llm,
                prompt_builder=self.prompt_builder,
                top_k=self.config.retrieval.top_k,
                max_steps=self.config.ma_rag.max_steps,
            )
        return self._graph

    def ingest(self, corpus):
        result = super().ingest(corpus)
        self._graph = None  # rebuild graph against the fresh retriever
        return result

    def load_index(self, path):
        result = super().load_index(path)
        self._graph = None
        return result

    def run(self, question: str) -> RagResponse:
        self._ensure_indexed()
        start = time.perf_counter()
        final_state = self.graph.invoke(initial_state(question))
        total_time = time.perf_counter() - start
        return self._normalize(final_state, total_time)

    def _normalize(self, state: dict, total_time: float) -> RagResponse:
        all_titles = state.get("all_retrieved_titles", [])
        all_docs = state.get("all_retrieved_docs", [])
        context = "\n\n".join(
            f"Title: {d.get('title', '')}\n{d.get('text', '')}" for d in all_docs
        )
        retrieval_time = float(state.get("retrieval_time", 0.0))
        generation_time = max(total_time - retrieval_time, 0.0)

        return RagResponse(
            question=state.get("question", ""),
            answer=state.get("final_answer", ""),
            retrieved_docs=all_docs,
            retrieved_titles=all_titles,
            context=context,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            total_time=total_time,
            history=state.get("history", []),
            reasoning_steps=len(state.get("plan", [])),
            pipeline=self.name,
        )
