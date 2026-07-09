"""Traditional (single-pass) RAG pipeline.

Refactor of the original ``Traditional_RAG`` package: retrieve top-k, build
a context block, generate an answer. Now expressed against RAGKit
interfaces with injectable components.
"""

from __future__ import annotations

import time

from ragkit.core.registry import PIPELINE_REGISTRY
from ragkit.core.types import RagResponse
from ragkit.pipelines.base import IndexedPipeline
from ragkit.prompts.templates import NOT_FOUND_RESPONSE


@PIPELINE_REGISTRY.register("traditional")
class TraditionalRAGPipeline(IndexedPipeline):
    name = "traditional_rag"

    def run(self, question: str) -> RagResponse:
        self._ensure_indexed()
        total_start = time.perf_counter()

        retrieval_start = time.perf_counter()
        docs = self.retriever.search(question, top_k=self.config.retrieval.top_k)
        retrieval_time = time.perf_counter() - retrieval_start

        context = self.prompt_builder.build_context(docs)

        # Grounded-only guarantee: if nothing relevant was retrieved (e.g. all
        # candidates were filtered out by min_score), do not call the LLM at
        # all — state clearly that the answer is not in the documents.
        if not docs:
            total_time = time.perf_counter() - total_start
            return RagResponse(
                question=question,
                answer=NOT_FOUND_RESPONSE,
                retrieved_docs=[],
                retrieved_titles=[],
                context="",
                retrieval_time=retrieval_time,
                generation_time=0.0,
                total_time=total_time,
                history=[],
                reasoning_steps=1,
                pipeline=self.name,
            )

        prompt = self.prompt_builder.traditional(context=context, question=question)

        generation_start = time.perf_counter()
        answer = self.llm.generate(prompt, temperature=self.config.llm.temperature)
        generation_time = time.perf_counter() - generation_start

        total_time = time.perf_counter() - total_start

        return RagResponse(
            question=question,
            answer=answer.strip(),
            retrieved_docs=docs,
            retrieved_titles=[d.get("title", "") for d in docs],
            context=context,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            total_time=total_time,
            history=[],
            reasoning_steps=1,
            pipeline=self.name,
        )
