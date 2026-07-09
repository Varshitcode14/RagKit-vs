"""Dense retriever = embedder + vector store.

Refactor of the original ``DenseRetriever`` with injected components
(no import-time model loading) and a configurable default ``top_k``.

Post-processing (added in the retrieval-quality pass, all opt-in via
:class:`~ragkit.core.config.RetrievalConfig`):

- **min_score**: drop weakly-similar chunks below a threshold.
- **dedupe**: remove near-identical chunks so the LLM context is not padded
  with the same passage twice.

To keep ``top_k`` results after filtering, the store is over-fetched by
``fetch_multiplier`` and the list is trimmed back down afterwards.
"""

from __future__ import annotations

import re

from ragkit.core.config import RetrievalConfig
from ragkit.core.interfaces import BaseEmbedder, BaseRetriever, BaseVectorStore
from ragkit.core.registry import RETRIEVER_REGISTRY

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall((text or "").lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


@RETRIEVER_REGISTRY.register("dense")
class DenseRetriever(BaseRetriever):
    def __init__(
        self,
        embedder: BaseEmbedder,
        store: BaseVectorStore,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.config = config or RetrievalConfig()

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        k = top_k or self.config.top_k

        # Over-fetch so score-filtering / dedup can still yield k results.
        fetch_k = k
        if self.config.dedupe or self.config.min_score is not None:
            fetch_k = max(k * self.config.fetch_multiplier, k)

        vector = self.embedder.encode(query)
        results = self.store.search(vector, top_k=fetch_k)
        results = self._postprocess(results)
        return results[:k]

    # ------------------------------------------------------------------
    def _postprocess(self, results: list[dict]) -> list[dict]:
        # 1) drop weak retrievals
        if self.config.min_score is not None:
            results = [r for r in results if float(r.get("score", 0.0)) >= self.config.min_score]

        # 2) remove near-identical chunks (keep the highest-scoring one,
        #    which comes first because results are similarity-ordered)
        if self.config.dedupe:
            results = self._dedupe(results)

        return results

    def _dedupe(self, results: list[dict]) -> list[dict]:
        kept: list[dict] = []
        kept_tokens: list[set[str]] = []
        for r in results:
            toks = _tokens(r.get("text", ""))
            if any(_jaccard(toks, prev) >= self.config.dedupe_threshold for prev in kept_tokens):
                continue
            kept.append(r)
            kept_tokens.append(toks)
        return kept
