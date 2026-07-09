"""Shared pipeline machinery.

``IndexedPipeline`` centralizes everything a RAG pipeline needs before it
can answer questions: it lazily builds the embedder / chunker / store / LLM
from config (or accepts injected components), and turns a corpus into a
searchable index. Concrete pipelines (traditional, MA-RAG) subclass it and
implement only :meth:`run`.
"""

from __future__ import annotations

from typing import Any

from ragkit.core.interfaces import (
    BasePipeline,
    BaseLLM,
    BaseEmbedder,
    BaseChunker,
    BaseVectorStore,
    BaseRetriever,
)
from ragkit.core.config import RagKitConfig
from ragkit.corpus.loaders import load_corpus
from ragkit.prompts.builder import PromptBuilder
from ragkit.utils.logger import get_logger

_log = get_logger("ragkit.pipeline")


class IndexedPipeline(BasePipeline):
    name = "indexed"

    def __init__(
        self,
        config: RagKitConfig | None = None,
        *,
        llm: BaseLLM | None = None,
        embedder: BaseEmbedder | None = None,
        chunker: BaseChunker | None = None,
        store: BaseVectorStore | None = None,
        retriever: BaseRetriever | None = None,
        prompt_builder: PromptBuilder | None = None,
    ) -> None:
        self.config = config or RagKitConfig()
        self._llm = llm
        self._embedder = embedder
        self._chunker = chunker
        self._store = store
        self._retriever = retriever
        self.prompt_builder = prompt_builder or PromptBuilder()

    # ── lazily-built components ───────────────────────────────────────
    @property
    def llm(self) -> BaseLLM:
        if self._llm is None:
            from ragkit.llms import create_llm
            self._llm = create_llm(self.config.llm)
        return self._llm

    @property
    def embedder(self) -> BaseEmbedder:
        if self._embedder is None:
            from ragkit.embedders import create_embedder
            self._embedder = create_embedder(self.config.embedding)
        return self._embedder

    @property
    def chunker(self) -> BaseChunker:
        if self._chunker is None:
            from ragkit.chunkers import create_chunker
            self._chunker = create_chunker(self.config.chunking)
        return self._chunker

    @property
    def store(self) -> BaseVectorStore:
        if self._store is None:
            from ragkit.stores import create_store
            self._store = create_store(self.config.store, dim=self.embedder.dimension)
        return self._store

    @property
    def retriever(self) -> BaseRetriever:
        if self._retriever is None:
            from ragkit.retrievers.dense import DenseRetriever
            self._retriever = DenseRetriever(
                embedder=self.embedder,
                store=self.store,
                config=self.config.retrieval,
            )
        return self._retriever

    # ── indexing ──────────────────────────────────────────────────────
    def ingest(self, corpus: Any) -> "IndexedPipeline":
        documents = load_corpus(corpus)
        chunks = self.chunker.split_many(documents)

        doc_dicts: list[dict] = []
        texts: list[str] = []
        for c in chunks:
            d = {"title": c.title, "text": c.text, "source": c.source}
            d.update(c.metadata)
            doc_dicts.append(d)
            texts.append(c.text)

        _log.info("Encoding %d chunks ...", len(texts))
        vectors = self.embedder.encode_batch(texts)
        self.store.add(vectors, doc_dicts)
        # rebuild retriever bound to the (now populated) store
        self._retriever = None
        _ = self.retriever
        return self

    def build(self, corpus: Any | None = None) -> "IndexedPipeline":
        """Alias for ingest; usable when corpus was provided at construction."""
        if corpus is not None:
            return self.ingest(corpus)
        return self

    def load_index(self, path: str) -> "IndexedPipeline":
        self.store.load(path)
        self._retriever = None
        _ = self.retriever
        return self

    def save_index(self, path: str) -> "IndexedPipeline":
        self.store.save(path)
        return self

    @property
    def is_indexed(self) -> bool:
        return self._store is not None and len(self._store) > 0

    def _ensure_indexed(self) -> None:
        if not self.is_indexed:
            raise RuntimeError(
                "Pipeline has no index. Call ingest(corpus) or load_index(path) "
                "before querying."
            )
