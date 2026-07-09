"""The high-level ``RagKit`` facade — the few-lines-of-code entry point.

Provider-first usage
--------------------
    from ragkit import RagKit

    # explicit key
    rag = RagKit(pipeline="ma_rag", llm="groq",
                 api_key="gsk_...", model="llama-3.3-70b-versatile")
    rag.ingest("./docs")
    print(rag.ask("What is RAG?"))

    # or let RAGKit read the key from the environment / .env:
    #   GROQ_API_KEY=gsk_xxx  (also OPENAI_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY)
    rag = RagKit(pipeline="ma_rag", llm="groq", model="llama-3.3-70b-versatile")
    rag.ingest("./docs")
    print(rag.ask("What is RAG?"))
"""

from __future__ import annotations

from typing import Any

from ragkit.core.config import RagKitConfig
from ragkit.core.interfaces import BaseEmbedder, BaseLLM
from ragkit.core.types import RagResponse
from ragkit.pipelines import create_pipeline


class RagKit:
    """A configured RAG system that hides all internal wiring."""

    def __init__(
        self,
        pipeline: str = "ma_rag",
        *,
        llm: str | BaseLLM | None = None,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        embedder: str | BaseEmbedder | None = None,
        embedding_model: str | None = None,
        top_k: int | None = None,
        corpus: Any | None = None,
        config: RagKitConfig | dict | None = None,
        **components: Any,
    ) -> None:
        if isinstance(config, dict):
            config = RagKitConfig.from_dict(config)
        self.config = config or RagKitConfig()

        # ── LLM selection ─────────────────────────────────────────────
        if isinstance(llm, BaseLLM):
            components["llm"] = llm  # inject instance directly
        elif isinstance(llm, str):
            self.config.llm.backend = llm
        if model is not None:
            self.config.llm.model = model
        if api_key is not None:
            self.config.llm.api_key = api_key
        if temperature is not None:
            self.config.llm.temperature = temperature

        # ── Embedder selection ────────────────────────────────────────
        if isinstance(embedder, BaseEmbedder):
            components["embedder"] = embedder
        elif isinstance(embedder, str):
            self.config.embedding.backend = embedder
        if embedding_model is not None:
            self.config.embedding.model = embedding_model

        # ── Retrieval ─────────────────────────────────────────────────
        if top_k is not None:
            self.config.retrieval.top_k = top_k

        self.corpus = corpus
        self.pipeline_name = pipeline
        self._pipeline = create_pipeline(pipeline, config=self.config, **components)

    # ── lifecycle ─────────────────────────────────────────────────────
    def build(self, corpus: Any | None = None) -> RagKit:
        """Ingest and index the corpus (uses the one given at construction)."""
        source = corpus if corpus is not None else self.corpus
        if source is None:
            raise ValueError(
                "No corpus provided. Pass corpus=... to RagKit(...), build(...), "
                "or call ingest(...)."
            )
        self._pipeline.ingest(source)
        return self

    def ingest(self, corpus: Any) -> RagKit:
        """Load + index a corpus (a directory, file, PDF, or list of docs)."""
        self._pipeline.ingest(corpus)
        return self

    def load_index(self, path: str) -> RagKit:
        self._pipeline.load_index(path)  # type: ignore[attr-defined]
        return self

    def save_index(self, path: str) -> RagKit:
        self._pipeline.save_index(path)  # type: ignore[attr-defined]
        return self

    # ── querying ──────────────────────────────────────────────────────
    def query(self, question: str) -> RagResponse:
        """Answer a question, returning a full :class:`RagResponse`."""
        return self._pipeline.run(question)

    def ask(self, question: str) -> str:
        """Answer a question, returning just the answer string."""
        return self._pipeline.ask(question)

    @property
    def pipeline(self):
        return self._pipeline
