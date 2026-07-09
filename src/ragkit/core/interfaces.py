"""Abstract interfaces for every replaceable RAGKit component.

The design rule: every moving part (LLM, embedder, vector store, retriever,
chunker, corpus loader, pipeline) has an interface here. Concrete backends
live in their own subpackages and are swapped via configuration/registry.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np

from ragkit.core.types import Document, RagResponse


class BaseLLM(ABC):
    """A text-in / text-out language model."""

    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        """Return a completion for ``prompt``."""


class BaseEmbedder(ABC):
    """Turns text into dense vectors."""

    @abstractmethod
    def encode(self, text: str) -> np.ndarray:
        """Encode a single string -> 1-D float32 vector."""

    def encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        """Encode many strings -> 2-D (n, dim) float32 array.

        Default implementation loops; backends may override for speed.
        """
        return np.vstack([self.encode(t) for t in texts]).astype(np.float32)

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimensionality."""


class BaseVectorStore(ABC):
    """Stores vectors + their document metadata and does similarity search."""

    @abstractmethod
    def add(self, vectors: np.ndarray, documents: Sequence[dict]) -> None:
        """Add vectors and their aligned document metadata."""

    @abstractmethod
    def search(self, vector: np.ndarray, top_k: int) -> list[dict]:
        """Return the top-k documents (as dicts, with a ``score`` key)."""

    @abstractmethod
    def save(self, path: str) -> None: ...

    @abstractmethod
    def load(self, path: str) -> None: ...

    @abstractmethod
    def __len__(self) -> int: ...


class BaseRetriever(ABC):
    """Retrieves documents relevant to a query."""

    @abstractmethod
    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """Return the most relevant documents for ``query``."""


class BaseChunker(ABC):
    """Splits documents into retrievable chunks."""

    @abstractmethod
    def split(self, document: Document) -> list[Document]: ...

    def split_many(self, documents: Sequence[Document]) -> list[Document]:
        chunks: list[Document] = []
        for doc in documents:
            chunks.extend(self.split(doc))
        return chunks


class BaseCorpusLoader(ABC):
    """Loads a corpus from some source into Documents."""

    @abstractmethod
    def load(self, source: Any) -> list[Document]: ...


class BasePipeline(ABC):
    """A complete RAG pipeline: ingest a corpus, then answer questions."""

    name: str = "base"

    @abstractmethod
    def ingest(self, corpus: Any) -> "BasePipeline":
        """Load + index a corpus so the pipeline can answer questions."""

    @abstractmethod
    def run(self, question: str) -> RagResponse:
        """Answer a question, returning a full :class:`RagResponse`."""

    # Convenience wrappers -------------------------------------------------
    def ask(self, question: str) -> str:
        """Answer a question, returning just the answer text."""
        return self.run(question).answer

    def query(self, question: str) -> RagResponse:
        return self.run(question)
