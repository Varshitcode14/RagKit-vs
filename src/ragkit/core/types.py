"""Core data types shared across RAGKit.

These are deliberately simple, dependency-free dataclasses so that every
component (loaders, chunkers, retrievers, pipelines, evaluation) speaks the
same language.

``Document`` supports mapping-style access (``doc["title"]`` / ``doc.get(...)``)
so that retrieval results behave like the plain dicts used throughout the
original research code, keeping ported pipeline logic faithful.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Iterable


@dataclass
class Document:
    """A single unit of retrievable text.

    Attributes:
        title:    Short human-readable identifier (also used as the unit of
                  relevance in retrieval metrics).
        text:     The passage content.
        source:   Where the document came from (file path, corpus name, ...).
        metadata: Arbitrary extra fields (tags, url, chunk index, ...).
        score:    Retrieval score, populated by a vector store / retriever.
    """

    title: str = ""
    text: str = ""
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float | None = None

    # -- mapping-style access (backwards compatible with dict-based code) --
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key) if hasattr(self, key) else self.metadata[key]

    def get(self, key: str, default: Any = None) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        return self.metadata.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Document":
        known = {"title", "text", "source", "metadata", "score"}
        meta = dict(data.get("metadata", {}))
        for k, v in data.items():
            if k not in known:
                meta[k] = v
        return cls(
            title=data.get("title", ""),
            text=data.get("text", ""),
            source=data.get("source", ""),
            metadata=meta,
            score=data.get("score"),
        )


@dataclass
class RagResponse:
    """Standardized result returned by every pipeline.

    Mirrors the result schema used by the original research pipelines so the
    evaluation harness keeps working unchanged.
    """

    question: str = ""
    answer: str = ""
    retrieved_docs: list[dict] = field(default_factory=list)
    retrieved_titles: list[str] = field(default_factory=list)
    context: str = ""
    retrieval_time: float = 0.0
    generation_time: float = 0.0
    total_time: float = 0.0
    history: list[dict] = field(default_factory=list)
    reasoning_steps: int = 1
    pipeline: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:  # so print(rag.query(...)) shows the answer
        return self.answer


def as_doc_dicts(docs: Iterable[Any]) -> list[dict]:
    """Normalize a mix of Documents / dicts into plain dicts."""
    out: list[dict] = []
    for d in docs:
        if isinstance(d, Document):
            out.append(d.to_dict())
        elif isinstance(d, dict):
            out.append(d)
        else:  # pragma: no cover - defensive
            out.append({"title": "", "text": str(d)})
    return out
