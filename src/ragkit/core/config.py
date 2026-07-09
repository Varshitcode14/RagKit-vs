"""Configuration objects for RAGKit.

All tunables live here as dataclasses so behavior is explicit and
serializable, replacing the scattered module-level constants and
import-time ``os.getenv`` calls in the original project.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EmbeddingConfig:
    backend: str = "sentence_transformer"
    model: str = "BAAI/bge-small-en-v1.5"
    normalize: bool = True


@dataclass
class StoreConfig:
    backend: str = "faiss"
    metric: str = "cosine"  # cosine | l2


@dataclass
class RetrievalConfig:
    backend: str = "dense"
    top_k: int = 5
    # Drop retrieved chunks whose similarity score is below this value.
    # None disables the filter (default, fully backward compatible).
    min_score: float | None = None
    # Remove near-duplicate chunks before they reach the context builder.
    dedupe: bool = True
    # Jaccard token-overlap above which two chunks are considered duplicates.
    dedupe_threshold: float = 0.95
    # Over-fetch factor so filtering/dedup can still return `top_k` results.
    fetch_multiplier: int = 4


@dataclass
class ChunkingConfig:
    backend: str = "passage"
    # Word-window size. Documents shorter than this stay whole, so short
    # passages are untouched while long documents (e.g. PDFs) are split.
    # Set to 0 to always keep whole documents.
    chunk_size: int = 220
    chunk_overlap: int = 40


@dataclass
class LLMConfig:
    # backend / provider name:
    #   groq | openai | google | anthropic | provider_manager | mock | <custom>
    backend: str = "provider_manager"
    model: str | None = None  # provider default used when None
    api_key: str | None = None  # falls back to the provider's env var
    temperature: float = 0.0
    # provider_manager (multi-provider fallback) knobs
    groq_model: str = "llama-3.3-70b-versatile"
    cerebras_model: str = "gpt-oss-120b"


@dataclass
class PathsConfig:
    """Where RAGKit reads/writes artifacts. Defaults to a workspace cache."""

    work_dir: Path = field(default_factory=lambda: Path(".ragkit"))

    @property
    def index_dir(self) -> Path:
        return self.work_dir / "indexes"

    @property
    def results_dir(self) -> Path:
        return self.work_dir / "results"


@dataclass
class MARagConfig:
    """Multi-agent RAG specific settings."""

    max_steps: int = 3
    verbose: bool = False


@dataclass
class RagKitConfig:
    """Top-level configuration aggregating every component's settings."""

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    store: StoreConfig = field(default_factory=StoreConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    ma_rag: MARagConfig = field(default_factory=MARagConfig)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["paths"]["work_dir"] = str(self.paths.work_dir)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> RagKitConfig:
        data = dict(data or {})
        cfg = cls()
        if "embedding" in data:
            cfg.embedding = EmbeddingConfig(**data["embedding"])
        if "store" in data:
            cfg.store = StoreConfig(**data["store"])
        if "retrieval" in data:
            cfg.retrieval = RetrievalConfig(**data["retrieval"])
        if "chunking" in data:
            cfg.chunking = ChunkingConfig(**data["chunking"])
        if "llm" in data:
            cfg.llm = LLMConfig(**data["llm"])
        if "paths" in data:
            wd = data["paths"].get("work_dir", ".ragkit")
            cfg.paths = PathsConfig(work_dir=Path(wd))
        if "ma_rag" in data:
            cfg.ma_rag = MARagConfig(**data["ma_rag"])
        return cfg
