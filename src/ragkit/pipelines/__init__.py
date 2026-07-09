"""RAG pipelines and the ``create_pipeline`` factory."""

from __future__ import annotations

from typing import Any

from ragkit.core.interfaces import BasePipeline
from ragkit.core.config import RagKitConfig
from ragkit.core.registry import PIPELINE_REGISTRY

# Import concrete pipelines so they register themselves.
from ragkit.pipelines.traditional import TraditionalRAGPipeline
from ragkit.pipelines.ma_rag import MARagPipeline


def create_pipeline(
    name: str = "ma_rag",
    config: RagKitConfig | None = None,
    **kwargs: Any,
) -> BasePipeline:
    """Build a pipeline by name.

    Args:
        name:   Registered pipeline name (e.g. "ma_rag", "traditional").
        config: Optional :class:`RagKitConfig`.
        kwargs: Injected components (llm, embedder, chunker, store,
                retriever, prompt_builder).
    """
    return PIPELINE_REGISTRY.create(name, config=config, **kwargs)


def available_pipelines() -> list[str]:
    return PIPELINE_REGISTRY.available()


__all__ = [
    "BasePipeline",
    "TraditionalRAGPipeline",
    "MARagPipeline",
    "create_pipeline",
    "available_pipelines",
]
