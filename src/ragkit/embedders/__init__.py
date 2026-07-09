"""Embedder backends."""

from ragkit.core.interfaces import BaseEmbedder
from ragkit.core.config import EmbeddingConfig
from ragkit.core.registry import EMBEDDER_REGISTRY
from ragkit.embedders.hashing import HashingEmbedder

# SentenceTransformer is imported lazily inside create_embedder so that the
# base package imports without the heavy dependency being installed.


def create_embedder(config: EmbeddingConfig | None = None) -> BaseEmbedder:
    config = config or EmbeddingConfig()
    if config.backend == "sentence_transformer":
        from ragkit.embedders.sentence_transformer import (  # noqa: F401
            SentenceTransformerEmbedder,
        )
    return EMBEDDER_REGISTRY.create(config.backend, config=config)


__all__ = ["BaseEmbedder", "HashingEmbedder", "create_embedder"]
