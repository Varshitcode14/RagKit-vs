"""Chunker backends."""

from ragkit.core.interfaces import BaseChunker
from ragkit.core.config import ChunkingConfig
from ragkit.core.registry import CHUNKER_REGISTRY
from ragkit.chunkers.passage import PassageChunker, NoOpChunker


def create_chunker(config: ChunkingConfig | None = None) -> BaseChunker:
    config = config or ChunkingConfig()
    return CHUNKER_REGISTRY.create(config.backend, config=config)


__all__ = ["BaseChunker", "PassageChunker", "NoOpChunker", "create_chunker"]
