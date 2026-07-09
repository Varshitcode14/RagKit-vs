"""Passage chunker.

Mirrors the original project's behavior (one short passage == one chunk)
by default, but can optionally split long documents into overlapping
word-windows when ``chunk_size > 0``.
"""

from __future__ import annotations

from ragkit.core.interfaces import BaseChunker
from ragkit.core.config import ChunkingConfig
from ragkit.core.types import Document
from ragkit.core.registry import CHUNKER_REGISTRY


@CHUNKER_REGISTRY.register("passage")
class PassageChunker(BaseChunker):
    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    def split(self, document: Document) -> list[Document]:
        size = self.config.chunk_size
        if size <= 0:
            return [document]

        words = document.text.split()
        if len(words) <= size:
            return [document]

        overlap = max(0, min(self.config.chunk_overlap, size - 1))
        step = size - overlap
        windows: list[str] = []
        for start in range(0, len(words), step):
            window = words[start:start + size]
            if not window:
                break
            windows.append(" ".join(window))
            if start + size >= len(words):
                break

        multiple = len(windows) > 1
        chunks: list[Document] = []
        for i, text in enumerate(windows):
            meta = dict(document.metadata)
            meta["chunk_index"] = i
            # keep the original title on metadata; give each chunk a unique
            # title so retrieved sources are distinguishable.
            meta["doc_title"] = document.title
            title = f"{document.title} [part {i + 1}]" if multiple else document.title
            chunks.append(
                Document(title=title, text=text, source=document.source, metadata=meta)
            )
        return chunks


@CHUNKER_REGISTRY.register("none")
class NoOpChunker(BaseChunker):
    """Keeps documents whole (explicit no-op)."""

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    def split(self, document: Document) -> list[Document]:
        return [document]
