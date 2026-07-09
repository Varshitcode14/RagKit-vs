"""RAGKit core: types, interfaces, config and registries."""

from ragkit.core.types import Document, RagResponse, as_doc_dicts
from ragkit.core.interfaces import (
    BaseLLM,
    BaseEmbedder,
    BaseVectorStore,
    BaseRetriever,
    BaseChunker,
    BaseCorpusLoader,
    BasePipeline,
)
from ragkit.core.config import (
    RagKitConfig,
    EmbeddingConfig,
    StoreConfig,
    RetrievalConfig,
    ChunkingConfig,
    LLMConfig,
    PathsConfig,
    MARagConfig,
)
from ragkit.core.registry import (
    LLM_REGISTRY,
    EMBEDDER_REGISTRY,
    STORE_REGISTRY,
    RETRIEVER_REGISTRY,
    CHUNKER_REGISTRY,
    PIPELINE_REGISTRY,
)

__all__ = [
    "Document",
    "RagResponse",
    "as_doc_dicts",
    "BaseLLM",
    "BaseEmbedder",
    "BaseVectorStore",
    "BaseRetriever",
    "BaseChunker",
    "BaseCorpusLoader",
    "BasePipeline",
    "RagKitConfig",
    "EmbeddingConfig",
    "StoreConfig",
    "RetrievalConfig",
    "ChunkingConfig",
    "LLMConfig",
    "PathsConfig",
    "MARagConfig",
    "LLM_REGISTRY",
    "EMBEDDER_REGISTRY",
    "STORE_REGISTRY",
    "RETRIEVER_REGISTRY",
    "CHUNKER_REGISTRY",
    "PIPELINE_REGISTRY",
]
