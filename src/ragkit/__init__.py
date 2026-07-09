"""RAGKit — build advanced (multi-agent) RAG systems in a few lines of code.

Quickstart
----------
    from ragkit import RagKit
    rag = RagKit(pipeline="ma_rag", corpus="./docs").build()
    print(rag.query("Explain Retrieval Augmented Generation"))

    # or, lower level:
    from ragkit import create_pipeline
    p = create_pipeline("traditional")
    p.ingest("./docs")
    print(p.ask("..."))
"""

from __future__ import annotations

__version__ = "0.1.0"

from ragkit import exceptions
from ragkit.core.config import RagKitConfig
from ragkit.core.types import Document, RagResponse
from ragkit.exceptions import (
    ConfigurationError,
    CorpusError,
    CorpusNotFoundError,
    EmbeddingError,
    EmptyCorpusError,
    LLMError,
    MissingAPIKeyError,
    MissingDependencyError,
    NotIndexedError,
    PipelineError,
    RagKitError,
    UnsupportedFileTypeError,
    VectorStoreError,
)
from ragkit.facade import RagKit
from ragkit.pipelines import available_pipelines, create_pipeline

__all__ = [
    "__version__",
    "RagKit",
    "create_pipeline",
    "available_pipelines",
    "RagKitConfig",
    "Document",
    "RagResponse",
    # exceptions
    "exceptions",
    "RagKitError",
    "ConfigurationError",
    "MissingDependencyError",
    "MissingAPIKeyError",
    "CorpusError",
    "CorpusNotFoundError",
    "UnsupportedFileTypeError",
    "EmptyCorpusError",
    "EmbeddingError",
    "VectorStoreError",
    "LLMError",
    "PipelineError",
    "NotIndexedError",
]
