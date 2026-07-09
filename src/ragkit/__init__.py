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

from ragkit.core.types import Document, RagResponse
from ragkit.core.config import RagKitConfig
from ragkit.facade import RagKit
from ragkit.pipelines import create_pipeline, available_pipelines

__all__ = [
    "__version__",
    "RagKit",
    "create_pipeline",
    "available_pipelines",
    "RagKitConfig",
    "Document",
    "RagResponse",
]
