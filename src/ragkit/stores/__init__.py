"""Vector store backends."""

from ragkit.core.interfaces import BaseVectorStore
from ragkit.core.config import StoreConfig
from ragkit.core.registry import STORE_REGISTRY


def create_store(config: StoreConfig | None = None, dim: int | None = None) -> BaseVectorStore:
    config = config or StoreConfig()
    if config.backend == "faiss":
        from ragkit.stores.faiss_store import FAISSVectorStore  # noqa: F401
    return STORE_REGISTRY.create(config.backend, config=config, dim=dim)


__all__ = ["BaseVectorStore", "create_store"]
