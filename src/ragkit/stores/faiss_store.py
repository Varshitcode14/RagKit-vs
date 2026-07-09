"""FAISS-backed vector store.

Refactor of the original ``FAISSStore``. The original could only *load* a
prebuilt index; this version can also *build* one incrementally (``add``),
persist it (``save``/``load``), and supports cosine (inner-product on
normalized vectors) or L2 metrics.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

import numpy as np

from ragkit.core.interfaces import BaseVectorStore
from ragkit.core.config import StoreConfig
from ragkit.core.registry import STORE_REGISTRY
from ragkit.utils.logger import get_logger

_log = get_logger("ragkit.store")

_INDEX_FILE = "faiss.index"
_META_FILE = "metadata.pkl"


@STORE_REGISTRY.register("faiss")
class FAISSVectorStore(BaseVectorStore):
    def __init__(self, config: StoreConfig | None = None, dim: int | None = None) -> None:
        self.config = config or StoreConfig()
        self.dim = dim
        self.index = None
        self.metadata: list[dict] = []
        if dim is not None:
            self._init_index(dim)

    def _init_index(self, dim: int) -> None:
        import faiss
        self.dim = dim
        # cosine == inner product on L2-normalized vectors.
        self.index = (
            faiss.IndexFlatL2(dim)
            if self.config.metric == "l2"
            else faiss.IndexFlatIP(dim)
        )

    def add(self, vectors: np.ndarray, documents: Sequence[dict]) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors[None, :]
        if self.index is None:
            self._init_index(vectors.shape[1])
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"Vector dim {vectors.shape[1]} != store dim {self.dim}"
            )
        self.index.add(vectors)
        self.metadata.extend(dict(d) for d in documents)

    def search(self, vector: np.ndarray, top_k: int) -> list[dict]:
        if self.index is None or len(self.metadata) == 0:
            return []
        vector = np.asarray(vector, dtype=np.float32)
        if vector.ndim == 1:
            vector = vector[None, :]
        top_k = min(top_k, len(self.metadata))
        scores, indices = self.index.search(vector, top_k)
        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            item = dict(self.metadata[idx])
            item["score"] = float(score)
            results.append(item)
        return results

    def save(self, path: str) -> None:
        import faiss
        d = Path(path)
        d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(d / _INDEX_FILE))
        with open(d / _META_FILE, "wb") as f:
            pickle.dump(self.metadata, f)
        _log.info("Saved FAISS store (%d vectors) to %s", len(self), d)

    def load(self, path: str) -> None:
        import faiss
        d = Path(path)
        index_file = d / _INDEX_FILE if d.is_dir() else d
        self.index = faiss.read_index(str(index_file))
        self.dim = self.index.d
        meta_file = (d / _META_FILE) if d.is_dir() else d.with_name(_META_FILE)
        with open(meta_file, "rb") as f:
            self.metadata = pickle.load(f)
        _log.info("Loaded FAISS store (%d vectors) from %s", len(self), d)

    @classmethod
    def from_path(cls, index_path: str, metadata_path: str,
                  config: StoreConfig | None = None) -> "FAISSVectorStore":
        """Load a legacy index where index + metadata live at explicit paths."""
        import faiss
        store = cls(config=config)
        store.index = faiss.read_index(str(index_path))
        store.dim = store.index.d
        with open(metadata_path, "rb") as f:
            store.metadata = pickle.load(f)
        return store

    def __len__(self) -> int:
        return len(self.metadata)
