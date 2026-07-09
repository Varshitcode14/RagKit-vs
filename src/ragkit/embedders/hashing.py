"""Deterministic hashing embedder — no model download, no network.

Uses the hashing trick over word tokens to produce fixed-size L2-normalized
vectors. It is not semantically strong, but it is fully offline and
deterministic, which makes it ideal for unit/integration tests and quick
demos. It also serves as a template for future "vectorless" backends.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

from ragkit.core.config import EmbeddingConfig
from ragkit.core.interfaces import BaseEmbedder
from ragkit.core.registry import EMBEDDER_REGISTRY

_TOKEN = re.compile(r"[a-z0-9]+")


@EMBEDDER_REGISTRY.register("hashing")
class HashingEmbedder(BaseEmbedder):
    def __init__(self, config: EmbeddingConfig | None = None, dim: int = 256) -> None:
        self.config = config or EmbeddingConfig()
        self._dim = dim

    def encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        for tok in _TOKEN.findall((text or "").lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            idx = h % self._dim
            sign = 1.0 if (h >> 8) & 1 else -1.0
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    @property
    def dimension(self) -> int:
        return self._dim
