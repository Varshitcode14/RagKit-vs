"""SentenceTransformer embedder (ported from the original ``Embedder``).

Preserves the original behavior: a process-wide singleton model, L2
normalized float32 vectors. TensorFlow is disabled before import to avoid
the Keras-3 crash noted in the original code.
"""

from __future__ import annotations

import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import numpy as np

from ragkit.core.interfaces import BaseEmbedder
from ragkit.core.config import EmbeddingConfig
from ragkit.core.registry import EMBEDDER_REGISTRY
from ragkit.utils.logger import get_logger

_log = get_logger("ragkit.embedder")


@EMBEDDER_REGISTRY.register("sentence_transformer")
class SentenceTransformerEmbedder(BaseEmbedder):
    """Dense embedder backed by sentence-transformers."""

    # cache models by name so repeated construction is cheap and avoids the
    # Windows "paging file too small" error from reloading the model.
    _models: dict[str, object] = {}

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig()
        self._model = self._load_model(self.config.model)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @classmethod
    def _load_model(cls, name: str):
        if name not in cls._models:
            from sentence_transformers import SentenceTransformer
            _log.info("Loading embedding model: %s", name)
            cls._models[name] = SentenceTransformer(name)
        return cls._models[name]

    def encode(self, text: str) -> np.ndarray:
        vec = self._model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize,
        ).astype(np.float32)
        return vec[0]

    def encode_batch(self, texts):
        return self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize,
            batch_size=32,
        ).astype(np.float32)

    @property
    def dimension(self) -> int:
        return self._dim
