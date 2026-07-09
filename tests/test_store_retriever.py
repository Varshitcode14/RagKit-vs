"""Integration tests for the FAISS store + dense retriever (offline).

Uses the HashingEmbedder so no model download is needed. Skipped if faiss
is not installed.
"""

import pytest

from ragkit.core.config import EmbeddingConfig, RetrievalConfig, StoreConfig
from ragkit.embedders import create_embedder

faiss = pytest.importorskip("faiss")

from ragkit.retrievers.dense import DenseRetriever  # noqa: E402
from ragkit.stores.faiss_store import FAISSVectorStore  # noqa: E402


def _embedder():
    return create_embedder(EmbeddingConfig(backend="hashing"))


def test_store_add_and_search():
    emb = _embedder()
    store = FAISSVectorStore(StoreConfig(metric="cosine"), dim=emb.dimension)
    docs = [
        {"title": "faiss", "text": "faiss vector similarity search library"},
        {"title": "cats", "text": "cats are small domestic animals"},
    ]
    vectors = emb.encode_batch([d["text"] for d in docs])
    store.add(vectors, docs)
    assert len(store) == 2

    results = store.search(emb.encode("vector search library"), top_k=1)
    assert results[0]["title"] == "faiss"
    assert "score" in results[0]


def test_store_save_load(tmp_path):
    emb = _embedder()
    store = FAISSVectorStore(StoreConfig(), dim=emb.dimension)
    docs = [{"title": "a", "text": "alpha"}]
    store.add(emb.encode_batch(["alpha"]), docs)
    store.save(str(tmp_path))

    store2 = FAISSVectorStore(StoreConfig())
    store2.load(str(tmp_path))
    assert len(store2) == 1
    assert store2.metadata[0]["title"] == "a"


def test_dense_retriever():
    emb = _embedder()
    store = FAISSVectorStore(StoreConfig(), dim=emb.dimension)
    docs = [
        {"title": "rag", "text": "retrieval augmented generation grounds answers"},
        {"title": "sky", "text": "the sky is blue on a clear day"},
    ]
    store.add(emb.encode_batch([d["text"] for d in docs]), docs)
    retriever = DenseRetriever(emb, store, RetrievalConfig(top_k=1))
    out = retriever.search("retrieval augmented generation")
    assert out[0]["title"] == "rag"
