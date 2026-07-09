"""Tests for the retrieval-quality features (offline).

Covers the configurable similarity threshold, near-duplicate removal, and the
grounded-only "not available" fallback. Uses the hashing embedder so no model
download is needed; skipped if faiss is missing.
"""

import pytest

from ragkit.core.config import EmbeddingConfig, RagKitConfig, RetrievalConfig, StoreConfig
from ragkit.embedders import create_embedder
from ragkit.prompts.templates import NOT_FOUND_RESPONSE

pytest.importorskip("faiss")

from ragkit import RagKit  # noqa: E402
from ragkit.llms.mock import MockLLM  # noqa: E402
from ragkit.retrievers.dense import DenseRetriever, _jaccard, _tokens  # noqa: E402
from ragkit.stores.faiss_store import FAISSVectorStore  # noqa: E402


def _embedder():
    return create_embedder(EmbeddingConfig(backend="hashing"))


def _store_with(docs):
    emb = _embedder()
    store = FAISSVectorStore(StoreConfig(), dim=emb.dimension)
    store.add(emb.encode_batch([d["text"] for d in docs]), docs)
    return emb, store


def test_config_defaults_are_backward_compatible():
    cfg = RetrievalConfig()
    assert cfg.min_score is None  # threshold off by default
    assert cfg.dedupe is True


def test_jaccard_helper():
    assert _jaccard(_tokens("a b c"), _tokens("a b c")) == 1.0
    assert _jaccard(_tokens("a b"), _tokens("c d")) == 0.0


def test_dedupe_removes_near_identical_chunks():
    docs = [
        {"title": "d1", "text": "faiss is a vector similarity search library"},
        {"title": "d2", "text": "faiss is a vector similarity search library"},  # dup
        {"title": "d3", "text": "cats are small domestic animals kept as pets"},
    ]
    emb, store = _store_with(docs)
    retriever = DenseRetriever(emb, store, RetrievalConfig(top_k=5, dedupe=True))
    results = retriever.search("vector similarity search")
    texts = [r["text"] for r in results]
    assert len(texts) == len(set(texts))  # no duplicate text survived


def test_min_score_filters_weak_matches():
    docs = [{"title": "d1", "text": "retrieval augmented generation grounds answers"}]
    emb, store = _store_with(docs)
    # Impossibly high threshold -> everything filtered out.
    retriever = DenseRetriever(emb, store, RetrievalConfig(top_k=3, min_score=0.99))
    assert retriever.search("something completely unrelated zzz") == []


def test_traditional_returns_not_found_when_all_filtered():
    corpus = [{"title": "d1", "text": "retrieval augmented generation grounds answers"}]
    cfg = RagKitConfig()
    cfg.embedding.backend = "hashing"
    cfg.retrieval.min_score = 0.999  # filter everything
    rag = RagKit(pipeline="traditional", corpus=corpus, config=cfg, llm=MockLLM()).build()
    resp = rag.query("totally unrelated question xyz")
    assert resp.answer == NOT_FOUND_RESPONSE
    assert resp.retrieved_titles == []
