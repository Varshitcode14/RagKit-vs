"""End-to-end pipeline tests, fully offline (MockLLM + HashingEmbedder).

Skipped if faiss is not installed / langgraph missing (ma_rag only).
"""

import pytest

from ragkit import RagKit, RagKitConfig, create_pipeline
from ragkit.llms.mock import MockLLM

pytest.importorskip("faiss")

CORPUS = [
    {"title": "rag", "text": "retrieval augmented generation grounds answers in evidence"},
    {"title": "marag", "text": "multi-agent rag decomposes questions into reasoning steps"},
    {"title": "faiss", "text": "faiss is a vector similarity search library"},
]


def _offline_config():
    cfg = RagKitConfig()
    cfg.embedding.backend = "hashing"
    return cfg


def test_traditional_pipeline_offline():
    rag = RagKit(
        pipeline="traditional", corpus=CORPUS, config=_offline_config(), llm=MockLLM()
    ).build()
    resp = rag.query("what is faiss?")
    assert resp.answer  # mock returns something
    assert resp.retrieved_titles
    assert resp.pipeline == "traditional_rag"


def test_ma_rag_pipeline_offline():
    pytest.importorskip("langgraph")
    rag = RagKit(pipeline="ma_rag", corpus=CORPUS, config=_offline_config(), llm=MockLLM()).build()
    resp = rag.query("what is retrieval augmented generation?")
    assert resp.pipeline == "ma_rag"
    assert resp.reasoning_steps >= 1
    assert resp.retrieved_titles  # accumulated across steps


def test_create_pipeline_requires_index():
    p = create_pipeline("traditional", config=_offline_config(), llm=MockLLM())
    with pytest.raises(RuntimeError):
        p.run("anything")  # not ingested yet


def test_ask_returns_string():
    rag = RagKit(
        pipeline="traditional", corpus=CORPUS, config=_offline_config(), llm=MockLLM()
    ).build()
    assert isinstance(rag.ask("what is faiss?"), str)
