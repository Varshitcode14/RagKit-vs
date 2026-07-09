"""Index persistence and pipeline switching (offline)."""

import pytest

from ragkit import RagKit, RagKitConfig, available_pipelines, create_pipeline
from ragkit.llms.mock import MockLLM

pytest.importorskip("faiss")

CORPUS = [
    {"title": "rag", "text": "retrieval augmented generation grounds answers in evidence"},
    {"title": "faiss", "text": "faiss is a vector similarity search library"},
]


def _config():
    cfg = RagKitConfig()
    cfg.embedding.backend = "hashing"
    return cfg


def test_save_and_load_index(tmp_path):
    index_dir = tmp_path / "idx"

    rag = RagKit(pipeline="traditional", corpus=CORPUS, config=_config(), llm=MockLLM()).build()
    rag.save_index(str(index_dir))
    assert (index_dir / "faiss.index").exists()
    assert (index_dir / "metadata.pkl").exists()

    rag2 = RagKit(pipeline="traditional", config=_config(), llm=MockLLM()).load_index(
        str(index_dir)
    )
    resp = rag2.query("what is faiss?")
    assert resp.retrieved_titles  # served from the reloaded index


def test_pipeline_switching():
    assert set(available_pipelines()) >= {"traditional", "ma_rag"}
    for name in ("traditional", "ma_rag"):
        if name == "ma_rag":
            pytest.importorskip("langgraph")
        p = create_pipeline(name, config=_config(), llm=MockLLM())
        p.ingest(CORPUS)
        assert p.run("what is faiss?").pipeline in ("traditional_rag", "ma_rag")
