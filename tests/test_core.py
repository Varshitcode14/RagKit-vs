"""Tests for core types, config, and registries."""

from ragkit.core.config import RagKitConfig
from ragkit.core.registry import Registry
from ragkit.core.types import Document, RagResponse, as_doc_dicts


def test_document_mapping_access():
    doc = Document(title="T", text="body", metadata={"tag": "x"})
    assert doc["title"] == "T"
    assert doc.get("missing", 42) == 42
    assert doc.get("tag") == "x"
    assert doc.to_dict()["text"] == "body"


def test_document_from_dict_moves_extra_to_metadata():
    doc = Document.from_dict({"title": "A", "text": "b", "extra": 1})
    assert doc.metadata["extra"] == 1


def test_rag_response_str_is_answer():
    r = RagResponse(answer="hello")
    assert str(r) == "hello"


def test_as_doc_dicts_mixed():
    out = as_doc_dicts([Document(title="a"), {"title": "b"}])
    assert out[0]["title"] == "a"
    assert out[1]["title"] == "b"


def test_config_roundtrip():
    cfg = RagKitConfig()
    cfg.retrieval.top_k = 9
    restored = RagKitConfig.from_dict(cfg.to_dict())
    assert restored.retrieval.top_k == 9


def test_registry_register_and_create():
    reg = Registry("thing")

    @reg.register("foo")
    def make(x):
        return x * 2

    assert reg.create("foo", 3) == 6
    assert "foo" in reg
    assert "foo" in reg.available()
