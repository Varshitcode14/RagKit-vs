"""Tests for the exception hierarchy and that errors stay backward compatible.

Each RAGKit error also subclasses the built-in it replaced, so existing
``except ValueError`` / ``except FileNotFoundError`` / ``except RuntimeError``
code keeps catching them.
"""

import pytest

from ragkit import exceptions as exc
from ragkit.core.config import LLMConfig
from ragkit.corpus import load_corpus
from ragkit.llms.providers import GroqLLM


def test_missing_api_key_is_valueerror(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEYS", raising=False)
    with pytest.raises(exc.MissingAPIKeyError):
        GroqLLM(LLMConfig(backend="groq"))
    with pytest.raises(ValueError):  # backward compatible
        GroqLLM(LLMConfig(backend="groq"))


def test_corpus_not_found_is_filenotfound():
    with pytest.raises(exc.CorpusNotFoundError):
        load_corpus("this/path/does/not/exist")
    with pytest.raises(FileNotFoundError):  # backward compatible
        load_corpus("this/path/does/not/exist")


def test_unsupported_file_type(tmp_path):
    bad = tmp_path / "data.xyz"
    bad.write_text("nope", encoding="utf-8")
    with pytest.raises(exc.UnsupportedFileTypeError):
        load_corpus(str(bad))
    with pytest.raises(ValueError):  # backward compatible
        load_corpus(str(bad))


def test_missing_dependency_message():
    e = exc.MissingDependencyError("faiss-cpu", extra="faiss")
    assert "ragkit-vs[faiss]" in str(e)
    assert isinstance(e, ImportError)  # backward compatible


def test_not_indexed_error_is_runtimeerror():
    pytest.importorskip("faiss")
    from ragkit import create_pipeline
    from ragkit.core.config import RagKitConfig
    from ragkit.llms.mock import MockLLM

    cfg = RagKitConfig()
    cfg.embedding.backend = "hashing"
    p = create_pipeline("traditional", config=cfg, llm=MockLLM())
    with pytest.raises(exc.NotIndexedError):
        p.run("anything")
    with pytest.raises(RuntimeError):  # backward compatible
        p.run("anything")
