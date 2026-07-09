"""Tests for corpus loading and chunking."""

import json

from ragkit.corpus.loaders import load_corpus
from ragkit.chunkers import create_chunker
from ragkit.core.config import ChunkingConfig
from ragkit.core.types import Document


def test_load_from_memory_list():
    docs = load_corpus([{"title": "t", "text": "hello"}, {"content": "world"}])
    assert len(docs) == 2
    assert docs[0].title == "t"
    assert docs[1].text == "world"


def test_load_text_dir(tmp_path):
    (tmp_path / "a.txt").write_text("alpha content", encoding="utf-8")
    (tmp_path / "b.md").write_text("beta content", encoding="utf-8")
    docs = load_corpus(str(tmp_path))
    titles = sorted(d.title for d in docs)
    assert titles == ["a", "b"]


def test_load_json_file(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps([{"title": "x", "text": "y"}]), encoding="utf-8")
    docs = load_corpus(str(p))
    assert docs[0].title == "x"


def test_load_mixed_directory(tmp_path):
    """A directory can hold multiple documents of mixed types."""
    (tmp_path / "note.txt").write_text("plain text doc", encoding="utf-8")
    (tmp_path / "data.json").write_text(
        json.dumps([{"title": "j1", "text": "one"}, {"title": "j2", "text": "two"}]),
        encoding="utf-8",
    )
    docs = load_corpus(str(tmp_path))
    titles = sorted(d.title for d in docs)
    assert titles == ["j1", "j2", "note"]  # 3 docs from 2 files


def test_passage_chunker_whole_doc():
    chunker = create_chunker(ChunkingConfig(chunk_size=0))
    chunks = chunker.split(Document(title="t", text="a b c d e"))
    assert len(chunks) == 1


def test_passage_chunker_windows():
    chunker = create_chunker(ChunkingConfig(chunk_size=3, chunk_overlap=1))
    text = " ".join(str(i) for i in range(10))
    chunks = chunker.split(Document(title="t", text=text))
    assert len(chunks) > 1
    # multi-chunk docs get distinguishable, numbered titles
    assert all(c.title.startswith("t [part ") for c in chunks)
    assert all(c.metadata["doc_title"] == "t" for c in chunks)
