"""Corpus loaders.

Turn a variety of sources into a list of :class:`Document`:

- a directory of ``.txt`` / ``.md`` files (title = file stem),
- a ``.json`` file: list of objects with ``title``/``text`` (or ``content``),
- a ``.jsonl`` file: one such object per line,
- an in-memory list of dicts or Documents.

This generalizes the original hard-coded ``corpus/ai_ml/documents.py`` list.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from ragkit.core.interfaces import BaseCorpusLoader
from ragkit.core.types import Document

_TEXT_EXT = {".txt", ".md", ".markdown", ".rst"}
_PDF_EXT = {".pdf"}


def _to_document(obj: Any, default_source: str = "") -> Document:
    if isinstance(obj, Document):
        return obj
    if isinstance(obj, dict):
        text = obj.get("text") or obj.get("content") or obj.get("body") or ""
        title = obj.get("title") or obj.get("id") or (text[:40] if text else "")
        known = {"title", "text", "content", "body", "source", "score", "metadata"}
        meta = dict(obj.get("metadata", {}))
        for k, v in obj.items():
            if k not in known:
                meta[k] = v
        return Document(
            title=str(title),
            text=str(text),
            source=str(obj.get("source", default_source)),
            metadata=meta,
        )
    return Document(title=str(obj)[:40], text=str(obj), source=default_source)


class CorpusLoader(BaseCorpusLoader):
    """Load documents from files, directories, or in-memory sequences."""

    def load(self, source: Any) -> list[Document]:
        if isinstance(source, (list, tuple)):
            return [_to_document(o, "memory") for o in source]

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Corpus source not found: {source}")

        if path.is_dir():
            return self._load_dir(path)
        return self._load_file(path)

    # -- helpers -----------------------------------------------------------
    def _load_dir(self, path: Path) -> list[Document]:
        """Load every supported file in a directory (mixed types allowed)."""
        docs: list[Document] = []
        for file in sorted(path.rglob("*")):
            if not file.is_file():
                continue
            suffix = file.suffix.lower()
            if suffix in _TEXT_EXT or suffix in _PDF_EXT or suffix in {".json", ".jsonl"}:
                docs.extend(self._load_file(file))
        if not docs:
            raise ValueError(
                f"No supported documents (.txt/.md/.pdf/.json/.jsonl) found under {path}"
            )
        return docs

    def _load_file(self, path: Path) -> list[Document]:
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            return self._load_jsonl(path)
        if suffix == ".json":
            return self._load_json(path)
        if suffix in _PDF_EXT:
            return [self._load_pdf(path)]
        if suffix in _TEXT_EXT:
            return [self._load_text_file(path)]
        raise ValueError(f"Unsupported corpus source: {path}")

    def _load_text_file(self, file: Path) -> Document:
        text = file.read_text(encoding="utf-8", errors="ignore")
        return Document(title=file.stem, text=text, source=str(file))

    def _load_pdf(self, file: Path) -> Document:
        try:
            from pypdf import PdfReader
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "PDF support needs pypdf. Install it with: pip install pypdf"
            ) from e
        reader = PdfReader(str(file))
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:  # noqa: BLE001 - skip unreadable pages
                continue
        text = "\n".join(pages).strip()
        return Document(
            title=file.stem,
            text=text,
            source=str(file),
            metadata={"pages": len(reader.pages), "type": "pdf"},
        )

    def _load_json(self, path: Path) -> list[Document]:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("documents", data.get("data", [data]))
        return [_to_document(o, str(path)) for o in data]

    def _load_jsonl(self, path: Path) -> list[Document]:
        docs: list[Document] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                docs.append(_to_document(json.loads(line), str(path)))
        return docs


def load_corpus(source: Any) -> list[Document]:
    """Convenience function: load a corpus from any supported source."""
    return CorpusLoader().load(source)


__all__ = ["CorpusLoader", "load_corpus"]
