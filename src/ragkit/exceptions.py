"""RAGKit exception hierarchy.

All RAGKit-specific errors derive from :class:`RagKitError`, so callers can do
``except RagKitError`` to catch anything the library raises. Where a more
specific built-in previously escaped (e.g. ``ValueError`` for a missing API
key, ``FileNotFoundError`` for a bad path, ``RuntimeError`` for an unindexed
pipeline), the corresponding RAGKit error *also* subclasses that built-in so
existing ``except ValueError`` / ``except FileNotFoundError`` code keeps working.
"""

from __future__ import annotations


class RagKitError(Exception):
    """Base class for every error raised by RAGKit."""


class ConfigurationError(RagKitError):
    """Invalid or inconsistent configuration."""


class MissingDependencyError(RagKitError, ImportError):
    """An optional backend dependency is not installed.

    Carries the pip ``extra`` that provides it so the message is actionable.
    """

    def __init__(self, package: str, extra: str | None = None) -> None:
        hint = f'pip install "ragkit-vs[{extra}]"' if extra else f"pip install {package}"
        super().__init__(f"Missing optional dependency '{package}'. Install it with: {hint}")
        self.package = package
        self.extra = extra


class MissingAPIKeyError(RagKitError, ValueError):
    """No API key was provided for the selected LLM provider."""


# --- corpus / ingestion ------------------------------------------------------
class CorpusError(RagKitError):
    """Something went wrong while loading a corpus."""


class CorpusNotFoundError(CorpusError, FileNotFoundError):
    """The corpus path does not exist."""


class UnsupportedFileTypeError(CorpusError, ValueError):
    """The corpus contains a file type RAGKit cannot load."""


class EmptyCorpusError(CorpusError, ValueError):
    """The corpus resolved to zero documents."""


# --- components --------------------------------------------------------------
class EmbeddingError(RagKitError):
    """The embedding model failed to load or encode."""


class VectorStoreError(RagKitError):
    """The vector store failed to build, search, save, or load."""


class LLMError(RagKitError):
    """The language model failed to produce a response."""


# --- pipeline ----------------------------------------------------------------
class PipelineError(RagKitError):
    """A pipeline-level error."""


class NotIndexedError(PipelineError, RuntimeError):
    """A query was issued before any corpus was ingested / index loaded."""


__all__ = [
    "RagKitError",
    "ConfigurationError",
    "MissingDependencyError",
    "MissingAPIKeyError",
    "CorpusError",
    "CorpusNotFoundError",
    "UnsupportedFileTypeError",
    "EmptyCorpusError",
    "EmbeddingError",
    "VectorStoreError",
    "LLMError",
    "PipelineError",
    "NotIndexedError",
]
