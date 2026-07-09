"""Retriever backends."""

from ragkit.core.interfaces import BaseRetriever
from ragkit.retrievers.dense import DenseRetriever

__all__ = ["BaseRetriever", "DenseRetriever"]
