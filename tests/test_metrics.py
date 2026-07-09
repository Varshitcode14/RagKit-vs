"""Tests for evaluation metrics (pure functions)."""

from ragkit.evaluation.metrics.retrieval import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    compute_retrieval_metrics,
)
from ragkit.evaluation.metrics.generation import (
    exact_match,
    token_f1,
    token_recall,
    compute_generation_metrics,
)


def test_precision_and_recall():
    retrieved = ["a", "b", "c"]
    relevant = ["a", "c"]
    assert precision_at_k(retrieved, relevant, 3) == 2 / 3
    assert recall_at_k(retrieved, relevant, 3) == 1.0


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "y", "a"], ["a"]) == 1 / 3
    assert reciprocal_rank(["x"], ["a"]) == 0.0


def test_retrieval_batch():
    m = compute_retrieval_metrics([["a", "b"]], [["a"]], k=2)
    assert m["hit_rate"] == 1.0
    assert m["mrr"] == 1.0


def test_exact_match_normalizes():
    assert exact_match("The Answer.", "the answer")


def test_token_f1_and_recall():
    assert token_f1("a b c", "a b c") == 1.0
    assert token_recall("a b", "a b c d") == 0.5


def test_generation_batch_without_embedder():
    m = compute_generation_metrics(["FAISS"], ["FAISS"])
    assert m["exact_match"] == 1.0
    assert "semantic_similarity" not in m
