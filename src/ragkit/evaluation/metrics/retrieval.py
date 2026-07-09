"""Retrieval metrics (pure functions, ported unchanged in behavior)."""

from __future__ import annotations


def precision_at_k(retrieved: list[str], relevant: list[str], k: int | None = None) -> float:
    if not retrieved:
        return 0.0
    top_k = retrieved[:k] if k else retrieved
    rel = set(relevant)
    return sum(1 for t in top_k if t in rel) / len(top_k)


def recall_at_k(retrieved: list[str], relevant: list[str], k: int | None = None) -> float:
    if not relevant:
        return 0.0
    top_k = retrieved[:k] if k else retrieved
    rel = set(relevant)
    return sum(1 for t in top_k if t in rel) / len(rel)


def hit_rate(
    retrieved_list: list[list[str]], relevant_list: list[list[str]], k: int | None = None
) -> float:
    if not retrieved_list:
        return 0.0
    hits = 0
    for retrieved, relevant in zip(retrieved_list, relevant_list):
        top_k = retrieved[:k] if k else retrieved
        if any(t in set(relevant) for t in top_k):
            hits += 1
    return hits / len(retrieved_list)


def reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    rel = set(relevant)
    for rank, title in enumerate(retrieved, start=1):
        if title in rel:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(retrieved_list: list[list[str]], relevant_list: list[list[str]]) -> float:
    if not retrieved_list:
        return 0.0
    total = sum(reciprocal_rank(r, rel) for r, rel in zip(retrieved_list, relevant_list))
    return total / len(retrieved_list)


def compute_retrieval_metrics(
    retrieved_list: list[list[str]],
    relevant_list: list[list[str]],
    k: int | None = None,
) -> dict[str, float]:
    n = len(retrieved_list)
    if n == 0:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "hit_rate": 0.0, "mrr": 0.0}
    precision = [precision_at_k(r, rel, k) for r, rel in zip(retrieved_list, relevant_list)]
    recall = [recall_at_k(r, rel, k) for r, rel in zip(retrieved_list, relevant_list)]
    return {
        "precision_at_k": sum(precision) / n,
        "recall_at_k": sum(recall) / n,
        "hit_rate": hit_rate(retrieved_list, relevant_list, k),
        "mrr": mean_reciprocal_rank(retrieved_list, relevant_list),
    }
