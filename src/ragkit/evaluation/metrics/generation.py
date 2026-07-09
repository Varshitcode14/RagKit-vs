"""Generation metrics.

Pure lexical metrics (EM, token F1, token recall) plus an optional
embedding-based semantic similarity. The semantic scorer accepts an
injected embedder so evaluation stays decoupled from any specific model
(and can run offline with the hashing embedder).
"""

from __future__ import annotations

import string
from collections import Counter

from ragkit.core.interfaces import BaseEmbedder


def _normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _tokenize(text: str) -> list[str]:
    return _normalize(text).split()


def exact_match(prediction: str, gold: str) -> bool:
    return _normalize(prediction) == _normalize(gold)


def token_f1(prediction: str, gold: str) -> float:
    pred, gold_toks = _tokenize(prediction), _tokenize(gold)
    if not pred or not gold_toks:
        return float(pred == gold_toks)
    common = Counter(pred) & Counter(gold_toks)
    num = sum(common.values())
    if num == 0:
        return 0.0
    precision = num / len(pred)
    recall = num / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def token_recall(prediction: str, gold: str) -> float:
    pred, gold_toks = _tokenize(prediction), _tokenize(gold)
    if not gold_toks:
        return 0.0
    common = Counter(pred) & Counter(gold_toks)
    return sum(common.values()) / len(gold_toks)


def semantic_similarity(prediction: str, gold: str, embedder: BaseEmbedder) -> float:
    if not prediction or not gold:
        return 0.0
    pred_vec = embedder.encode(prediction)
    gold_vec = embedder.encode(gold)
    sim = float((pred_vec * gold_vec).sum())
    return max(0.0, min(1.0, sim))


def compute_generation_metrics(
    predictions: list[str],
    gold_answers: list[str],
    embedder: BaseEmbedder | None = None,
) -> dict[str, float]:
    if not predictions:
        base = {"exact_match": 0.0, "token_f1": 0.0, "token_recall": 0.0}
        if embedder is not None:
            base["semantic_similarity"] = 0.0
        return base

    n = len(predictions)
    metrics = {
        "exact_match": sum(float(exact_match(p, g)) for p, g in zip(predictions, gold_answers)) / n,
        "token_f1": sum(token_f1(p, g) for p, g in zip(predictions, gold_answers)) / n,
        "token_recall": sum(token_recall(p, g) for p, g in zip(predictions, gold_answers)) / n,
    }
    if embedder is not None:
        metrics["semantic_similarity"] = sum(
            semantic_similarity(p, g, embedder) for p, g in zip(predictions, gold_answers)
        ) / n
    return metrics
