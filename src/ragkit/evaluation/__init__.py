"""Evaluation: metrics, LLM judge, and a decoupled inference/scoring runner."""

from ragkit.evaluation.metrics import (
    compute_generation_metrics,
    compute_retrieval_metrics,
)
from ragkit.evaluation.runner import (
    run_inference,
    evaluate_predictions,
    compare,
    load_dataset,
)

__all__ = [
    "compute_generation_metrics",
    "compute_retrieval_metrics",
    "run_inference",
    "evaluate_predictions",
    "compare",
    "load_dataset",
]
