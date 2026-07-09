"""Evaluation runner.

Decouples inference from scoring:
  - run_inference()        -> runs a pipeline over a dataset, saves predictions
  - evaluate_predictions() -> loads predictions, computes metrics
  - compare()              -> evaluates several prediction files side by side

All artifacts are written under a results directory (never mixed with source).
Supports HotpotQA-style and custom {question, answer, supporting_titles} data.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ragkit.core.interfaces import BasePipeline, BaseEmbedder, BaseLLM
from ragkit.evaluation.metrics.generation import compute_generation_metrics
from ragkit.evaluation.metrics.retrieval import compute_retrieval_metrics
from ragkit.evaluation.judge import batch_judge

DEFAULT_RESULTS_DIR = Path("results")


# ── dataset loading ───────────────────────────────────────────────────
def load_dataset(dataset_path: str | Path, num_samples: int | None = None) -> list[dict]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if num_samples:
        data = data[:num_samples]

    out = []
    for s in data:
        if isinstance(s.get("supporting_facts"), dict):
            sup = s["supporting_facts"].get("title", [])
        else:
            sup = s.get("supporting_titles", [])
        out.append({
            "id": s.get("id", ""),
            "question": s["question"],
            "answer": s["answer"],
            "supporting_titles": sup,
        })
    return out


# ── inference ─────────────────────────────────────────────────────────
def run_inference(
    pipeline: BasePipeline,
    dataset: str | Path | list[dict],
    pipeline_name: str | None = None,
    num_samples: int | None = None,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> Path:
    name = pipeline_name or getattr(pipeline, "name", "pipeline")
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"{name}_preds.json"

    samples = load_dataset(dataset, num_samples) if not isinstance(dataset, list) else dataset

    predictions: list[dict] = []
    start = time.time()
    for idx, sample in enumerate(samples, start=1):
        try:
            result = pipeline.run(sample["question"])
            predictions.append({
                "id": sample.get("id", ""),
                "question": sample["question"],
                "gold": sample["answer"],
                "prediction": result.answer,
                "retrieved_titles": result.retrieved_titles,
                "supporting_titles": sample.get("supporting_titles", []),
                "context": result.context,
                "total_time": result.total_time,
                "retrieval_time": result.retrieval_time,
                "generation_time": result.generation_time,
                "reasoning_steps": result.reasoning_steps,
                "pipeline": result.pipeline,
            })
            _save_json(predictions, output_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[{idx}] ERROR: {exc}")
    print(f"Done: {len(predictions)}/{len(samples)} in {time.time() - start:.1f}s -> {output_path}")
    return output_path


# ── scoring ───────────────────────────────────────────────────────────
def evaluate_predictions(
    predictions_path: str | Path,
    k: int = 5,
    embedder: BaseEmbedder | None = None,
    judge_llm: BaseLLM | None = None,
) -> dict:
    with open(predictions_path, "r", encoding="utf-8") as f:
        predictions: list[dict] = json.load(f)
    if not predictions:
        return {}

    preds = [p["prediction"] for p in predictions]
    golds = [p["gold"] for p in predictions]
    retrieved = [p.get("retrieved_titles", []) for p in predictions]
    relevant = [p.get("supporting_titles", []) for p in predictions]

    metrics: dict[str, Any] = {}
    metrics.update(compute_generation_metrics(preds, golds, embedder=embedder))
    metrics.update(compute_retrieval_metrics(retrieved, relevant, k=k))

    def _avg(key: str) -> float:
        vals = [p[key] for p in predictions if isinstance(p.get(key), (int, float))]
        return sum(vals) / len(vals) if vals else 0.0

    for key in ("total_time", "retrieval_time", "generation_time", "reasoning_steps"):
        metrics[f"avg_{key}"] = _avg(key)

    if judge_llm is not None:
        judged = batch_judge(judge_llm, predictions)
        for key in ("judge_correctness_score", "judge_faithfulness_score"):
            scores = [r[key] for r in judged if key in r]
            if scores:
                metrics[key] = sum(scores) / len(scores)
        _save_json(judged, Path(predictions_path).with_suffix(".judged.json"))

    return metrics


def compare(paths: dict[str, str | Path], k: int = 5, **kwargs) -> dict[str, dict]:
    return {label: evaluate_predictions(path, k=k, **kwargs) for label, path in paths.items()}


def _save_json(data, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
