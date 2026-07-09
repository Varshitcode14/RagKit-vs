"""Benchmark two pipelines on the same QA dataset and write a comparison.

Offline by default (hashing embedder + MockLLM). Swap in real components for
meaningful numbers. Writes per-pipeline predictions + a comparison JSON to
the results directory.

Run:
    python benchmarks/run_benchmark.py
"""

from __future__ import annotations

import json
from pathlib import Path

from ragkit import RagKit, RagKitConfig
from ragkit.embedders import create_embedder
from ragkit.evaluation import evaluate_predictions, run_inference
from ragkit.llms.mock import MockLLM

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DOCS = ROOT / "examples" / "data" / "docs"
DATASET = ROOT / "examples" / "data" / "qa_dataset.json"
RESULTS = ROOT / "results"


def main() -> None:
    config = RagKitConfig()
    config.embedding.backend = "hashing"
    embedder = create_embedder(config.embedding)

    comparison: dict[str, dict] = {}
    for pipeline_name in ("traditional", "ma_rag"):
        try:
            rag = RagKit(
                pipeline=pipeline_name, corpus=str(DOCS), config=config, llm=MockLLM()
            ).build()
        except Exception as exc:  # e.g. langgraph missing for ma_rag
            print(f"Skipping {pipeline_name}: {exc}")
            continue

        preds = run_inference(
            pipeline=rag.pipeline,
            dataset=str(DATASET),
            pipeline_name=f"bench_{pipeline_name}",
            results_dir=RESULTS,
        )
        comparison[pipeline_name] = evaluate_predictions(preds, k=5, embedder=embedder)

    out = RESULTS / "benchmark_comparison.json"
    out.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    print("\nComparison:")
    for name, metrics in comparison.items():
        print(f"\n{name}")
        for key, value in metrics.items():
            print(f"  {key:<24} {value:.4f}")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
