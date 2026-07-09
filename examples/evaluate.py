"""Evaluation example — run a pipeline over a QA dataset and score it.

Offline by default (hashing embedder + mock LLM). Results are written to
the ``results/`` directory. Swap MockLLM/hashing for real components to get
meaningful scores.

Run:
    python examples/evaluate.py
"""

from pathlib import Path

from ragkit import RagKit, RagKitConfig
from ragkit.embedders import create_embedder
from ragkit.evaluation import evaluate_predictions, run_inference
from ragkit.llms.mock import MockLLM

HERE = Path(__file__).parent
DOCS = HERE / "data" / "docs"
DATASET = HERE / "data" / "qa_dataset.json"
RESULTS = HERE.parent / "results"


def main() -> None:
    config = RagKitConfig()
    config.embedding.backend = "hashing"

    rag = RagKit(pipeline="traditional", corpus=str(DOCS), config=config, llm=MockLLM()).build()

    preds_path = run_inference(
        pipeline=rag.pipeline,
        dataset=str(DATASET),
        pipeline_name="traditional_offline",
        results_dir=RESULTS,
    )

    embedder = create_embedder(config.embedding)  # hashing -> offline semantic score
    metrics = evaluate_predictions(preds_path, k=5, embedder=embedder)

    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key:<24} {value:.4f}")


if __name__ == "__main__":
    main()
