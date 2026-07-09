"""Fully offline demo — no API keys, no model downloads.

Uses the MockLLM and the HashingEmbedder so the entire multi-agent pipeline
runs end-to-end anywhere. Great for verifying the install and architecture.

Run:
    python examples/offline_demo.py
"""

from pathlib import Path

from ragkit import RagKit, RagKitConfig
from ragkit.llms.mock import MockLLM

DOCS = Path(__file__).parent / "data" / "docs"


def main() -> None:
    # Offline config: deterministic hashing embedder + mock LLM.
    config = RagKitConfig()
    config.embedding.backend = "hashing"

    for pipeline in ("traditional", "ma_rag"):
        print("\n" + "=" * 60)
        print(f"Pipeline: {pipeline}")
        print("=" * 60)

        rag = RagKit(
            pipeline=pipeline,
            corpus=str(DOCS),
            config=config,
            llm=MockLLM(),
        ).build()

        response = rag.query("What is Retrieval Augmented Generation?")
        print("Answer          :", response.answer)
        print("Retrieved titles:", response.retrieved_titles)
        print("Reasoning steps :", response.reasoning_steps)


if __name__ == "__main__":
    main()
