"""Basic example — runs fully offline, no API keys, no model downloads.

Uses the built-in MockLLM and the deterministic `hashing` embedder so you can
see the end-to-end flow immediately.

Run:
    python examples/basic.py
"""

from pathlib import Path

from ragkit import RagKit, RagKitConfig
from ragkit.llms.mock import MockLLM

DOCS = Path(__file__).parent / "data" / "docs"


def main() -> None:
    config = RagKitConfig()
    config.embedding.backend = "hashing"  # offline, deterministic

    rag = RagKit(
        pipeline="traditional",
        corpus=str(DOCS),
        config=config,
        llm=MockLLM(),
    ).build()

    response = rag.query("What is Retrieval Augmented Generation?")
    print("Answer :", response.answer)
    print("Sources:", response.retrieved_titles)


if __name__ == "__main__":
    main()
