"""Build an index once, persist it, then reload it without re-embedding.

Runs fully offline (MockLLM + hashing embedder) so it needs no keys.

Run:
    python examples/save_load.py
"""

import tempfile
from pathlib import Path

from ragkit import RagKit, RagKitConfig
from ragkit.llms.mock import MockLLM

DOCS = Path(__file__).parent / "data" / "docs"


def _config() -> RagKitConfig:
    config = RagKitConfig()
    config.embedding.backend = "hashing"
    return config


def main() -> None:
    index_dir = Path(tempfile.mkdtemp()) / "index"

    # 1) Build and save.
    rag = RagKit(pipeline="traditional", corpus=str(DOCS), config=_config(), llm=MockLLM()).build()
    rag.save_index(str(index_dir))
    print("Saved index to:", index_dir)

    # 2) Reload into a fresh instance (no re-embedding of the corpus).
    rag2 = RagKit(pipeline="traditional", config=_config(), llm=MockLLM()).load_index(
        str(index_dir)
    )
    response = rag2.query("What is Retrieval Augmented Generation?")
    print("Answer :", response.answer)
    print("Sources:", response.retrieved_titles)


if __name__ == "__main__":
    main()
