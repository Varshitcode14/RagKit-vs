"""Ingest a folder of documents (mixed .txt/.md/.pdf/.json/.jsonl).

Requires a provider key in your environment / .env (GROQ_API_KEY here).
Install: pip install "ragkit-vs[embeddings,faiss,ma_rag,groq]"

Run:
    python examples/folder.py
"""

from pathlib import Path

from ragkit import RagKit

DOCS = Path(__file__).parent / "data" / "docs"


def main() -> None:
    rag = RagKit(pipeline="traditional", llm="groq", corpus=str(DOCS)).build()
    print(rag.ask("Which library is commonly used as the vector store?"))


if __name__ == "__main__":
    main()
