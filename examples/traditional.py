"""Traditional (single-pass) RAG: retrieve top-k, then answer.

Fast and cheap — one LLM call per question. Set GROQ_API_KEY in your .env.

Run:
    python examples/traditional.py
"""

from pathlib import Path

from ragkit import RagKit

DOCS = Path(__file__).parent / "data" / "docs"


def main() -> None:
    rag = RagKit(pipeline="traditional", llm="groq", corpus=str(DOCS)).build()
    print(rag.ask("What problem does Retrieval Augmented Generation reduce?"))


if __name__ == "__main__":
    main()
