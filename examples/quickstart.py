"""Quickstart — the few-lines-of-code experience (real models + LLM).

Requires:
  - the embeddings extra:  pip install -e ".[embeddings,faiss,ma_rag,llm]"
  - provider API keys in the environment / a .env file, e.g. GROQ_API_KEYS

Run:
    python examples/quickstart.py
"""

from pathlib import Path

from ragkit import RagKit

DOCS = Path(__file__).parent / "data" / "docs"


def main() -> None:
    rag = RagKit(pipeline="ma_rag", corpus=str(DOCS)).build()
    response = rag.query("Explain Retrieval Augmented Generation.")
    print("Q:", response.question)
    print("A:", response.answer)
    print("Steps:", response.reasoning_steps)
    print("Titles:", response.retrieved_titles)


if __name__ == "__main__":
    main()
