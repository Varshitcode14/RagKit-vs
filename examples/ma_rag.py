"""Multi-Agent RAG: plan -> retrieve -> extract -> answer per step -> synthesize.

Better for multi-hop questions. Set GROQ_API_KEY in your .env.
Install: pip install "ragkit-vs[embeddings,faiss,ma_rag,groq]"

Run:
    python examples/ma_rag.py
"""

from pathlib import Path

from ragkit import RagKit

DOCS = Path(__file__).parent / "data" / "docs"


def main() -> None:
    rag = RagKit(pipeline="ma_rag", llm="groq", corpus=str(DOCS)).build()
    response = rag.query("What is the key advantage of Multi-Agent RAG over single-pass RAG?")
    print("Answer         :", response.answer)
    print("Reasoning steps:", response.reasoning_steps)
    print("Sources        :", response.retrieved_titles)


if __name__ == "__main__":
    main()
