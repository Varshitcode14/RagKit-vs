"""Using Groq as the LLM provider.

The key is read from GROQ_API_KEY (env / .env), or pass api_key=... explicitly.
Install: pip install "ragkit-vs[embeddings,faiss,ma_rag,groq]"

Run:
    python examples/groq.py
"""

from pathlib import Path

from ragkit import RagKit

DOCS = Path(__file__).parent / "data" / "docs"


def main() -> None:
    rag = RagKit(
        pipeline="ma_rag",
        llm="groq",
        model="llama-3.3-70b-versatile",
        # api_key="gsk_...",   # optional; otherwise read from GROQ_API_KEY
        corpus=str(DOCS),
    ).build()
    print(rag.ask("What is Retrieval Augmented Generation?"))


if __name__ == "__main__":
    main()
