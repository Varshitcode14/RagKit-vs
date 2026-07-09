"""Using OpenAI as the LLM provider.

The key is read from OPENAI_API_KEY (env / .env), or pass api_key=... explicitly.
Install: pip install "ragkit-vs[embeddings,faiss,ma_rag,openai]"

Run:
    python examples/openai.py
"""

from pathlib import Path

from ragkit import RagKit

DOCS = Path(__file__).parent / "data" / "docs"


def main() -> None:
    rag = RagKit(
        pipeline="traditional",
        llm="openai",
        model="gpt-4o-mini",
        # api_key="sk-...",   # optional; otherwise read from OPENAI_API_KEY
        corpus=str(DOCS),
    ).build()
    print(rag.ask("What is Retrieval Augmented Generation?"))


if __name__ == "__main__":
    main()
