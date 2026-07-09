"""Sample: run RAGKit over a PDF (or a folder of mixed documents).

Provider-first API. RAGKit reads your key from the environment / a .env file,
so you only need:

    .env
    ----
    GROQ_API_KEY=gsk_xxxxxxxx

Then run:
    python examples/rag_with_pdf.py

Swap the provider by changing llm="groq" to "openai" / "anthropic" / "google"
and setting the matching key (OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY).
"""

from pathlib import Path

from ragkit import RagKit

HERE = Path(__file__).parent

# --- Corpus: a single PDF, OR a whole folder of docs (.pdf/.txt/.md/.json) ---
PDF = HERE / "System Design Interview by Alex Xu.pdf"
CORPUS = str(PDF)                 # single PDF
# CORPUS = str(HERE / "data" / "docs")   # <- or a folder with multiple documents


def main() -> None:
    # llm="groq" + model=...  ->  key comes from GROQ_API_KEY in your .env
    rag = RagKit(
        pipeline="ma_rag",              # or "traditional"
        llm="groq",
        model="llama-3.3-70b-versatile",
        top_k=5,
        # api_key="gsk_...",            # optional: pass the key explicitly instead
    )

    print(f"Ingesting: {CORPUS}")
    rag.ingest(CORPUS)
    print("Index ready.\n")

    questions = [
        "What is this document about?",
        "Explain how to design a rate limiter.",
    ]
    for q in questions:
        print("=" * 70)
        print("Q:", q)
        response = rag.query(q)
        print("A:", response.answer)
        print("Sources:", response.retrieved_titles)
        print("Reasoning steps:", response.reasoning_steps)
        print()


if __name__ == "__main__":
    main()
