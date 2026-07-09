"""Ingest a single PDF and ask questions about it.

Install: pip install "ragkit-vs[embeddings,faiss,ma_rag,groq,pdf]"
Set GROQ_API_KEY in your environment / .env.

Run:
    python examples/pdf.py
"""

from ragkit import RagKit

# Point this at your own PDF.
PDF_PATH = "path/to/your/document.pdf"


def main() -> None:
    rag = RagKit(pipeline="ma_rag", llm="groq", corpus=PDF_PATH).build()
    response = rag.query("What is this document about?")
    print("Answer :", response.answer)
    print("Sources:", response.retrieved_titles)


if __name__ == "__main__":
    main()
