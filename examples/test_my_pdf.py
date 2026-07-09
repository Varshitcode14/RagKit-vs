"""One-off test: run RAGKit over the attached System Design PDF with a REAL LLM.

This uses the default `provider_manager` backend, which reads credentials from
the environment. Since the available credentials here are AWS Bedrock, the
provider chain (Groq -> Cerebras -> Bedrock) falls through to Bedrock.

The reference project's .env (with the AWS keys) is loaded explicitly so you
don't have to copy it into this folder.

Run:
    python examples/test_my_pdf.py
"""

from pathlib import Path

from dotenv import load_dotenv

# Load credentials from the reference project's .env (AWS Bedrock keys live there).
_REF_ENV = Path(r"e:\kiran kumar\Desktop\RagKit\Varshit - Multi Agent RAG Project\.env")
if _REF_ENV.exists():
    load_dotenv(_REF_ENV)

from ragkit import RagKit  # noqa: E402

HERE = Path(__file__).parent
PDF = HERE / "System Design Interview by Alex Xu.pdf"


def main() -> None:
    # traditional = single LLM call (fast/cheap) for a quick smoke test.
    rag = RagKit(pipeline="traditional", top_k=4)

    print(f"Ingesting: {PDF.name}  (embeds the PDF once, may take ~1 min) ...")
    rag.ingest(str(PDF))
    print("Index ready.\n")

    question = "How do you design a rate limiter?"
    print("=" * 70)
    print("Q:", question)
    resp = rag.query(question)
    print("A:", resp.answer)
    print("\nSources:", resp.retrieved_titles)
    print(f"Timing: retrieval={resp.retrieval_time:.2f}s  generation={resp.generation_time:.2f}s")


if __name__ == "__main__":
    main()
