# RAGKit

An extensible toolkit for building advanced **Retrieval-Augmented Generation**
systems — including **multi-agent RAG** — in a few lines of code.

RAGKit hides the internal wiring (embedders, vector stores, retrievers,
agents, LLM providers) behind clean, replaceable interfaces so you can go from
a folder of documents to grounded answers immediately, and swap any component
when you need to.

```python
from ragkit import RagKit

# Pick a provider; the key is read from your .env (GROQ_API_KEY=...).
rag = RagKit(pipeline="ma_rag", llm="groq", model="llama-3.3-70b-versatile")
rag.ingest("./docs")            # a folder, a .pdf, a .json, or a list of docs
print(rag.ask("Explain Retrieval Augmented Generation"))
```

Pass the key explicitly if you prefer:

```python
rag = RagKit(pipeline="ma_rag", llm="groq", api_key="gsk_...",
             model="llama-3.3-70b-versatile")
```

Or use the lower-level factory:

```python
from ragkit import create_pipeline

pipeline = create_pipeline("traditional")
pipeline.ingest("./docs")
print(pipeline.ask("Which library is used as the vector store?"))
```

## Features

- **Provider-first API**: choose `llm="groq" | "openai" | "anthropic" | "google"`;
  keys are read from standard env vars (`GROQ_API_KEY`, `OPENAI_API_KEY`,
  `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`) or passed as `api_key=...`.
- **Two pipelines out of the box**: `traditional` (single-pass) and `ma_rag`
  (planner → step-definer → retriever → extractor → QA → final-answer).
- **Multi-document corpora**: ingest a folder of mixed files, a **PDF**, a
  `.json`/`.jsonl`, or an in-memory list — long docs are auto-chunked.
- **Everything is replaceable**: LLM, embedder, vector store, retriever,
  chunker, corpus loader, prompts — each has an interface and a registry.
- **Multiple backends**: SentenceTransformer / hashing embedders, FAISS store,
  single providers, plus a `provider_manager` (Groq → Cerebras → AWS Bedrock)
  with automatic fallback.
- **Runs offline**: a `MockLLM` + `hashing` embedder let the whole stack run
  with no API keys and no model downloads (used by the test suite).
- **Evaluation built in**: EM, token F1/recall, semantic similarity,
  Precision@K, Recall@K, Hit Rate, MRR, and an LLM-as-a-judge — all writing to
  a dedicated `results/` directory.

## Installation

```cmd
pip install -e .
```

Optional extras (choose what you need):

```cmd
pip install -e ".[embeddings,faiss,ma_rag,llm,eval,dev]"
```

| Extra        | Enables                                             |
|--------------|-----------------------------------------------------|
| `embeddings` | SentenceTransformer embeddings                      |
| `faiss`      | FAISS vector store                                  |
| `ma_rag`     | LangGraph multi-agent pipeline                      |
| `llm`        | Groq / Cerebras / OpenAI / Bedrock provider SDKs    |
| `eval`       | plotting for benchmarks                             |
| `dev`        | pytest                                              |
| `all`        | everything                                          |

## Configuration

Create a `.env` file (see `.env.example`) with the key for your provider —
RAGKit loads it automatically:

```
GROQ_API_KEY=gsk_xxxxxxxx
# OPENAI_API_KEY=sk-xxxxxxxx
# GOOGLE_API_KEY=xxxxxxxx
# ANTHROPIC_API_KEY=sk-ant-xxxxxxxx
```

The `provider_manager` backend (the default when no `llm=` is given) instead
uses the multi-provider fallback and its own variables:

```
GROQ_API_KEYS=key1,key2
CEREBRAS_API_KEYS=key1
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
```

### Try it on a PDF

```cmd
python examples\rag_with_pdf.py
```

## Project layout

See [`docs/architecture.md`](docs/architecture.md) for the component map and
[`docs/developer_guide.md`](docs/developer_guide.md) for how to add a new
backend.

## License

MIT
