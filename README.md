# RAGKit

**Build advanced Retrieval-Augmented Generation (RAG) systems — including
multi-agent RAG — in a few lines of code.**

RAGKit hides the plumbing (embedders, vector stores, retrievers, agents, LLM
providers) behind clean, replaceable interfaces so you can go from a folder of
documents to grounded answers immediately, and swap any component when you need
to.

> PyPI distribution name: **`ragkit-vs`** · import name: **`ragkit`**

```python
from ragkit import RagKit

rag = RagKit(
    pipeline="ma_rag",
    llm="groq",
    model="llama-3.3-70b-versatile",
    corpus="docs/",
).build()

response = rag.query("What is RAG?")
print(response.answer)
```

## Table of contents
- [Features](#features)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Examples](#examples)
- [Supported pipelines](#supported-pipelines)
- [Supported LLM providers](#supported-llm-providers)
- [Supported document formats](#supported-document-formats)
- [Configuration](#configuration)
- [Error handling](#error-handling)
- [Contributing](#contributing)
- [FAQ](#faq)
- [Roadmap](#roadmap)
- [License](#license)

## Features
- **Provider-first, few-line API** — pick a provider, point at a corpus, ask.
- **Two pipelines**: `traditional` (single-pass) and `ma_rag` (planner →
  step-definer → retriever → extractor → QA → final-answer, multi-hop).
- **Grounded-only answering** — prompts force answers from retrieved context;
  when the answer isn't present, RAGKit says so instead of hallucinating.
- **Configurable retrieval quality** — similarity threshold + near-duplicate
  chunk removal.
- **Everything is replaceable** — LLM, embedder, vector store, retriever,
  chunker, corpus loader, and prompts each have an interface + registry.
- **Runs offline** — a `MockLLM` + `hashing` embedder let the whole stack run
  with no API keys and no model downloads (used by the test suite).
- **Typed** (ships `py.typed`) and **lightweight** by default.

## Installation

```bash
pip install ragkit-vs
```

The base install is tiny. Install only the extras you need:

| Extra | Enables | Example |
|-------|---------|---------|
| `embeddings` | SentenceTransformer embeddings | `pip install "ragkit-vs[embeddings]"` |
| `faiss` | FAISS vector store | `pip install "ragkit-vs[faiss]"` |
| `ma_rag` | Multi-agent pipeline (LangGraph) | `pip install "ragkit-vs[ma_rag]"` |
| `pdf` | PDF ingestion (pypdf) | `pip install "ragkit-vs[pdf]"` |
| `groq` | Groq provider | `pip install "ragkit-vs[groq]"` |
| `openai` | OpenAI provider | `pip install "ragkit-vs[openai]"` |
| `anthropic` | Anthropic provider | `pip install "ragkit-vs[anthropic]"` |
| `google` | Google Gemini provider | `pip install "ragkit-vs[google]"` |
| `bedrock` | AWS Bedrock (boto3) | `pip install "ragkit-vs[bedrock]"` |
| `cerebras` | Cerebras provider | `pip install "ragkit-vs[cerebras]"` |
| `all` | everything (dev) | `pip install "ragkit-vs[all]"` |

Typical setup for a Groq-powered RAG over PDFs:

```bash
pip install "ragkit-vs[embeddings,faiss,ma_rag,groq,pdf]"
```

## Quick start

1. Put your provider key in a `.env` file (see [`.env.example`](.env.example)):
   ```
   GROQ_API_KEY=gsk_xxxxxxxx
   ```
2. Ask questions over your documents:
   ```python
   from ragkit import RagKit

   rag = RagKit(pipeline="ma_rag", llm="groq", corpus="docs/").build()
   print(rag.ask("Explain Retrieval Augmented Generation"))
   ```

No key? Run fully offline:

```python
from ragkit import RagKit, RagKitConfig
from ragkit.llms.mock import MockLLM

cfg = RagKitConfig()
cfg.embedding.backend = "hashing"   # deterministic, no downloads

rag = RagKit(pipeline="traditional", corpus="docs/", config=cfg, llm=MockLLM()).build()
print(rag.ask("What is in my documents?"))
```

## Examples
Runnable scripts live in [`examples/`](examples):

| File | What it shows |
|------|---------------|
| `basic.py` | Offline end-to-end (no keys) |
| `traditional.py` | Single-pass pipeline |
| `ma_rag.py` | Multi-agent pipeline |
| `folder.py` | Ingest a folder of mixed documents |
| `pdf.py` | Ingest a PDF |
| `groq.py` / `openai.py` | Choosing a provider |
| `save_load.py` | Persist and reload an index |

## Supported pipelines
- **`traditional`** — retrieve top-k chunks, build context, one LLM call.
  Fast and cheap.
- **`ma_rag`** — decomposes the question into steps, retrieves and reasons per
  step, then synthesizes a final answer. Better for multi-hop questions.

List them at runtime: `ragkit.available_pipelines()`.

## Supported LLM providers
`groq`, `openai`, `anthropic`, `google` (Gemini), plus a `provider_manager`
multi-provider fallback and an offline `mock`. Keys are read from standard
environment variables (`GROQ_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`,
`ANTHROPIC_API_KEY`) or passed via `api_key=...`.

## Supported document formats
Folders of mixed files, plus individual files: `.txt`, `.md`, `.pdf`,
`.json`, `.jsonl`, or an in-memory list of `{"title", "text"}` dicts. Long
documents are automatically chunked.

## Configuration
Everything is tunable through `RagKitConfig` (all fields have sensible
defaults):

```python
from ragkit import RagKit, RagKitConfig

cfg = RagKitConfig()
cfg.retrieval.top_k = 8
cfg.retrieval.min_score = 0.3     # ignore weak retrievals
cfg.chunking.chunk_size = 300     # words per chunk (0 = whole document)

rag = RagKit(pipeline="ma_rag", llm="groq", corpus="docs/", config=cfg).build()
```

Persist and reload an index:

```python
rag.save_index(".ragkit/my_index")
rag2 = RagKit(pipeline="traditional", llm="groq").load_index(".ragkit/my_index")
```

## Error handling
RAGKit raises clear, typed exceptions (all subclass `ragkit.RagKitError`):

```python
from ragkit import RagKit
from ragkit.exceptions import MissingAPIKeyError, CorpusNotFoundError

try:
    RagKit(pipeline="ma_rag", llm="groq", corpus="docs/").build()
except MissingAPIKeyError as e:
    print("Set your API key:", e)
except CorpusNotFoundError as e:
    print("Bad corpus path:", e)
```

For backward compatibility these also subclass the relevant built-ins
(`MissingAPIKeyError` is a `ValueError`, `CorpusNotFoundError` is a
`FileNotFoundError`, `NotIndexedError` is a `RuntimeError`, etc.).

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md). In short:

```bash
pip install -e ".[all]"
pytest
ruff check .
```

The test suite runs offline (mock LLM + hashing embedder).

## FAQ
**Do I need API keys to try it?** No — use the `mock` LLM + `hashing`
embedder (see `examples/basic.py`).

**Why `ragkit-vs` on PyPI but `import ragkit`?** The distribution name is
`ragkit-vs` (the `ragkit` name was taken); the import name stays `ragkit`.

**Does it work on Colab / Linux / macOS / Windows?** Yes — all backends ship
prebuilt wheels for the major platforms.

**How do I avoid installing torch?** Only the `embeddings` extra pulls torch.
Use the `hashing` embedder for a torch-free setup.

## Roadmap
- Hybrid (dense + sparse) retrieval and reranking
- Automatic, persistent index caching
- Additional vector store backends
- CLI and optional API server

## License
[MIT](LICENSE) © K Varshit
