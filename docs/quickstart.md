# Quickstart

## 1. Install

```cmd
pip install -e ".[embeddings,faiss,ma_rag,llm]"
```

## 2. Provide API keys (for real LLM answers)

Create a `.env` file (or set environment variables):

```
GROQ_API_KEYS=your_key_here
```

RAGKit falls back Groq → Cerebras → AWS Bedrock automatically.

## 3. Ask a question over your documents

```python
from ragkit import RagKit

rag = RagKit(pipeline="ma_rag", corpus="./docs").build()
response = rag.query("Explain Retrieval Augmented Generation")

print(response.answer)
print(response.retrieved_titles)
print(response.reasoning_steps)
```

`corpus` can be a directory of `.txt`/`.md` files, a `.json`/`.jsonl` file, or
an in-memory list of `{"title", "text"}` dicts.

## 4. No keys? Run fully offline

```python
from ragkit import RagKit, RagKitConfig
from ragkit.llms.mock import MockLLM

cfg = RagKitConfig()
cfg.embedding.backend = "hashing"   # deterministic, no downloads

rag = RagKit(pipeline="traditional", corpus="./docs", config=cfg, llm=MockLLM()).build()
print(rag.ask("What is in my documents?"))
```

## 5. Persist and reuse an index

```python
rag.save_index(".ragkit/my_index")
# later ...
rag2 = RagKit(pipeline="traditional").load_index(".ragkit/my_index")
```

## 6. Command line

```cmd
ragkit info
ragkit query --pipeline traditional --corpus ./docs "your question"
```
