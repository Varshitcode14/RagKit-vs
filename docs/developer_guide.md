# Developer Guide

RAGKit is built so every component is replaceable. To add a backend you write
a class implementing the relevant interface and register it. No pipeline code
changes.

## Add a new LLM

```python
from ragkit.core.interfaces import BaseLLM
from ragkit.core.registry import LLM_REGISTRY

@LLM_REGISTRY.register("my_llm")
class MyLLM(BaseLLM):
    def __init__(self, config=None):
        self.config = config

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        return call_my_model(prompt, temperature)
```

Use it:

```python
from ragkit import RagKit, RagKitConfig
cfg = RagKitConfig()
cfg.llm.backend = "my_llm"
rag = RagKit(pipeline="ma_rag", corpus="./docs", config=cfg).build()
```

Or inject an instance directly (bypasses config/registry):

```python
rag = RagKit(pipeline="ma_rag", corpus="./docs", llm=MyLLM()).build()
```

## Add a new embedder / store / retriever / chunker

Same pattern with the matching interface and registry:

| Component | Interface          | Registry             |
|-----------|--------------------|----------------------|
| Embedder  | `BaseEmbedder`     | `EMBEDDER_REGISTRY`  |
| Store     | `BaseVectorStore`  | `STORE_REGISTRY`     |
| Retriever | `BaseRetriever`    | `RETRIEVER_REGISTRY` |
| Chunker   | `BaseChunker`      | `CHUNKER_REGISTRY`   |
| Pipeline  | `BasePipeline`     | `PIPELINE_REGISTRY`  |

## Add a new pipeline

Subclass `IndexedPipeline` (gets corpus loading + indexing for free) and
implement `run`:

```python
from ragkit.core.registry import PIPELINE_REGISTRY
from ragkit.core.types import RagResponse
from ragkit.pipelines.base import IndexedPipeline

@PIPELINE_REGISTRY.register("my_pipeline")
class MyPipeline(IndexedPipeline):
    name = "my_pipeline"

    def run(self, question: str) -> RagResponse:
        self._ensure_indexed()
        docs = self.retriever.search(question)
        answer = self.llm.generate(self.prompt_builder.build_context(docs))
        return RagResponse(question=question, answer=answer,
                           retrieved_docs=docs, pipeline=self.name)
```

Register it by importing it (add to `pipelines/__init__.py`).

## Running tests

```cmd
pytest
```

The suite is fully offline (MockLLM + hashing embedder). Tests that need FAISS
or LangGraph are skipped automatically if those extras are not installed.

## Design rules

- No side effects at import time (no model loading, no provider construction).
- Components are lazy and injectable.
- Pipelines depend on interfaces, never on concrete backends.
- Evaluation artifacts go to `results/`, never mixed with source.
