# RAGKit Architecture

## Package layout

```
src/ragkit/
  core/          # types, interfaces, config, registries (the contracts)
  llms/          # LLM backends: provider_manager (Groq->Cerebras->Bedrock), mock
  embedders/     # sentence_transformer, hashing (offline)
  stores/        # faiss vector store (build / add / search / save / load)
  chunkers/      # passage / no-op
  corpus/        # loaders: dir of txt|md, json, jsonl, in-memory
  retrievers/    # dense (embedder + store)
  prompts/       # templates + PromptBuilder
  pipelines/     # base (indexing), traditional, ma_rag/ (graph, agents, state)
  evaluation/    # metrics/ (retrieval, generation), judge, runner
  utils/         # logging
  facade.py      # RagKit high-level API
  cli.py         # `ragkit` command line
examples/        # offline_demo, quickstart, evaluate + sample data
tests/           # offline unit + integration tests
benchmarks/      # benchmark runner
results/         # evaluation outputs (git-ignored)
docs/            # this documentation
```

## Layered design

```
                    +-------------------+
   user code  --->  |     RagKit        |   facade (build / query / ask)
                    +-------------------+
                             |
                    +-------------------+
                    |   create_pipeline |   registry lookup by name
                    +-------------------+
                             |
              +--------------+---------------+
              |                              |
     +-----------------+          +--------------------------+
     |  Traditional    |          |        MA-RAG            |
     |  RAG pipeline   |          |  (LangGraph state machine)|
     +-----------------+          +--------------------------+
              |                              |
              +--------------+---------------+
                             |
        IndexedPipeline: loader -> chunker -> embedder -> store -> retriever
                             |
        +----------+----------+-----------+-----------+
        |          |          |           |           |
      LLM      Embedder    Store      Retriever    Chunker      (all via interfaces)
```

## Interfaces (`core/interfaces.py`)

Every replaceable component implements an abstract base class:

- `BaseLLM.generate(prompt, temperature) -> str`
- `BaseEmbedder.encode(text) / encode_batch(texts) / dimension`
- `BaseVectorStore.add / search / save / load / __len__`
- `BaseRetriever.search(query, top_k)`
- `BaseChunker.split(document)`
- `BaseCorpusLoader.load(source)`
- `BasePipeline.ingest(corpus) / run(question) -> RagResponse`

## Registries (`core/registry.py`)

Each component kind has a name→factory registry. Backends register with a
decorator (`@LLM_REGISTRY.register("mock")`) and are built by name from
`RagKitConfig`. This is what makes `create_pipeline("ma_rag")` and per-component
swapping possible without editing pipeline code.

## MA-RAG control flow

```
planner
  -> set_goal -> step_definer -> retriever -> extractor -> qa
     -> update_history -> next_step --(more steps?)--> set_goal
                                     --(done)--------> final_answer -> END
```

Titles and documents are accumulated across **all** reasoning steps
(`all_retrieved_titles` / `all_retrieved_docs`) so multi-hop retrieval metrics
are valid — a fix carried over from the reference implementation.

## Key differences from the research reference

- No import-time singletons: components are lazy and injected, so importing
  `ragkit` never loads a model or constructs a provider.
- Config is explicit (`RagKitConfig`) instead of scattered module constants.
- The FAISS store can build/persist indexes, not just load a prebuilt one.
- The whole stack is runnable offline for tests via `MockLLM` + `hashing`.
