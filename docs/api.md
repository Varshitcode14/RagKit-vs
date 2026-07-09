# API Reference

## `ragkit.RagKit`

High-level facade.

```python
RagKit(pipeline="ma_rag", corpus=None, config=None, **components)
```

- `pipeline`: registered pipeline name (`"ma_rag"`, `"traditional"`).
- `corpus`: directory, json/jsonl file, or list of dicts (optional here;
  can be passed to `build`).
- `config`: `RagKitConfig` or a plain dict.
- `**components`: inject `llm`, `embedder`, `chunker`, `store`, `retriever`,
  `prompt_builder`.

Methods:

| Method                    | Description                                   |
|---------------------------|-----------------------------------------------|
| `build(corpus=None)`      | Ingest + index the corpus. Returns `self`.    |
| `ingest(corpus)`          | Ingest an additional corpus. Returns `self`.  |
| `query(question)`         | Returns a `RagResponse`.                      |
| `ask(question)`           | Returns the answer `str`.                     |
| `save_index(path)`        | Persist the vector index.                     |
| `load_index(path)`        | Load a persisted index.                       |
| `pipeline`                | The underlying `BasePipeline`.                |

## `ragkit.create_pipeline(name, config=None, **components)`

Build a pipeline directly. Same component-injection kwargs as `RagKit`.
`ragkit.available_pipelines()` lists registered names.

## `ragkit.RagResponse`

| Field              | Type        | Meaning                              |
|--------------------|-------------|--------------------------------------|
| `question`         | `str`       | The input question                   |
| `answer`           | `str`       | Final answer (`str(response)` = this)|
| `retrieved_docs`   | `list[dict]`| Retrieved documents (with `score`)   |
| `retrieved_titles` | `list[str]` | Titles of retrieved docs             |
| `context`          | `str`       | Concatenated context used            |
| `retrieval_time`   | `float`     | Seconds spent retrieving             |
| `generation_time`  | `float`     | Seconds spent generating             |
| `total_time`       | `float`     | End-to-end seconds                   |
| `history`          | `list[dict]`| Per-step reasoning (MA-RAG)          |
| `reasoning_steps`  | `int`       | Number of reasoning steps            |
| `pipeline`         | `str`       | Pipeline name                        |

## `ragkit.RagKitConfig`

Aggregates: `embedding`, `store`, `retrieval`, `chunking`, `llm`, `paths`,
`ma_rag`. Each is a dataclass; see `ragkit/core/config.py`. Serialize with
`.to_dict()` / rebuild with `RagKitConfig.from_dict(...)`.

## Evaluation (`ragkit.evaluation`)

```python
from ragkit.evaluation import run_inference, evaluate_predictions, compare

preds = run_inference(pipeline, dataset="qa.json", results_dir="results")
metrics = evaluate_predictions(preds, k=5, embedder=None, judge_llm=None)
```

- `compute_retrieval_metrics(retrieved, relevant, k)` → precision@k, recall@k,
  hit_rate, mrr.
- `compute_generation_metrics(preds, golds, embedder=None)` → exact_match,
  token_f1, token_recall, (optional) semantic_similarity.
