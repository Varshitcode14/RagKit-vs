# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Retrieval quality: prompts now enforce answering **only** from retrieved
  context; when the answer is absent, pipelines respond with a clear
  "The answer is not available in the provided documents." instead of using
  outside knowledge.
- Retrieval post-processing: added a configurable similarity threshold
  (`RetrievalConfig.min_score`) and near-duplicate chunk removal
  (`RetrievalConfig.dedupe` / `dedupe_threshold`). Defaults preserve existing
  behavior (threshold off, dedup on for exact/near-identical chunks only).
- Packaging cleanup and metadata for the first public release.
- PyPI distribution name is `ragkit-vs`; the import name remains `ragkit`.
- Split provider dependencies into per-provider extras
  (`groq`, `openai`, `anthropic`, `google`, `bedrock`, `cerebras`).
  The `llm` extra is retained as a backward-compatible meta-extra.
- Added Ruff configuration and richer PyPI metadata (classifiers, URLs, keywords).

### Added
- `LICENSE` (MIT), `CHANGELOG.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`.
- `py.typed` marker for PEP 561 typing support.

## [0.1.0] - 2026-07-09

### Added
- Initial release.
- High-level `RagKit` facade and `create_pipeline` factory.
- Pipelines: `traditional` (single-pass) and `ma_rag` (multi-agent, multi-hop).
- Provider-first LLM selection: `groq`, `openai`, `anthropic`, `google`,
  plus a `provider_manager` multi-provider fallback and an offline `mock`.
- Embedders: `sentence_transformer` and an offline `hashing` embedder.
- FAISS vector store with build / search / save / load.
- Corpus loaders for folders, `.txt`/`.md`, `.pdf`, `.json`, `.jsonl`, and
  in-memory lists.
- Evaluation metrics (retrieval + generation), an LLM judge, and a runner.
- Documentation, examples, and an offline test suite.

[Unreleased]: https://github.com/Varshitcode14/RagKit-vs/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Varshitcode14/RagKit-vs/releases/tag/v0.1.0
