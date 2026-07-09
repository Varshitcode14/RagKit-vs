# Contributing to RAGKit

Thanks for your interest in improving RAGKit! This guide explains how to set up
a development environment and the conventions we follow.

## Development setup

```bash
git clone https://github.com/Varshitcode14/RagKit-vs
cd RagKit-vs

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# Editable install with all backends + dev tools
pip install -e ".[all]"
```

If you only need the offline path (no model downloads, no API keys), the base
install plus `faiss` is enough to run most of the test suite:

```bash
pip install -e ".[faiss,ma_rag,dev]"
```

## Running the tests

```bash
pytest
```

The suite is designed to run **offline**: it uses a `MockLLM` and a
deterministic `hashing` embedder. Tests that require FAISS or LangGraph are
skipped automatically when those extras are not installed.

## Linting and formatting

We use [Ruff](https://docs.astral.sh/ruff/):

```bash
ruff check .
ruff format .
```

## Conventions

- **Backward compatibility first.** The public API (`RagKit`, `create_pipeline`,
  `RagResponse`, `RagKitConfig`, and registered pipeline/LLM names) must not
  break within a major version.
- **No import-time side effects.** Importing `ragkit` must not load models or
  construct providers. Keep heavy work lazy.
- **Every component is replaceable.** New backends implement an interface in
  `ragkit.core.interfaces` and register themselves in the matching registry.
- **Type hints and docstrings** on public functions and classes.

## Adding a backend

See [`docs/developer_guide.md`](docs/developer_guide.md) for a step-by-step
example of adding an LLM, embedder, store, retriever, chunker, or pipeline.

## Pull requests

1. Create a feature branch.
2. Add or update tests.
3. Ensure `pytest` and `ruff check` pass.
4. Update `CHANGELOG.md` under `[Unreleased]`.
5. Open a PR with a clear description of the change and its motivation.

## Reporting issues

Please use the issue templates and include the RAGKit version, Python version,
OS, a minimal reproduction, and the full error message.
