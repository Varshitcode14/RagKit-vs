# Release checklist

Steps to cut a new release of **ragkit-vs**.

## 1. Pre-flight
- [ ] `CHANGELOG.md` updated: move `[Unreleased]` items under the new version + date.
- [ ] Version bumped in `pyproject.toml` (`[project].version`) and
      `src/ragkit/__init__.py` (`__version__`) — they must match.
- [ ] All tests pass: `pytest -q`.
- [ ] Lint clean: `ruff check .`.
- [ ] Docs/examples reflect any user-facing change.

## 2. Build
```bash
python -m pip install --upgrade build twine
python -m build
twine check dist/*
```
- [ ] `twine check` reports `PASSED` for both the wheel and sdist.

## 3. Smoke test the built artifact
```bash
python -m venv /tmp/ragkit-smoke
# activate it, then:
pip install "dist/ragkit_vs-<version>-py3-none-any.whl[faiss,ma_rag]"
python -c "import ragkit; print(ragkit.__version__, ragkit.available_pipelines())"
```

## 4. Publish
```bash
# TestPyPI first (recommended)
twine upload --repository testpypi dist/*
# then the real index
twine upload dist/*
```

## 5. Tag & GitHub release
```bash
git tag v<version>
git push origin v<version>
```
- [ ] Create a GitHub release from the tag with the CHANGELOG notes.

## 6. Post-release
- [ ] Add a fresh `[Unreleased]` section to `CHANGELOG.md`.
- [ ] Verify the PyPI page renders (README, links, classifiers).
