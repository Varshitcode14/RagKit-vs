"""Component registries — the backbone of RAGKit's pluggability.

Every component kind (llm, embedder, store, retriever, chunker, pipeline)
has a name -> factory registry. Backends register themselves with a
decorator, and callers build them by name. This is what lets users (and
``create_pipeline``) swap any component without touching pipeline code.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


class Registry:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._factories: dict[str, Callable[..., Any]] = {}

    def register(self, name: str) -> Callable[[T], T]:
        def deco(obj: T) -> T:
            key = name.lower()
            if key in self._factories:
                raise ValueError(f"{self.kind} '{name}' is already registered.")
            self._factories[key] = obj  # type: ignore[assignment]
            return obj

        return deco

    def register_instance(self, name: str, factory: Callable[..., Any]) -> None:
        self._factories[name.lower()] = factory

    def create(self, name: str, *args: Any, **kwargs: Any) -> Any:
        key = name.lower()
        if key not in self._factories:
            raise KeyError(f"Unknown {self.kind}: '{name}'. Available: {sorted(self._factories)}")
        return self._factories[key](*args, **kwargs)

    def available(self) -> list[str]:
        return sorted(self._factories)

    def __contains__(self, name: str) -> bool:
        return name.lower() in self._factories


# One registry per component kind.
LLM_REGISTRY = Registry("llm")
EMBEDDER_REGISTRY = Registry("embedder")
STORE_REGISTRY = Registry("store")
RETRIEVER_REGISTRY = Registry("retriever")
CHUNKER_REGISTRY = Registry("chunker")
PIPELINE_REGISTRY = Registry("pipeline")
