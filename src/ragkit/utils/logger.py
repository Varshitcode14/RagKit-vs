"""Lightweight logging helpers.

A thin wrapper around the stdlib ``logging`` module plus a small agent
tracer ported from the original project (used for verbose MA-RAG traces).
"""

from __future__ import annotations

import logging
import time

_CONFIGURED = False


def get_logger(name: str = "ragkit") -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
        root = logging.getLogger("ragkit")
        root.addHandler(handler)
        root.setLevel(logging.WARNING)
        _CONFIGURED = True
    return logging.getLogger(name)


class AgentLogger:
    """Verbose step tracer for agent execution (opt-in)."""

    def __init__(self, agent_name: str, enabled: bool = False) -> None:
        self.agent_name = agent_name
        self.enabled = enabled
        self.start: float | None = None

    def begin(self, task: str) -> None:
        if not self.enabled:
            return
        self.start = time.time()
        print("\n" + "=" * 70)
        print(f"[agent] {self.agent_name}")
        print("=" * 70)
        print(f"Task : {task}\n")

    def end(self, output: str = "") -> None:
        if not self.enabled or self.start is None:
            return
        elapsed = time.time() - self.start
        print("\n" + "-" * 70)
        if output:
            print(output)
        print(f"\nCompleted in : {elapsed:.2f} sec")
        print("=" * 70)
