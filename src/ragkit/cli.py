"""Minimal command-line interface for RAGKit.

Usage:
    ragkit query --pipeline ma_rag --corpus ./docs "your question"
    ragkit info
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ragkit", description="RAGKit CLI")
    sub = parser.add_subparsers(dest="command")

    q = sub.add_parser("query", help="Ingest a corpus and answer a question.")
    q.add_argument("question")
    q.add_argument("--pipeline", default="ma_rag")
    q.add_argument("--corpus", required=True)
    q.add_argument("--top-k", type=int, default=None)

    sub.add_parser("info", help="Show available pipelines.")

    args = parser.parse_args(argv)

    if args.command == "info":
        from ragkit.pipelines import available_pipelines
        print("Available pipelines:", ", ".join(available_pipelines()))
        return 0

    if args.command == "query":
        from ragkit import RagKit, RagKitConfig
        config = RagKitConfig()
        if args.top_k:
            config.retrieval.top_k = args.top_k
        rag = RagKit(pipeline=args.pipeline, corpus=args.corpus, config=config).build()
        print(rag.query(args.question).answer)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
