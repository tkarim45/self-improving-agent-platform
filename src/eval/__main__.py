"""Eval CLI.

    python -m src.eval retrieval
    python -m src.eval retrieval --reranker cross-encoder/ms-marco-MiniLM-L-6-v2 \
        --out eval/retrieval/report.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.eval.retrieval import DEFAULT_SPEC, format_report, run
from src.index.store import HybridIndex


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.eval")
    sub = parser.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("retrieval", help="run the labeled retrieval eval")
    r.add_argument("--tenant", default="duckdb")
    r.add_argument("--spec", default=str(DEFAULT_SPEC))
    r.add_argument("--index-root", default="data/index")
    r.add_argument("--reranker", default="none", help="'none', 'lexical', or a cross-encoder")
    r.add_argument(
        "--corpus",
        default="data/corpus/duckdb",
        help="corpus path, used to build the link graph (empty disables graph arms)",
    )
    r.add_argument("--out", default="", help="write the markdown report here")
    args = parser.parse_args(argv)

    if args.cmd == "retrieval":
        results, queries, spec = run(
            tenant=args.tenant,
            spec_path=Path(args.spec),
            index_root=args.index_root,
            reranker_name=args.reranker,
            corpus_path=args.corpus,
        )
        index = HybridIndex.load(args.tenant, root=args.index_root)
        report = format_report(results, queries, spec, index)
        print(report)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(report, encoding="utf-8")
            print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
