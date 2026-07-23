"""Ingest CLI: parse -> chunk -> embed -> BM25 + FAISS.

    python -m src.ingest data/corpus/duckdb --tenant duckdb
    python -m src.ingest data/corpus/duckdb/sql/select.md --tenant duckdb --query "QUALIFY"
"""

from __future__ import annotations

import argparse
import sys
import time

from src.index.store import HybridIndex
from src.ingest.chunker import HeadingChunker
from src.ingest.loaders import load_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.ingest")
    parser.add_argument("path", help="file or directory to ingest")
    parser.add_argument("--tenant", default="duckdb")
    parser.add_argument(
        "--embedder",
        default="hashing",
        help="'hashing' (offline, default) or a sentence-transformers model name",
    )
    parser.add_argument("--index-root", default="data/index")
    parser.add_argument("--max-chars", type=int, default=1800)
    parser.add_argument("--min-chars", type=int, default=300)
    parser.add_argument("--rebuild", action="store_true", help="ignore any existing index")
    parser.add_argument("--query", default="", help="run one smoke query after ingesting")
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    docs = load_path(args.path, args.tenant)
    if not docs:
        print(f"nothing loadable at {args.path}", file=sys.stderr)
        return 1

    chunker = HeadingChunker(max_chars=args.max_chars, min_chars=args.min_chars)
    chunks = [c for doc in docs for c in chunker.chunk(doc)]

    index = None
    if not args.rebuild:
        try:
            index = HybridIndex.load(args.tenant, root=args.index_root)
            print(f"loaded existing index: {len(index)} chunks")
        except (FileNotFoundError, ValueError):
            index = None
    if index is None:
        index = HybridIndex(args.tenant, embedder_name=args.embedder, root=args.index_root)

    added = index.add(chunks)
    path = index.save()
    elapsed = time.perf_counter() - t0

    print(
        f"ingested {len(docs)} docs -> {len(chunks)} chunks "
        f"({added} new, {len(chunks) - added} already indexed) in {elapsed:.1f}s"
    )
    print(f"index: {len(index)} chunks, embedder={index.embedder_name} -> {path}")

    if args.query:
        print(f"\nsmoke query: {args.query!r}")
        for label, retriever in (("bm25", index.sparse), ("dense", index.dense)):
            hits = retriever.search(args.query, args.tenant, k=3)
            print(f"  {label}:")
            for hit in hits:
                head = " > ".join(hit.chunk.heading_path) or hit.chunk.source_path
                preview = " ".join(hit.chunk.text.split())[:90]
                print(f"    {hit.score:7.3f}  {head}\n              {preview}...")
            if not hits:
                print("    (no hits)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
