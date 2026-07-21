"""Fetch the DuckDB documentation corpus.

The corpus is the DuckDB docs site source (github.com/duckdb/duckdb-web), which is Jekyll
markdown with YAML frontmatter. It is fetched, never committed: `data/` is gitignored, and
`--ref` pins a commit so an ingest is reproducible.

    python -m src.corpus fetch
    python -m src.corpus fetch --subset sql --limit 200

The repo ships the docs once per release (`current`, `lts`, `1.0` ... `1.3`) — about 2,500
pages that are roughly seven near-identical copies of ~400. Ingesting all of them would fill
the index with duplicate chunks and make a retrieval eval unanswerable, since "the correct
chunk" would exist in seven versions at once. So a fetch pins ONE version, `current` by
default.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/duckdb/duckdb-web.git"
CACHE = Path("data/.cache/duckdb-web")
DEST = Path("data/corpus/duckdb")
DEFAULT_VERSION = "current"


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{proc.stderr.strip()}")


def clone(ref: str | None = None) -> Path:
    if CACHE.exists():
        print(f"cache hit: {CACHE}")
        return CACHE
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    print(f"cloning {REPO} (shallow) ...")
    _run(["git", "clone", "--depth", "1", "--filter=blob:none", REPO, str(CACHE)])
    if ref:
        _run(["git", "-C", str(CACHE), "fetch", "--depth", "1", "origin", ref])
        _run(["git", "-C", str(CACHE), "checkout", ref])
    return CACHE


def head_sha() -> str:
    out = subprocess.run(
        ["git", "-C", str(CACHE), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return out.stdout.strip()[:12] if out.returncode == 0 else "unknown"


def collect(version: str = DEFAULT_VERSION, subset: str = "", limit: int = 0) -> list[Path]:
    """Find the docs markdown for one release, optionally narrowed further."""
    docs_root = CACHE / "docs"
    if not docs_root.exists():
        raise FileNotFoundError(f"{docs_root} missing — repo layout changed?")

    versions = sorted(p.name for p in docs_root.iterdir() if p.is_dir())
    if version not in versions:
        raise RuntimeError(f"version {version!r} not in docs/ — available: {versions}")

    root = docs_root / version
    files = sorted(p for p in root.rglob("*.md") if p.is_file())
    if subset:
        files = [p for p in files if subset in str(p.relative_to(root))]
    if limit:
        files = files[:limit]
    return files


def fetch(
    version: str = DEFAULT_VERSION, subset: str = "", limit: int = 0, ref: str | None = None
) -> Path:
    clone(ref)
    files = collect(version, subset, limit)
    if not files:
        raise RuntimeError(f"no markdown matched version={version!r} subset={subset!r}")
    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True, exist_ok=True)

    root = CACHE / "docs" / version
    for src in files:
        dst = DEST / src.relative_to(root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    total_kb = sum(p.stat().st_size for p in DEST.rglob("*.md")) / 1024
    print(
        f"corpus: {len(files)} pages, {total_kb:.0f} KB -> {DEST}  "
        f"(duckdb-web@{head_sha()} docs/{version})"
    )
    return DEST


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.corpus")
    sub = parser.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="clone duckdb-web and stage one version of its docs")
    f.add_argument("--version", default=DEFAULT_VERSION, help="docs release dir (default: current)")
    f.add_argument("--subset", default="", help="only paths containing this substring")
    f.add_argument("--limit", type=int, default=0, help="cap number of pages (0 = all)")
    f.add_argument("--ref", default=None, help="pin a commit sha for reproducibility")
    args = parser.parse_args(argv)

    if args.cmd == "fetch":
        fetch(args.version, args.subset, args.limit, args.ref)
    return 0


if __name__ == "__main__":
    sys.exit(main())
