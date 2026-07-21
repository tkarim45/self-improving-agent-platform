"""Load source files into Documents.

Milestone 0 handles markdown and plain text, which covers the DuckDB docs corpus. PDF and
HTML loaders land in Milestone 1 behind the same `load_path` entry point.
"""

from __future__ import annotations

import re
from pathlib import Path

from src.types import Document, stable_id

TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".mdx"}

_FRONTMATTER = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)


def strip_frontmatter(text: str) -> tuple[str, dict[str, str]]:
    """Pull YAML frontmatter off a docs page.

    The DuckDB docs site uses Jekyll, so nearly every page starts with a `title:` block. That
    title is the best section label available, so it goes into metadata rather than the body.
    """
    m = _FRONTMATTER.match(text)
    if not m:
        return text, {}
    meta: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip().strip("\"'")
    return text[m.end() :], meta


def load_file(path: Path, tenant: str, root: Path | None = None) -> Document | None:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return None
    raw = path.read_text(encoding="utf-8", errors="replace")
    body, meta = strip_frontmatter(raw)
    if not body.strip():
        return None
    rel = str(path.relative_to(root)) if root else str(path)
    title = meta.get("title") or path.stem.replace("-", " ").replace("_", " ")
    return Document(
        doc_id=stable_id(tenant, rel),
        tenant=tenant,
        text=body,
        source_path=rel,
        title=title,
        metadata=meta,
    )


def load_path(path: str | Path, tenant: str) -> list[Document]:
    """Load one file or every supported file under a directory."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"no such path: {p}")
    if p.is_file():
        doc = load_file(p, tenant, root=p.parent)
        return [doc] if doc else []
    docs = []
    for f in sorted(p.rglob("*")):
        if f.is_file():
            doc = load_file(f, tenant, root=p)
            if doc:
                docs.append(doc)
    return docs
