"""Milestone 0 acceptance: a document goes in, a query gets the right chunk back."""

from __future__ import annotations

from src.index.store import HybridIndex
from src.ingest import HeadingChunker, load_path
from src.ingest.__main__ import main

PAGE = """---
title: ASOF Join
---

ASOF joins are for time series data.

## Motivation

Often you want the most recent price at or before a trade's timestamp. An equality join
cannot express that, because the exact timestamp rarely matches.

## Syntax

Use ASOF JOIN with an inequality condition on the ordering column.

```sql
SELECT t.id, p.price
FROM trades t ASOF JOIN prices p ON t.ts >= p.ts;
```
"""

OTHER = """---
title: COPY Statement
---

## Writing files

COPY exports a table to Parquet or CSV on local disk.
"""


def test_end_to_end_ingest_then_retrieve(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "asof.md").write_text(PAGE)
    (corpus / "copy.md").write_text(OTHER)

    docs = load_path(corpus, tenant="duckdb")
    chunks = [c for d in docs for c in HeadingChunker().chunk(d)]
    index = HybridIndex("duckdb", root=tmp_path / "index")
    assert index.add(chunks) == len(chunks) > 0
    index.save()

    hits = index.sparse.search("most recent price before a trade", "duckdb", k=2)
    assert hits and "ASOF" in " ".join(hits[0].chunk.heading_path)


def test_cli_ingests_and_persists(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "asof.md").write_text(PAGE)

    rc = main(
        [
            str(corpus),
            "--tenant",
            "duckdb",
            "--index-root",
            str(tmp_path / "index"),
            "--query",
            "asof join syntax",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "ingested 1 docs" in out
    assert "smoke query" in out
    assert (tmp_path / "index" / "duckdb" / "manifest.json").exists()


def test_cli_reingest_adds_nothing_new(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "asof.md").write_text(PAGE)
    argv = [str(corpus), "--tenant", "duckdb", "--index-root", str(tmp_path / "index")]

    main(argv)
    capsys.readouterr()
    main(argv)
    assert "0 new" in capsys.readouterr().out


def test_cli_rejects_empty_path(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main([str(empty), "--index-root", str(tmp_path / "index")]) == 1
