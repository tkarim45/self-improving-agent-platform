"""Retrieval + persistence. Offline: the hashing embedder means no model download."""

from __future__ import annotations

import math

import pytest

from src.index.embedders import HashingEmbedder
from src.index.store import HybridIndex
from src.types import Chunk, stable_id

CORPUS = {
    "qualify": "QUALIFY filters the output of a window function, like HAVING does for GROUP BY.",
    "pivot": "The PIVOT statement rotates rows into columns using an aggregate expression.",
    "copy": "COPY writes a table to a Parquet or CSV file on disk.",
    "asof": "ASOF joins match each left row to the closest earlier right row by timestamp.",
}


def make_chunks(tenant: str = "duckdb") -> list[Chunk]:
    return [
        Chunk(
            chunk_id=stable_id(tenant, name),
            doc_id=f"doc-{name}",
            tenant=tenant,
            text=text,
            source_path=f"sql/{name}.md",
            heading_path=("DuckDB", "SQL", name.upper()),
        )
        for name, text in CORPUS.items()
    ]


@pytest.fixture
def index(tmp_path) -> HybridIndex:
    idx = HybridIndex("duckdb", embedder_name="hashing", root=tmp_path)
    idx.add(make_chunks())
    return idx


# --- embedder -------------------------------------------------------------------------


def test_hashing_embedder_is_deterministic_across_instances():
    a = HashingEmbedder(dim=64).encode(["window function"])[0]
    b = HashingEmbedder(dim=64).encode(["window function"])[0]
    assert a == b


def test_hashing_embedder_returns_unit_vectors():
    vec = HashingEmbedder(dim=64).encode(["some text here"])[0]
    assert math.isclose(math.sqrt(sum(v * v for v in vec)), 1.0, rel_tol=1e-6)


def test_hashing_embedder_handles_empty_text():
    vec = HashingEmbedder(dim=32).encode([""])[0]
    assert len(vec) == 32 and all(v == 0.0 for v in vec)


# --- retrieval ------------------------------------------------------------------------


def test_bm25_ranks_the_right_chunk_first(index):
    hits = index.sparse.search("how does QUALIFY work", "duckdb", k=3)
    assert hits and hits[0].chunk.doc_id == "doc-qualify"


def test_dense_ranks_the_right_chunk_first(index):
    hits = index.dense.search("rotate rows into columns", "duckdb", k=3)
    assert hits and hits[0].chunk.doc_id == "doc-pivot"


def test_retrievers_respect_tenant_isolation(tmp_path):
    idx = HybridIndex("duckdb", root=tmp_path)
    idx.add(make_chunks("duckdb"))
    # A chunk belonging to another tenant is refused by this tenant's index.
    assert idx.add(make_chunks("other")) == 0
    assert idx.sparse.search("QUALIFY", "other", k=3) == []
    assert idx.dense.search("QUALIFY", "other", k=3) == []


def test_search_on_empty_index_returns_nothing(tmp_path):
    idx = HybridIndex("duckdb", root=tmp_path)
    assert idx.sparse.search("anything", "duckdb") == []
    assert idx.dense.search("anything", "duckdb") == []


def test_bm25_ignores_a_query_with_no_indexable_tokens(index):
    assert index.sparse.search("!!! ???", "duckdb") == []


# --- persistence ----------------------------------------------------------------------


def test_add_is_idempotent_for_unchanged_content(index):
    assert index.add(make_chunks()) == 0
    assert len(index) == len(CORPUS)


def test_save_load_roundtrip_preserves_ranking(tmp_path):
    idx = HybridIndex("duckdb", root=tmp_path)
    idx.add(make_chunks())
    before = [h.chunk_id for h in idx.dense.search("closest earlier row", "duckdb", k=3)]
    idx.save()

    reloaded = HybridIndex.load("duckdb", root=tmp_path)
    assert len(reloaded) == len(idx)
    after = [h.chunk_id for h in reloaded.dense.search("closest earlier row", "duckdb", k=3)]
    assert before == after
    assert reloaded.sparse.search("parquet", "duckdb", k=1)[0].chunk.doc_id == "doc-copy"


def test_load_without_an_index_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        HybridIndex.load("ghost", root=tmp_path)


def test_incremental_add_after_reload(tmp_path):
    idx = HybridIndex("duckdb", root=tmp_path)
    idx.add(make_chunks())
    idx.save()

    reloaded = HybridIndex.load("duckdb", root=tmp_path)
    new = Chunk(
        chunk_id=stable_id("duckdb", "unnest"),
        doc_id="doc-unnest",
        tenant="duckdb",
        text="UNNEST expands a list column into one row per element.",
        source_path="sql/unnest.md",
        heading_path=("DuckDB", "SQL", "UNNEST"),
    )
    assert reloaded.add([new]) == 1
    reloaded.save()

    again = HybridIndex.load("duckdb", root=tmp_path)
    assert len(again) == len(CORPUS) + 1
    assert again.dense.search("expand a list column", "duckdb", k=1)[0].chunk.doc_id == "doc-unnest"


def test_citation_is_human_readable(index):
    chunk = index.chunks[0]
    assert chunk.citation.startswith(f"[{chunk.chunk_id}]")
    assert "DuckDB > SQL" in chunk.citation
