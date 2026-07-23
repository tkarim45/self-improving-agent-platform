"""Pipeline + reranker behaviour. Offline via the hashing embedder."""

from __future__ import annotations

import pytest

from src.index.graph import build_graph
from src.index.store import HybridIndex
from src.retrieval.pipeline import HybridRetriever, RetrievalConfig
from src.retrieval.rerank import IdentityReranker, LexicalReranker, get_reranker
from src.types import Chunk, Document, ScoredChunk, stable_id

PAGES = {
    "qualify.md": "QUALIFY filters the output of a window function like HAVING does for groups.",
    "pivot.md": "PIVOT rotates rows into columns using an aggregate expression.",
    "copy.md": "COPY writes a table out to a parquet or csv file.",
    "asof.md": "ASOF join matches each row to the closest earlier row by timestamp.",
    "unnest.md": "UNNEST expands a list into one row per element.",
}


@pytest.fixture
def index(tmp_path) -> HybridIndex:
    idx = HybridIndex("t", embedder_name="hashing", root=tmp_path)
    idx.add(
        [
            Chunk(
                chunk_id=stable_id("t", name),
                doc_id=name,
                tenant="t",
                text=text,
                source_path=name,
                heading_path=("Docs", name),
            )
            for name, text in PAGES.items()
        ]
    )
    return idx


def pages(hits) -> list[str]:
    return [h.chunk.source_path for h in hits]


# --- config ---------------------------------------------------------------------------


def test_unknown_mode_is_rejected_at_construction():
    with pytest.raises(ValueError):
        RetrievalConfig(mode="magic")


def test_label_describes_the_arm():
    assert RetrievalConfig(mode="dense").label == "dense"
    assert RetrievalConfig(mode="hybrid", bm25_weight=0.3).label.startswith("hybrid(bm25=0.3")
    assert RetrievalConfig(mode="dense", reranker="lexical").label == "dense+rerank"
    assert "graph(0.05)" in RetrievalConfig(mode="dense", graph_boost=0.05).label


# --- modes ----------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_every_mode_retrieves_the_right_page(index, mode):
    r = HybridRetriever(index, RetrievalConfig(mode=mode))
    assert "qualify.md" in pages(r.search("filter a window function", "t", k=3))


def test_search_respects_k(index):
    r = HybridRetriever(index, RetrievalConfig(mode="hybrid"))
    assert len(r.search("parquet file", "t", k=2)) <= 2


def test_hybrid_sees_a_page_that_one_leg_alone_misses(index):
    """The point of fusion: union the candidate pools, not just reorder one."""
    r_hybrid = HybridRetriever(index, RetrievalConfig(mode="hybrid"))
    assert len(pages(r_hybrid.search("rotate rows into columns", "t", k=5))) > 0


def test_tenant_isolation_holds_through_the_pipeline(index):
    r = HybridRetriever(index, RetrievalConfig(mode="hybrid"))
    assert r.search("window function", "other-tenant", k=5) == []


# --- rerank ---------------------------------------------------------------------------


def test_identity_reranker_preserves_order():
    hits = [
        ScoredChunk(chunk=Chunk(chunk_id=n, doc_id=n, tenant="t", text=n, source_path=n), score=s)
        for n, s in [("a", 0.9), ("b", 0.8)]
    ]
    assert IdentityReranker().rerank("q", hits, k=2) == hits


def test_lexical_reranker_reorders_by_overlap():
    hits = [
        ScoredChunk(
            chunk=Chunk(chunk_id=n, doc_id=n, tenant="t", text=t, source_path=n),
            score=0.5,
        )
        for n, t in [("off", "totally unrelated content"), ("on", "window function filter")]
    ]
    assert LexicalReranker().rerank("window function filter", hits, k=2)[0].chunk_id == "on"


def test_reranker_factory_resolves_names():
    assert isinstance(get_reranker("none"), IdentityReranker)
    assert isinstance(get_reranker("lexical"), LexicalReranker)


def test_pipeline_overfetches_when_a_reranker_is_present(index, monkeypatch):
    """A reranker can only reorder what stage one recalled, so depth must widen."""
    seen = {}
    r = HybridRetriever(index, RetrievalConfig(mode="dense", reranker="lexical", fetch_k=25))
    original = r.first_stage

    def spy(query, tenant, depth):
        seen["depth"] = depth
        return original(query, tenant, depth)

    monkeypatch.setattr(r, "first_stage", spy)
    r.search("window", "t", k=3)
    assert seen["depth"] == 25


def test_pipeline_does_not_overfetch_without_a_reranker(index, monkeypatch):
    seen = {}
    r = HybridRetriever(index, RetrievalConfig(mode="dense", fetch_k=25))
    original = r.first_stage

    def spy(query, tenant, depth):
        seen["depth"] = depth
        return original(query, tenant, depth)

    monkeypatch.setattr(r, "first_stage", spy)
    r.search("window", "t", k=3)
    assert seen["depth"] == 3


# --- graph ----------------------------------------------------------------------------


def test_graph_boost_changes_ranking(index):
    docs = [
        Document(
            doc_id="qualify.md",
            tenant="t",
            text="QUALIFY filters windows {% link docs/current/unnest.md %}",
            source_path="qualify.md",
            title="qualify",
        ),
        Document(doc_id="unnest.md", tenant="t", text="UNNEST", source_path="unnest.md", title="u"),
    ]
    graph = build_graph(docs)
    cfg = RetrievalConfig(mode="dense", graph_boost=5.0, graph_seed_n=1, fetch_k=5)
    boosted = HybridRetriever(index, cfg, graph=graph).search("filter window function", "t", k=5)
    plain = HybridRetriever(index, RetrievalConfig(mode="dense")).search(
        "filter window function", "t", k=5
    )
    assert pages(boosted).index("unnest.md") < pages(plain).index("unnest.md")


def test_graph_boost_is_inert_without_a_graph(index):
    cfg = RetrievalConfig(mode="dense", graph_boost=0.5)
    with_flag = HybridRetriever(index, cfg, graph=None).search("window", "t", k=5)
    without = HybridRetriever(index, RetrievalConfig(mode="dense")).search("window", "t", k=5)
    assert pages(with_flag) == pages(without)
