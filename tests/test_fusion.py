from __future__ import annotations

from src.retrieval.fusion import reciprocal_rank_fusion
from src.types import Chunk, ScoredChunk


def chunk(name: str) -> Chunk:
    return Chunk(chunk_id=name, doc_id=name, tenant="t", text=name, source_path=f"{name}.md")


def run(*names: str) -> list[ScoredChunk]:
    return [ScoredChunk(chunk=chunk(n), score=1.0, retriever="x") for n in names]


def test_empty_runs_return_empty():
    assert reciprocal_rank_fusion({}) == []


def test_rrf_score_matches_the_formula():
    fused = reciprocal_rank_fusion({"a": run("x")}, k=60)
    assert fused[0].score == 1 / 61


def test_agreement_across_runs_beats_a_single_top_hit():
    """A doc ranked 2nd by both retrievers should beat one ranked 1st by only one.

    This is the property RRF is chosen for, so it is worth pinning rather than assuming.
    """
    fused = reciprocal_rank_fusion({"bm25": run("solo", "both"), "dense": run("other", "both")})
    assert fused[0].chunk_id == "both"


def test_duplicate_ids_are_merged_not_repeated():
    fused = reciprocal_rank_fusion({"bm25": run("a", "b"), "dense": run("a", "b")})
    assert [f.chunk_id for f in fused] == ["a", "b"]


def test_retriever_field_records_every_contributing_run():
    fused = reciprocal_rank_fusion({"bm25": run("a"), "dense": run("a")})
    assert fused[0].retriever == "bm25+dense"


def test_weights_shift_the_winner():
    runs = {"bm25": run("keyword_hit"), "dense": run("semantic_hit")}
    bm25_heavy = reciprocal_rank_fusion(runs, weights={"bm25": 5.0, "dense": 1.0})
    dense_heavy = reciprocal_rank_fusion(runs, weights={"bm25": 1.0, "dense": 5.0})
    assert bm25_heavy[0].chunk_id == "keyword_hit"
    assert dense_heavy[0].chunk_id == "semantic_hit"


def test_zero_weight_removes_a_run_from_contention():
    runs = {"bm25": run("keyword_hit"), "dense": run("semantic_hit")}
    fused = reciprocal_rank_fusion(runs, weights={"bm25": 0.0, "dense": 1.0})
    assert fused[0].chunk_id == "semantic_hit"


def test_top_k_truncates():
    assert len(reciprocal_rank_fusion({"a": run("x", "y", "z")}, top_k=2)) == 2


def test_larger_k_flattens_rank_differences():
    """Bigger k damps the top of each list, so rank 1 and rank 2 move closer together."""
    def gap(k: int) -> float:
        fused = reciprocal_rank_fusion({"a": run("x", "y")}, k=k)
        return fused[0].score - fused[1].score

    assert gap(10) > gap(1000)
