"""M5 stage 2 — reranker flywheel, offline. No training, no model download: the mining logic,
the promote/reject decision, and the promotion machinery, all with fakes."""

from __future__ import annotations

from src.eval.retrieval import ConfigResult, EvalQuery
from src.flywheel.promote import PromotionLog
from src.flywheel.reranker_cycle import active_reranker_name, decide
from src.flywheel.reranker_train import (
    MIN_PAIRS,
    Triple,
    _pairs_and_labels,
    mine_triples,
    train_reranker,
)
from src.types import Chunk, ScoredChunk


def _chunk(cid: str, page: str, text: str = "body") -> Chunk:
    return Chunk(chunk_id=cid, doc_id=page, tenant="t", text=text, source_path=page)


class FakeRetriever:
    """Returns a fixed ranked list per query, so mining is deterministic and offline."""

    def __init__(self, ranking: list[Chunk]) -> None:
        self._ranking = ranking

    def search(self, query: str, tenant: str, k: int = 50) -> list[ScoredChunk]:
        return [ScoredChunk(chunk=c, score=1.0 - i * 0.01) for i, c in enumerate(self._ranking)]


def _result(label: str, r1: float, r3: float, r10: float, mrr: float, ndcg: float) -> ConfigResult:
    return ConfigResult(label=label, recall={1: r1, 3: r3, 10: r10}, mrr=mrr, ndcg10=ndcg)


# --- mining ---------------------------------------------------------------------------


def test_mining_reads_positives_and_hard_negatives_off_the_first_stage():
    ranking = [
        _chunk("c1", "wrong_a.md"),  # hard negative (ranked #1, not relevant)
        _chunk("c2", "right.md"),  # positive
        _chunk("c3", "wrong_b.md"),  # hard negative
    ]
    q = EvalQuery(id="q1", query="how", relevant=["right.md"])
    report = mine_triples(FakeRetriever(ranking), [q], "t", max_neg_per_query=2)
    assert report.queries_used == 1
    assert len(report.triples) == 2  # one positive paired with each of two hard negatives
    for tr in report.triples:
        assert "body" in tr.positive and "body" in tr.negative


def test_mining_skips_a_query_with_no_hard_negative():
    ranking = [_chunk("c1", "right.md")]  # everything retrieved is relevant -> no negative
    q = EvalQuery(id="q1", query="how", relevant=["right.md"])
    report = mine_triples(FakeRetriever(ranking), [q], "t")
    assert report.queries_used == 0 and report.queries_skipped == 1
    assert report.triples == []


def test_pairs_and_labels_expand_each_triple_into_pos_and_neg():
    triples = [Triple("q", "good", "bad")]
    pairs, labels = _pairs_and_labels(triples)
    assert pairs == [["q", "good"], ["q", "bad"]]
    assert labels == [1.0, 0.0]


def test_train_refuses_below_the_pair_floor_without_touching_torch(tmp_path):
    # One triple = 2 pairs, well under MIN_PAIRS — must refuse before importing torch.
    report = train_reranker([Triple("q", "g", "b")], tmp_path / "out")
    assert report.refused and report.n_pairs < MIN_PAIRS
    assert not (tmp_path / "out" / "model.safetensors").exists()


# --- the decision ---------------------------------------------------------------------


def test_decide_promotes_on_strict_dominance():
    base = _result("base", 0.30, 0.60, 0.80, 0.50, 0.55)
    tuned = _result("tuned", 0.35, 0.66, 0.82, 0.55, 0.60)  # strictly beats on all bands
    d = decide(base, tuned)
    assert d.promote is True
    assert "dominates" in d.reason


def test_decide_rejects_when_a_promotion_metric_is_flat():
    # The real 2026-07-24 run: recall@3/@10/nDCG up, but MRR flat -> no dominance.
    base = _result("base", 0.357, 0.671, 0.814, 0.558, 0.608)
    tuned = _result("tuned", 0.314, 0.743, 0.871, 0.555, 0.625)
    d = decide(base, tuned)
    assert d.promote is False
    assert "mrr" in d.reason


def test_decide_rejects_on_canary_regression_even_if_ranking_improves():
    base = _result("base", 0.30, 0.60, 0.85, 0.50, 0.55)
    tuned = _result("tuned", 0.40, 0.70, 0.80, 0.60, 0.65)  # recall@10 dropped 0.85 -> 0.80
    d = decide(base, tuned)
    assert d.promote is False
    assert "canary" in d.reason and "recall@10" in d.reason


# --- promotion machinery (the promote path, exercised without training) ---------------


def test_reranker_promotion_sets_active_and_rolls_back(tmp_path):
    log = PromotionLog(log_path=tmp_path / "promotions.jsonl", active_path=tmp_path / "active.json")
    assert active_reranker_name(log) == "none"  # nothing promoted yet

    log.record(
        ts="2026-07-24T00:00:00",
        component="reranker",
        candidate_version="reranker-test-10t",
        artifact="configs/candidates/reranker-test-10t",
        decision={"promote": True, "reason": "dominates"},
        shadow={},
        promoted=True,
    )
    assert active_reranker_name(log) == "configs/candidates/reranker-test-10t"

    log.rollback("reranker", ts="2026-07-24T01:00:00")
    assert active_reranker_name(log) == "none"  # back to the default identity reranker


def test_get_reranker_active_is_identity_when_nothing_promoted(monkeypatch):
    from src.retrieval.rerank import get_reranker

    # No reranker in the active config -> "active" resolves to the identity control arm.
    # Patch the class method so the lazily-imported PromotionLog() inside get_reranker sees it.
    monkeypatch.setattr(PromotionLog, "active", lambda self: {})
    assert get_reranker("active").name == "identity"
