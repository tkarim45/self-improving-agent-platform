"""M5 stage 2 — the reranker flywheel cycle: mine -> train -> shadow -> canary -> decide.

Mirrors the stage-1 router cycle, one component over. The "shadow" is a replay of the frozen
M1 retrieval eval (deterministic, zero Bedrock spend) scoring the fine-tuned reranker against
the off-the-shelf one; the "canary" is a no-regression guard on Recall@10 — the reranker must
not evict a relevant page out of the top 10 the agent depends on. Promotion is dominance-only
on the ranks-2-to-10 metrics a reranker actually moves (MRR, nDCG@10, Recall@3), the exact
band the M1 finding said the cross-encoder lives in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.eval.retrieval import ConfigResult, evaluate, load_spec
from src.flywheel.reranker_train import (
    MIN_PAIRS,
    MiningReport,
    mine_triples,
    train_reranker_subprocess,
)
from src.index.graph import build_graph
from src.index.store import HybridIndex
from src.ingest.loaders import load_path
from src.retrieval.pipeline import HybridRetriever, RetrievalConfig
from src.retrieval.rerank import get_reranker

# The reranker earns promotion only if it does not regress the recall the agent relies on.
CANARY_METRIC = "recall@10"


@dataclass
class RerankerDecision:
    promote: bool
    reason: str
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)  # arm -> {metric: val}
    lift: dict[str, float] = field(default_factory=dict)


def _summary(result: ConfigResult) -> dict[str, float]:
    return {
        "recall@1": round(result.recall.get(1, 0.0), 4),
        "recall@3": round(result.recall.get(3, 0.0), 4),
        "recall@10": round(result.recall.get(10, 0.0), 4),
        "mrr": round(result.mrr, 4),
        "ndcg@10": round(result.ndcg10, 4),
    }


def _score_arm(index, graph, tenant, queries, reranker_name: str) -> ConfigResult:
    cfg = RetrievalConfig(mode="dense", reranker=reranker_name, fetch_k=50, version="dense+rr")
    retriever = HybridRetriever(index, cfg, reranker=get_reranker(reranker_name), graph=graph)
    return evaluate(retriever, queries, tenant, cfg.label)


def decide(base: ConfigResult, tuned: ConfigResult, eps: float = 1e-4) -> RerankerDecision:
    """Dominance on the reranker's own band (MRR, nDCG@10, Recall@3), gated by a Recall@10
    canary. A tie or any regression on a promotion metric fails — the same "must strictly
    beat, never just match" bar the router used."""
    b, t = _summary(base), _summary(tuned)
    lift = {m: round(t[m] - b[m], 4) for m in b}
    metrics = {"base_reranker": b, "tuned_reranker": t}

    if t["recall@10"] < b["recall@10"] - eps:
        return RerankerDecision(
            False,
            f"canary regressed: {CANARY_METRIC} {t['recall@10']:.3f} < {b['recall@10']:.3f} "
            "— the tuned reranker drops a relevant page out of the top 10",
            metrics,
            lift,
        )
    promote_metrics = ["mrr", "ndcg@10", "recall@3"]
    beats = {m: t[m] > b[m] + eps for m in promote_metrics}
    if all(beats.values()):
        gains = ", ".join(f"{m} +{lift[m]:+.3f}".replace("++", "+") for m in promote_metrics)
        return RerankerDecision(True, f"dominates on the rerank band ({gains})", metrics, lift)
    losing = [m for m, ok in beats.items() if not ok]
    return RerankerDecision(
        False,
        f"no dominance — did not strictly beat base on {', '.join(losing)}",
        metrics,
        lift,
    )


@dataclass
class RerankerCycleResult:
    mining: MiningReport
    trained: bool
    decision: RerankerDecision | None
    train_reason: str = ""
    artifact: str | None = None
    candidate_version: str = ""


def run_reranker_cycle(
    ts: str,
    tenant: str = "duckdb",
    index_root: str = "data/index",
    corpus_path: str = "",
    spec_path: Path | None = None,
    candidates_dir: Path = Path("configs/candidates"),
    base_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    epochs: int = 2,
    promotion_log=None,
) -> RerankerCycleResult:
    """One full stage-2 cycle. `ts` is caller-supplied (no clock reads), so a replay is
    reproducible. Returns the mining report, the decision, and the saved artifact path."""
    from src.eval.retrieval import DEFAULT_SPEC

    queries, _spec = load_spec(spec_path or DEFAULT_SPEC)
    index = HybridIndex.load(tenant, root=index_root)
    graph = build_graph(load_path(corpus_path, tenant)) if corpus_path else None

    base_cfg = RetrievalConfig(mode="dense", fetch_k=50)
    base_retriever = HybridRetriever(index, base_cfg, reranker=get_reranker("none"), graph=graph)
    mining = mine_triples(base_retriever, queries, tenant)

    if mining.n_pairs < MIN_PAIRS:
        return RerankerCycleResult(
            mining=mining,
            trained=False,
            decision=None,
            train_reason=f"only {mining.n_pairs} pairs (< {MIN_PAIRS}) — refusing to fit",
        )

    version = f"reranker-{ts[:10]}-{len(mining.triples)}t"
    out_dir = candidates_dir / version
    # Train out-of-process: faiss is resident here (mining + the index), and faiss+torch's
    # backward pass segfaults on macOS. The subprocess imports neither faiss nor this index.
    train_report = train_reranker_subprocess(
        mining.triples, out_dir, base_model=base_model, epochs=epochs
    )
    if train_report.refused:
        return RerankerCycleResult(
            mining=mining, trained=False, decision=None, train_reason=train_report.reason
        )

    base_result = _score_arm(index, graph, tenant, queries, base_model)
    tuned_result = _score_arm(index, graph, tenant, queries, str(out_dir))
    decision = decide(base_result, tuned_result)

    if promotion_log is not None:
        promotion_log.record(
            ts=ts,
            component="reranker",
            candidate_version=version,
            artifact=str(out_dir),
            decision={"promote": decision.promote, "reason": decision.reason},
            shadow={"metrics": decision.metrics, "lift": decision.lift},
            promoted=decision.promote,
        )

    return RerankerCycleResult(
        mining=mining,
        trained=True,
        decision=decision,
        artifact=str(out_dir),
        candidate_version=version,
    )


def active_reranker_name(log) -> str:
    """The reranker the flywheel last promoted, or 'none' if nothing has been promoted."""
    entry = log.active().get("reranker")
    if entry and entry.get("kind") == "learned" and entry.get("artifact"):
        return entry["artifact"]
    return "none"
