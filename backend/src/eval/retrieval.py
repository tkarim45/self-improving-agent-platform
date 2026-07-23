"""Retrieval evaluation: run configs against the labeled set, report Recall@k / MRR / nDCG.

Scoring is at **page level**. A retrieved chunk counts as a hit on its source page, and the
ranked chunk list is collapsed to ranked unique pages before metrics are computed. Two
reasons: the labels are page-level (see eval/retrieval/duckdb.yaml), and a config that
returns three chunks of the right page has found one answer, not three.

Multi-hop queries are scored separately and stricter. `recall@k` gives a query 0.5 for
finding one of two required pages, but a multi-hop answer needs both — so those queries also
report **coverage**, the fraction of queries where *every* required page made the top k.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.eval.metrics import mean, mrr, ndcg_at_k, recall_at_k
from src.index.graph import build_graph
from src.index.store import HybridIndex
from src.ingest.loaders import load_path
from src.retrieval.pipeline import HybridRetriever, RetrievalConfig
from src.retrieval.rerank import get_reranker

DEFAULT_SPEC = Path("eval/retrieval/duckdb.yaml")
K_VALUES = (1, 3, 5, 10)


@dataclass
class EvalQuery:
    id: str
    query: str
    relevant: list[str]
    kind: str = "single"
    note: str = ""


@dataclass
class ConfigResult:
    label: str
    recall: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg10: float = 0.0
    multihop_recall10: float = 0.0
    multihop_coverage10: float = 0.0
    latency_ms: float = 0.0
    per_query: dict[str, float] = field(default_factory=dict)


def load_spec(path: Path = DEFAULT_SPEC) -> tuple[list[EvalQuery], dict[str, Any]]:
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    queries = [
        EvalQuery(
            id=q["id"],
            query=q["query"],
            relevant=list(q["relevant"]),
            kind=q.get("kind", "single"),
            note=q.get("note", ""),
        )
        for q in spec["queries"]
    ]
    return queries, spec


def ranked_pages(hits) -> list[str]:
    """Collapse ranked chunks to ranked unique source pages, keeping best rank per page."""
    seen: list[str] = []
    for hit in hits:
        page = hit.chunk.source_path
        if page not in seen:
            seen.append(page)
    return seen


def evaluate(
    retriever: HybridRetriever,
    queries: list[EvalQuery],
    tenant: str,
    label: str,
    depth: int = 10,
) -> ConfigResult:
    import time

    result = ConfigResult(label=label)
    recalls: dict[int, list[float]] = {k: [] for k in K_VALUES}
    rrs, ndcgs, latencies = [], [], []
    mh_recalls, mh_coverage = [], []

    for q in queries:
        t0 = time.perf_counter()
        hits = retriever.search(q.query, tenant, k=depth)
        latencies.append((time.perf_counter() - t0) * 1000)

        pages = ranked_pages(hits)
        for k in K_VALUES:
            recalls[k].append(recall_at_k(pages, q.relevant, k))
        rrs.append(mrr(pages, q.relevant))
        ndcgs.append(ndcg_at_k(pages, q.relevant, 10))
        result.per_query[q.id] = recall_at_k(pages, q.relevant, depth)

        if q.kind == "multi_hop":
            mh_recalls.append(recall_at_k(pages, q.relevant, 10))
            mh_coverage.append(1.0 if set(q.relevant) <= set(pages[:10]) else 0.0)

    result.recall = {k: mean(v) for k, v in recalls.items()}
    result.mrr = mean(rrs)
    result.ndcg10 = mean(ndcgs)
    result.multihop_recall10 = mean(mh_recalls)
    result.multihop_coverage10 = mean(mh_coverage)
    result.latency_ms = mean(latencies)
    return result


def build_configs(reranker_name: str) -> list[RetrievalConfig]:
    """The comparison arms. Each isolates one variable against the one before it.

    The weighted-hybrid arm exists to separate two explanations of a weak hybrid score: RRF
    being the wrong fusion for this corpus, versus RRF being fed a lopsided pair of runs. And
    the reranker is applied to BOTH first stages, because reranking only the weaker one would
    measure the first stage, not the reranker.
    """
    configs = [
        RetrievalConfig(mode="bm25", reranker="none", version="bm25"),
        RetrievalConfig(mode="dense", reranker="none", version="dense"),
        RetrievalConfig(mode="hybrid", reranker="none", version="hybrid"),
        RetrievalConfig(
            mode="hybrid", reranker="none", bm25_weight=0.3, dense_weight=1.0, version="hybrid-w"
        ),
    ]
    # 0.05 is the swept optimum, not a guess. The boost has a sharp cliff: at >=0.2 multi-hop
    # coverage halves (0.667 -> 0.333), because the docs average 5.7 links per page, so a
    # strong boost floods the top 10 with topically-adjacent neighbours of the seed pages.
    configs.append(
        RetrievalConfig(mode="dense", graph_boost=0.05, fetch_k=50, version="dense+graph")
    )
    if reranker_name not in ("", "none"):
        configs += [
            RetrievalConfig(mode="dense", reranker=reranker_name, fetch_k=50, version="dense+rr"),
            RetrievalConfig(mode="hybrid", reranker=reranker_name, fetch_k=50, version="hybrid+rr"),
        ]
    return configs


def run(
    tenant: str = "duckdb",
    spec_path: Path = DEFAULT_SPEC,
    index_root: str = "data/index",
    reranker_name: str = "none",
    corpus_path: str = "",
) -> tuple[list[ConfigResult], list[EvalQuery], dict[str, Any]]:
    queries, spec = load_spec(spec_path)
    index = HybridIndex.load(tenant, root=index_root)
    graph = build_graph(load_path(corpus_path, tenant)) if corpus_path else None
    if graph is not None:
        print(f"link graph: {graph.stats()}", file=sys.stderr)

    results = []
    for cfg in build_configs(reranker_name):
        retriever = HybridRetriever(
            index, cfg, reranker=get_reranker(cfg.reranker), graph=graph
        )
        results.append(evaluate(retriever, queries, tenant, cfg.label))
    return results, queries, spec


def format_report(
    results: list[ConfigResult],
    queries: list[EvalQuery],
    spec: dict[str, Any],
    index: HybridIndex,
) -> str:
    n_multi = sum(1 for q in queries if q.kind == "multi_hop")
    lines = [
        "# Retrieval eval — DuckDB docs (Milestone 1)",
        "",
        f"- Corpus: `{spec.get('corpus', 'unknown')}`",
        f"- Index: {len(index)} chunks, embedder `{index.embedder_name}`",
        f"- Queries: {len(queries)} ({len(queries) - n_multi} single-hop, {n_multi} multi-hop)",
        "- Labels are page-level; ranked chunks are collapsed to ranked unique pages.",
        "",
        "## Results",
        "",
        "| config | R@1 | R@3 | R@5 | R@10 | MRR | nDCG@10 | latency |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.label} | {r.recall[1]:.3f} | {r.recall[3]:.3f} | {r.recall[5]:.3f} | "
            f"{r.recall[10]:.3f} | {r.mrr:.3f} | {r.ndcg10:.3f} | {r.latency_ms:.1f} ms |"
        )

    lines += [
        "",
        f"## Multi-hop ({n_multi} queries)",
        "",
        "`coverage` = every required page present in the top 10, which is what a multi-hop",
        "answer actually needs. `recall@10` gives partial credit and so reads higher.",
        "",
        "| config | recall@10 | coverage@10 |",
        "|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.label} | {r.multihop_recall10:.3f} | {r.multihop_coverage10:.3f} |"
        )

    # Queries every arm missed are the honest part of the report: they say what the corpus
    # or the labels get wrong, and they are the seed set for M5's failure mining.
    misses = [
        q for q in queries if all(r.per_query.get(q.id, 0.0) == 0.0 for r in results)
    ]
    lines += ["", "## Missed by every config", ""]
    if misses:
        for q in misses:
            lines.append(f"- `{q.id}` {q.query!r} -> {', '.join(q.relevant)}")
    else:
        lines.append("None.")
    lines.append("")
    return "\n".join(lines)
