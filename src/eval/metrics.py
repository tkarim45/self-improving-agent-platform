"""Retrieval metrics, from scratch.

Written out rather than imported so the definitions are inspectable — the recall@k a report
quotes should be the recall@k the reader thinks it is.

All four take `retrieved` (ranked ids, best first) and `relevant` (the labeled set) and
assume binary relevance, which is what a page-level label gives.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of the relevant items that appear in the top k.

    With a single relevant item this is 1.0 or 0.0 — "did we find it at all", the metric
    that actually predicts whether the agent can answer.
    """
    gold = set(relevant)
    if not gold:
        return 0.0
    return len(gold & set(retrieved[:k])) / len(gold)


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    if k <= 0:
        return 0.0
    gold = set(relevant)
    return len([r for r in retrieved[:k] if r in gold]) / k


def mrr(retrieved: Sequence[str], relevant: Iterable[str], k: int | None = None) -> float:
    """Reciprocal rank of the first relevant hit. Rewards putting the answer at position 1."""
    gold = set(relevant)
    cutoff = len(retrieved) if k is None else k
    for rank, item in enumerate(retrieved[:cutoff], start=1):
        if item in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Normalized discounted cumulative gain, binary relevance.

    Unlike recall it is rank-sensitive, and unlike MRR it credits every relevant hit rather
    than only the first — the metric to read when a query has several right answers.
    """
    gold = set(relevant)
    if not gold or k <= 0:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1) for rank, item in enumerate(retrieved[:k], 1) if item in gold
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0
