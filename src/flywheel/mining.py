"""Failure mining — turn the trace store into training signal.

Reads every trace M3 persisted and classifies each into a failure mode, because different
failures feed different parts of the flywheel:

  ok             — grounded answer (or correct abstention). Supplies a POSITIVE router label:
                   the tier that served it was sufficient.
  bad_routing    — the cheap tier failed and escalation to strong then succeeded. The most
                   valuable record in the store: an observed (query -> strong-was-needed)
                   label, not a guess.
  bad_retrieval  — retrieval returned nothing relevant enough to cite (no citations, empty or
                   useless retrieved set). Feeds the reranker/embedder fine-tune later.
  bad_reasoning  — retrieval produced passages but the answer is ungrounded anyway. The model
                   had the material and did not use it. Feeds prompt/critic work.
  exhausted      — ran out of tool budget. A control failure, not a knowledge one.
  blocked        — stopped at the guardrail. Not a quality failure at all.

Every failed trace is also a candidate *hard eval case* (M5 step 2). The contamination guard
is structural: queries are split train/holdout by a deterministic hash BEFORE any use, and a
query used to train a candidate is never used to evaluate it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from src.ops.trace_store import TraceStore


@dataclass
class MinedRecord:
    trace_id: str
    query: str
    mode: str  # ok | bad_routing | bad_retrieval | bad_reasoning | exhausted | blocked
    tier: str  # tier that served the final answer
    sufficient_tier: str | None  # cheapest tier OBSERVED to suffice, None if none did
    escalated: bool
    grounded: bool
    citation_rate: float
    cost_usd: float
    split: str = "train"  # train | holdout, by query hash

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def split_of(query: str, holdout_fraction: float = 0.3) -> str:
    """Deterministic train/holdout split by query content.

    Hash-based so the split survives re-mining and re-ordering — the same query always lands
    on the same side, which is what makes the contamination guard structural rather than a
    convention someone has to remember.
    """
    h = int(hashlib.sha256(query.strip().lower().encode()).hexdigest()[:8], 16)
    return "holdout" if (h % 1000) < holdout_fraction * 1000 else "train"


def classify(row: dict[str, Any]) -> tuple[str, str | None]:
    """(failure mode, cheapest sufficient tier or None) from a trace-store row."""
    import json

    payload = json.loads(row["payload"]) if isinstance(row.get("payload"), str) else {}
    scores = payload.get("scores", {})

    if row["guard_action"] == "block" or scores.get("blocked"):
        return "blocked", None
    if scores.get("exhausted"):
        return "exhausted", None

    grounded = bool(row["grounded"])
    escalated = bool(row["escalated"])
    retrieved = payload.get("retrieved", [])

    if grounded:
        if escalated:
            # Cheap failed, strong worked: strong is the cheapest tier observed to suffice.
            return "bad_routing", "strong"
        return "ok", row["model_tier"]

    # Not grounded. Was the material even there?
    if not retrieved:
        return "bad_retrieval", None
    if row["invalid_cites"] > 0 or row["citation_rate"] == 0.0:
        return "bad_reasoning", None
    return "bad_reasoning", None


def mine(store: TraceStore, holdout_fraction: float = 0.3) -> list[MinedRecord]:
    records: list[MinedRecord] = []
    for row in store.recent(limit=100_000):
        mode, sufficient = classify(row)
        records.append(
            MinedRecord(
                trace_id=row["trace_id"],
                query=row["query"],
                mode=mode,
                tier=row["model_tier"],
                sufficient_tier=sufficient,
                escalated=bool(row["escalated"]),
                grounded=bool(row["grounded"]),
                citation_rate=row["citation_rate"],
                cost_usd=row["cost_usd"],
                split=split_of(row["query"], holdout_fraction),
            )
        )
    return records


@dataclass
class RouterDataset:
    """(query -> cheapest sufficient tier) pairs, train side only."""

    queries: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)  # "cheap" | "strong"

    def __len__(self) -> int:
        return len(self.queries)

    @property
    def label_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for label in self.labels:
            out[label] = out.get(label, 0) + 1
        return out


def router_dataset(records: list[MinedRecord]) -> RouterDataset:
    """Supervision for the learned router, from OBSERVED outcomes only.

    A query labels "cheap" when the cheap tier's answer was grounded, and "strong" when cheap
    failed and escalation succeeded. Queries where nothing sufficed teach the router nothing
    about tier choice (the failure is elsewhere) and are excluded. Holdout-side queries are
    excluded here by construction — they are the shadow eval's material.
    """
    ds = RouterDataset()
    for r in records:
        if r.split != "train" or r.sufficient_tier is None:
            continue
        if r.mode == "ok" and r.tier == "cheap":
            ds.queries.append(r.query)
            ds.labels.append("cheap")
        elif r.mode == "bad_routing":
            ds.queries.append(r.query)
            ds.labels.append("strong")
        elif r.mode == "ok" and r.tier == "strong" and not r.escalated:
            # Routed strong directly and it worked — but we never observed cheap on this
            # query, so "strong was NEEDED" is not established. Skip rather than teach the
            # router the incumbent's possibly-wasteful habit. (M2 measured exactly this
            # waste: 2.7x cost for no detectable gain.)
            continue
    return ds


def hard_cases(records: list[MinedRecord]) -> list[dict[str, Any]]:
    """Failed queries as candidate golden cases (M5 step 2). Held out by construction:
    anything that becomes an eval case is banned from training data by the same hash split."""
    return [
        {"query": r.query, "mode": r.mode, "trace_id": r.trace_id}
        for r in records
        if r.mode in ("bad_retrieval", "bad_reasoning", "exhausted")
    ]


def summarize(records: list[MinedRecord]) -> dict[str, Any]:
    modes: dict[str, int] = {}
    for r in records:
        modes[r.mode] = modes.get(r.mode, 0) + 1
    return {
        "n": len(records),
        "by_mode": dict(sorted(modes.items())),
        "train": sum(1 for r in records if r.split == "train"),
        "holdout": sum(1 for r in records if r.split == "holdout"),
        "escalations": sum(1 for r in records if r.escalated),
        "total_cost": round(sum(r.cost_usd for r in records), 4),
    }
