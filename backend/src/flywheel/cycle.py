"""One flywheel cycle as a library function.

Extracted from the CLI so the M6 simulator can run cycles against its own isolated state
(its own trace DB and its own promotion log) — the improvement curve has to *show* the
promotion happening, which means the simulation must start from the un-promoted heuristic
rather than inheriting whatever production last promoted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.flywheel import mining
from src.flywheel.promote import PromotionLog, active_router
from src.flywheel.router_train import RouterTrainer
from src.flywheel.shadow import decide, shadow
from src.ops.trace_store import TraceStore


@dataclass
class CycleResult:
    ran: bool
    promoted: bool = False
    reason: str = ""
    dataset_size: int = 0
    shadow: dict[str, Any] | None = None
    unpriced_queries: list[tuple[str, str]] = field(default_factory=list)  # (query, tier)
    candidate_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ran": self.ran,
            "promoted": self.promoted,
            "reason": self.reason,
            "dataset_size": self.dataset_size,
            "candidate_version": self.candidate_version,
        }


def run_cycle(
    store: TraceStore,
    log: PromotionLog,
    ts: str,
    candidates_dir: Path,
    holdout_fraction: float = 0.3,
    min_holdout: int = 5,
    min_hours: float = 12.0,
    golden_records: Path = Path("eval/golden/records.json"),
) -> CycleResult:
    """mine -> train -> shadow -> canary -> decide -> log. Returns what happened.

    `unpriced_queries` is the actionable part of a rejection: the holdout queries whose
    candidate-chosen tier has no observation yet. The simulator's shadow-sampling step runs
    exactly those live, which is how an unattended loop gathers the evidence it needs
    instead of stalling on "insufficient data" forever.
    """
    records = mining.mine(store, holdout_fraction=holdout_fraction)
    ds = mining.router_dataset(records)

    if log.too_soon("router", ts, min_hours=min_hours):
        return CycleResult(ran=False, reason="retrain-frequency cap")

    trainer = RouterTrainer(out_dir=candidates_dir)
    version = f"router-{ts[:13].replace(':', '')}-{len(ds)}ex"
    try:
        candidate, info = trainer.train(ds, version)
    except ValueError as exc:
        return CycleResult(ran=False, reason=str(exc), dataset_size=len(ds))
    cfg = trainer.to_candidate(candidate, info, version)

    incumbent = active_router(log)
    report = shadow(records, incumbent, candidate)

    canary_ok = _canary(candidate, golden_records)
    decision = decide(report, canary_ok, min_holdout=min_holdout)

    # (query, tier-the-candidate-proposed) for every unpriced holdout choice, so the caller
    # can run exactly those live under exactly that tier.
    unpriced_queries = [
        (row["query"], row["candidate"])
        for row in report.per_query
        if row.get("candidate_outcome") == "unknown"
    ]

    log.record(
        ts=ts,
        component="router",
        candidate_version=cfg.version,
        artifact=cfg.artifact_path,
        decision=decision.to_dict(),
        shadow=report.to_dict(),
        promoted=decision.promote,
    )
    return CycleResult(
        ran=True,
        promoted=decision.promote,
        reason=decision.reason,
        dataset_size=len(ds),
        shadow={k: v for k, v in report.to_dict().items() if k != "per_query"},
        unpriced_queries=unpriced_queries,
        candidate_version=cfg.version,
    )


def _canary(candidate_router, golden_records: Path) -> bool:
    from src.eval.golden import gate_from_records, load_cases, load_records

    cases, _ = load_cases()
    records = load_records(golden_records)
    report = gate_from_records(records, cases, threshold=0.75)
    for case in cases:
        candidate_router.route(case["question"])
    return report.passed
