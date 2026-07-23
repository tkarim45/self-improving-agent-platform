"""Shadow evaluation + promote-on-lift.

The shadow compares candidate vs incumbent router on the HOLDOUT split, by replay: for each
holdout query we already observed what each tier actually did (grounded or not, and at what
cost), so both routers' choices can be priced without spending a token. A choice for which no
observation exists is counted honestly as `unknown`, never assumed.

Promotion rule, in order:
  1. CANARY (safety valve): the candidate must not regress the frozen golden gate — the
     execution oracle from M4. This is the reward-hacking defense: a candidate that looks
     better on soft metrics but breaks executable ground truth is rejected outright.
  2. QUALITY: holdout success rate must not drop (within a small tolerance — n is small and
     pretending otherwise would be manufacturing significance).
  3. COST: given quality holds, the candidate must be cheaper; or given cost holds, quality
     must be higher. Equal on both -> reject (no reason to churn configs).

With the sample sizes this project has, a bootstrap p-value would be theater. The honest
statement is the paired comparison itself: on N holdout queries, quality X vs Y, cost A vs B,
and the decision rule applied to those numbers. `experimentation-engine`-grade stats arrive
when M6's simulator produces volume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.flywheel.mining import MinedRecord


@dataclass
class ArmResult:
    name: str
    n: int = 0
    successes: int = 0
    cost_usd: float = 0.0
    unknown: int = 0  # chose a tier with no observed outcome

    @property
    def success_rate(self) -> float:
        return self.successes / self.n if self.n else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n": self.n,
            "success_rate": round(self.success_rate, 4),
            "cost_usd": round(self.cost_usd, 4),
            "unknown": self.unknown,
        }


@dataclass
class ShadowReport:
    incumbent: ArmResult
    candidate: ArmResult
    per_query: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "incumbent": self.incumbent.to_dict(),
            "candidate": self.candidate.to_dict(),
            "per_query": self.per_query,
        }


def _observations(records: list[MinedRecord]) -> dict[str, dict[str, dict[str, Any]]]:
    """query -> tier -> {success, cost} from what actually ran.

    An escalated trace observed BOTH tiers: cheap failed (its cost is buried in the combined
    total) and strong succeeded. The split below apportions cost by the M2-measured ratio
    rather than pretending the strong run was free.
    """
    obs: dict[str, dict[str, dict[str, Any]]] = {}
    for r in records:
        entry = obs.setdefault(r.query, {})
        if r.mode == "blocked":
            continue
        if r.escalated:
            # Escalation = cheap attempt failed + strong attempt with outcome r.grounded.
            # M2 measured strong ≈ 4x a cheap run on the same question; split on that.
            entry["cheap"] = {"success": False, "cost": r.cost_usd * 0.2}
            entry["strong"] = {"success": r.grounded, "cost": r.cost_usd * 0.8}
        else:
            entry[r.tier] = {"success": r.grounded, "cost": r.cost_usd}
    return obs


def shadow(
    records: list[MinedRecord],
    incumbent_router,
    candidate_router,
    split: str = "holdout",
) -> ShadowReport:
    obs = _observations([r for r in records if r.split == split and r.mode != "blocked"])
    inc = ArmResult(name="incumbent")
    cand = ArmResult(name="candidate")
    per_query = []

    for query, tiers in obs.items():
        row: dict[str, Any] = {"query": query[:60]}
        for arm, router in (("incumbent", incumbent_router), ("candidate", candidate_router)):
            result = inc if arm == "incumbent" else cand
            choice = router.route(query).tier
            seen = tiers.get(choice)
            row[arm] = choice
            if seen is None:
                result.unknown += 1
                row[f"{arm}_outcome"] = "unknown"
                continue
            result.n += 1
            result.successes += int(seen["success"])
            result.cost_usd += seen["cost"]
            row[f"{arm}_outcome"] = "ok" if seen["success"] else "fail"
        per_query.append(row)

    return ShadowReport(incumbent=inc, candidate=cand, per_query=per_query)


@dataclass
class PromotionDecision:
    promote: bool
    reason: str
    checks: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"promote": self.promote, "reason": self.reason, "checks": self.checks}


def decide(
    report: ShadowReport,
    canary_ok: bool,
    quality_tolerance: float = 0.02,
    min_holdout: int = 5,
) -> PromotionDecision:
    inc, cand = report.incumbent, report.candidate
    checks = {
        "canary_ok": canary_ok,
        "holdout_n": min(inc.n, cand.n),
        "quality": {"incumbent": inc.success_rate, "candidate": cand.success_rate},
        "cost": {"incumbent": round(inc.cost_usd, 4), "candidate": round(cand.cost_usd, 4)},
        "unknown_choices": cand.unknown,
    }

    if not canary_ok:
        return PromotionDecision(False, "REJECT: candidate regressed the frozen canary "
                                 "(execution-oracle golden gate) — reward-hacking guard", checks)
    if min(inc.n, cand.n) < min_holdout:
        n = min(inc.n, cand.n)
        return PromotionDecision(
            False, f"REJECT: holdout too small ({n} < {min_holdout}) to support a claim", checks
        )
    if cand.unknown > 0:
        return PromotionDecision(False, f"REJECT: candidate made {cand.unknown} choice(s) with "
                                 "no observed outcome — price them live before promoting", checks)

    quality_delta = cand.success_rate - inc.success_rate
    cost_delta = cand.cost_usd - inc.cost_usd

    if quality_delta < -quality_tolerance:
        return PromotionDecision(False, f"REJECT: quality dropped "
                                 f"{inc.success_rate:.0%} -> {cand.success_rate:.0%}", checks)
    if quality_delta >= 0 and cost_delta < 0:
        saving = -cost_delta / inc.cost_usd if inc.cost_usd else 0.0
        return PromotionDecision(True, f"PROMOTE: quality held "
                                 f"({inc.success_rate:.0%} -> {cand.success_rate:.0%}) at "
                                 f"{saving:.0%} lower cost", checks)
    if quality_delta > quality_tolerance and cost_delta <= 0:
        return PromotionDecision(True, f"PROMOTE: quality up "
                                 f"{inc.success_rate:.0%} -> {cand.success_rate:.0%} at no "
                                 "added cost", checks)
    return PromotionDecision(False, "REJECT: no measured lift on either axis", checks)
