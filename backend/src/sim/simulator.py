"""M6 usage simulator — N simulated weeks of support traffic driving the flywheel unattended.

Each week:
  1. Sample K queries from the pool (deterministic: seeded by week number, so a re-run
     produces the same traffic — no wall-clock, no RNG state).
  2. Serve them with the agent under the CURRENT active config (`active_router` of the sim's
     own promotion log — the config the flywheel last promoted, or the heuristic at week 0).
  3. Run one flywheel cycle over everything traced so far.
  4. If the cycle rejected for lack of observations, spend a bounded SHADOW BUDGET running
     exactly the unpriced (query, tier) pairs live — the production shadow-deployment
     pattern, and the step that makes the loop genuinely unattended: it gathers the evidence
     it needs instead of stalling on "insufficient data".
  5. Record weekly metrics: grounded rate, cost/query, tier mix, promotions.

The whole simulation runs against its OWN state directory (trace DB, promotion log, active
config, candidates), so it starts from the un-promoted heuristic and the curve shows the
transition — and so it never touches the real production config.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.flywheel.cycle import run_cycle
from src.flywheel.promote import PromotionLog, active_router
from src.ops.trace_store import TraceStore

POOL_PATH = Path("eval/sim/query_pool.yaml")


@dataclass
class WeekReport:
    week: int
    router_version: str
    n_queries: int = 0
    grounded: int = 0
    escalations: int = 0
    cost_usd: float = 0.0
    tier_mix: dict[str, int] = field(default_factory=dict)
    cycle: dict[str, Any] = field(default_factory=dict)
    shadow_sampled: int = 0
    shadow_cost_usd: float = 0.0

    @property
    def grounded_rate(self) -> float:
        return self.grounded / self.n_queries if self.n_queries else 0.0

    @property
    def cost_per_query(self) -> float:
        return self.cost_usd / self.n_queries if self.n_queries else 0.0

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__ | {
            "grounded_rate": round(self.grounded_rate, 4),
            "cost_per_query": round(self.cost_per_query, 5),
        }


def load_pool(path: Path = POOL_PATH) -> list[tuple[str, str]]:
    """[(query, category)] flattened with category weights applied as repetition."""
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    pool: list[tuple[str, str]] = []
    for cat, body in spec["categories"].items():
        for q in body["queries"]:
            pool.extend([(q, cat)] * int(body["weight"]))
    return pool


def sample_week(pool: list[tuple[str, str]], week: int, k: int) -> list[tuple[str, str]]:
    """Deterministic weekly sample. Hash-ranked so each week draws a different but
    reproducible mix; sampling across weeks repeats queries, as real support traffic does."""
    ranked = sorted(
        pool,
        key=lambda item: hashlib.sha256(f"w{week}:{item[0]}".encode()).hexdigest(),
    )
    seen: list[tuple[str, str]] = []
    for q, cat in ranked:
        if all(q != s[0] for s in seen):
            seen.append((q, cat))
        if len(seen) == k:
            break
    return seen


class Simulator:
    def __init__(
        self,
        state_dir: Path,
        provider,
        agent_factory,
        k_per_week: int = 12,
        shadow_budget: int = 4,
        spend_limit_usd: float = 0.15,
    ) -> None:
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.store = TraceStore(state_dir / "traces.db")
        self.log = PromotionLog(
            log_path=state_dir / "promotions.jsonl",
            active_path=state_dir / "active.json",
        )
        self.provider = provider
        self.agent_factory = agent_factory  # (provider, router) -> agent
        self.k = k_per_week
        self.shadow_budget = shadow_budget
        self.spend_limit_usd = spend_limit_usd
        self.pool = load_pool()

    def _serve(self, query: str, router, ts: str) -> dict[str, Any]:
        agent = self.agent_factory(self.provider, router)
        run = agent.run_detailed(query, tenant="duckdb")
        self.store.write(run.trace, ts=ts, guard_action=run.guard_action)
        return {
            "tier": run.routing.tier,
            "escalated": run.routing.escalated,
            "grounded": run.citation_report.grounded,
            "cost": run.cost["total_usd"],
        }

    def run_week(self, week: int) -> WeekReport:
        router = active_router(self.log)
        version = self.log.active()["router"]["version"]
        report = WeekReport(week=week, router_version=version)

        # 1-2: serve the week's traffic under the current config.
        for i, (query, _cat) in enumerate(sample_week(self.pool, week, self.k)):
            ts = f"2026-08-{2 + week:02d}T{10 + i // 60:02d}:{i % 60:02d}:00"
            try:
                served = self._serve(query, router, ts)
            except Exception as exc:  # noqa: BLE001 - a bad query must not kill the week
                print(f"    w{week} serve FAILED {type(exc).__name__}: {str(exc)[:60]}")
                continue
            report.n_queries += 1
            report.grounded += int(served["grounded"])
            report.escalations += int(served["escalated"])
            report.cost_usd += served["cost"]
            report.tier_mix[served["tier"]] = report.tier_mix.get(served["tier"], 0) + 1

        # 3: one flywheel cycle over everything traced so far.
        cycle = run_cycle(
            self.store,
            self.log,
            ts=f"2026-08-{2 + week:02d}T20:00:00",
            candidates_dir=self.state_dir / "candidates",
            min_hours=0.0,  # the weekly cadence IS the frequency cap inside the sim
        )
        report.cycle = cycle.to_dict()

        # 4: shadow-sample unpriced candidate choices, within budget.
        if cycle.ran and not cycle.promoted and cycle.unpriced_queries:
            from src.agent.router import AlwaysRouter

            for j, (query, tier) in enumerate(cycle.unpriced_queries[: self.shadow_budget]):
                ts = f"2026-08-{2 + week:02d}T21:{j:02d}:00"
                try:
                    served = self._serve(query, AlwaysRouter(tier), ts)
                except Exception as exc:  # noqa: BLE001
                    print(f"    w{week} shadow FAILED {type(exc).__name__}: {str(exc)[:60]}")
                    continue
                report.shadow_sampled += 1
                report.shadow_cost_usd += served["cost"]

        return report

    def run(self, weeks: int) -> list[WeekReport]:
        reports = []
        for week in range(weeks):
            report = self.run_week(week)
            reports.append(report)
            promo = "PROMOTED " + report.cycle.get("candidate_version", "") \
                if report.cycle.get("promoted") else report.cycle.get("reason", "")[:48]
            print(
                f"week {week}: router={report.router_version[:26]:26} "
                f"grounded {report.grounded}/{report.n_queries} "
                f"${report.cost_per_query:.4f}/q  tiers={report.tier_mix}  "
                f"shadow+{report.shadow_sampled}  cycle: {promo}"
            )
        (self.state_dir / "weekly.json").write_text(
            json.dumps([r.to_dict() for r in reports], indent=2)
        )
        return reports
