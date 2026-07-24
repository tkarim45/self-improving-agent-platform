"""Promotion log, active-config versioning, rollback.

The promotion log is append-only JSONL: every retrain -> shadow -> decide cycle writes one
entry, promoted or not, because rejected candidates are evidence too (the M6 curve should
show the flywheel *declining* to churn, not just its wins). The active config is a small
JSON pointer, and rollback is writing the previous pointer back — the artifacts themselves
are immutable and keep their version names.

Safety valve: a minimum interval between promotions per component, enforced from the log
itself (timestamps are supplied by the caller, never read from the clock — same rule as the
trace store, so replays are reproducible).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONFIG_DIR = Path("configs")
LOG_PATH = CONFIG_DIR / "promotions.jsonl"
ACTIVE_PATH = CONFIG_DIR / "active.json"

DEFAULT_ACTIVE = {
    "router": {"kind": "heuristic", "artifact": None, "version": "heuristic-v0"},
    # The reranker component (M5 stage 2). Default = the identity control arm (no rerank),
    # which is what a rollback with no prior promotion restores to.
    "reranker": {"kind": "identity", "artifact": None, "version": "identity-v0"},
}


@dataclass
class PromotionLog:
    log_path: Path = LOG_PATH
    active_path: Path = ACTIVE_PATH

    def entries(self) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []
        return [json.loads(x) for x in self.log_path.read_text().splitlines() if x.strip()]

    def active(self) -> dict[str, Any]:
        if not self.active_path.exists():
            return json.loads(json.dumps(DEFAULT_ACTIVE))
        return json.loads(self.active_path.read_text())

    def _write_active(self, active: dict[str, Any]) -> None:
        self.active_path.parent.mkdir(parents=True, exist_ok=True)
        self.active_path.write_text(json.dumps(active, indent=2))

    def promotions_for(self, component: str) -> list[dict[str, Any]]:
        return [e for e in self.entries() if e["component"] == component and e["promoted"]]

    def too_soon(self, component: str, ts: str, min_hours: float = 12.0) -> bool:
        """Retrain-frequency cap. Compares ISO timestamps lexicographically-safely via
        fromisoformat; the caller supplies `ts`."""
        from datetime import datetime

        promos = self.promotions_for(component)
        if not promos:
            return False
        last = datetime.fromisoformat(promos[-1]["ts"])
        now = datetime.fromisoformat(ts)
        return (now - last).total_seconds() < min_hours * 3600

    def record(
        self,
        ts: str,
        component: str,
        candidate_version: str,
        artifact: str | None,
        decision: dict[str, Any],
        shadow: dict[str, Any],
        promoted: bool,
    ) -> dict[str, Any]:
        entry = {
            "ts": ts,
            "component": component,
            "candidate_version": candidate_version,
            "artifact": artifact,
            "promoted": promoted,
            "decision": decision,
            "shadow": shadow,
            "previous": self.active().get(component),
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        if promoted:
            active = self.active()
            active[component] = {
                "kind": "learned",
                "artifact": artifact,
                "version": candidate_version,
            }
            self._write_active(active)
        return entry

    def rollback(self, component: str, ts: str) -> dict[str, Any]:
        """Restore the previous config for a component, from the last promoted entry."""
        promos = self.promotions_for(component)
        if not promos:
            raise ValueError(f"nothing to roll back: no promotions for {component!r}")
        previous = promos[-1]["previous"] or DEFAULT_ACTIVE[component]
        active = self.active()
        active[component] = previous
        self._write_active(active)
        entry = {
            "ts": ts,
            "component": component,
            "candidate_version": previous.get("version"),
            "artifact": previous.get("artifact"),
            "promoted": True,
            "decision": {"promote": True, "reason": "ROLLBACK to previous config"},
            "shadow": {},
            "previous": promos[-1],
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        return previous


def active_router(log: PromotionLog):
    """Instantiate whatever router the active config points at."""
    from src.agent.router import HeuristicRouter

    entry = log.active().get("router", DEFAULT_ACTIVE["router"])
    if entry["kind"] == "learned" and entry.get("artifact"):
        from src.flywheel.router_train import LearnedRouter

        return LearnedRouter.load(Path(entry["artifact"]))
    return HeuristicRouter()
