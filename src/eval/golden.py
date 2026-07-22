"""Golden eval runner + CI gate.

Runs the agent on each golden case, scores it by the case's own kind (execution / reference /
abstention), and computes an aggregate the CI gate thresholds on. A change that regresses
quality below the threshold fails the gate — proven in M4 with a deliberately-worse prompt
that goes red where the good prompt goes green.

The gate has two run modes, because a real quality gate costs money and CI should not spend on
every PR:
  - live  : actually run the agent on real Bedrock, score with execution + judge. The true
            quality gate; run nightly or on demand.
  - replay: score a set of frozen golden records (already-run answers) with the deterministic
            checks only. Free, no secrets, catches regressions in scoring/execution code and
            deterministic config. This is what runs in CI on every PR.

Execution is the objective scorer; the LLM-judge is optional and additive. A case's pass/fail
comes from its kind, so the aggregate is not a single opaque number — it is "8/8 exec passed,
3/3 reference cited, 1/1 abstained".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.eval.execcheck import ExecResult, check_answer
from src.eval.scorers import score_answer

DEFAULT_SPEC = Path("eval/golden/duckdb.yaml")


@dataclass
class CaseResult:
    id: str
    kind: str
    passed: bool
    detail: str = ""
    cost_usd: float = 0.0
    exec_result: dict | None = None
    answer: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "passed": self.passed,
            "detail": self.detail,
            "cost_usd": round(self.cost_usd, 6),
            "exec_result": self.exec_result,
        }


@dataclass
class GateReport:
    results: list[CaseResult] = field(default_factory=list)
    threshold: float = 0.75

    @property
    def score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.passed for r in self.results) / len(self.results)

    @property
    def passed(self) -> bool:
        return self.score >= self.threshold

    @property
    def total_cost(self) -> float:
        return sum(r.cost_usd for r in self.results)

    def by_kind(self) -> dict[str, tuple[int, int]]:
        out: dict[str, list[int]] = {}
        for r in self.results:
            acc = out.setdefault(r.kind, [0, 0])
            acc[0] += int(r.passed)
            acc[1] += 1
        return {k: (p, n) for k, (p, n) in out.items()}


def load_cases(path: Path = DEFAULT_SPEC) -> tuple[list[dict], dict]:
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    return spec["cases"], spec


def score_case(
    case: dict, answer: str, cited_ids: list[str], retrieved_ids: list[str]
) -> CaseResult:
    """Score one answer by its case kind. Deterministic — no model call."""
    kind = case["kind"]

    if kind == "exec":
        er: ExecResult = check_answer(answer, case.get("setup", ""), [], case["expected"])
        return CaseResult(
            id=case["id"], kind=kind, passed=er.passed,
            detail=er.detail, exec_result=er.to_dict(), answer=answer,
        )

    if kind == "reference":
        # Passes if the answer cites at least one expected page. Citations are chunk ids; map
        # them to pages via the retrieved set is not available here, so the caller passes the
        # cited *pages* in `cited_ids` for reference cases (see run_golden).
        want = set(case["expect_pages"])
        hit = want & set(cited_ids)
        return CaseResult(
            id=case["id"], kind=kind, passed=bool(hit),
            detail=f"cited {sorted(hit)}" if hit else f"cited none of {sorted(want)}",
            answer=answer,
        )

    if kind == "abstain":
        s = score_answer(answer, cited_ids, retrieved_ids)
        return CaseResult(
            id=case["id"], kind=kind, passed=s.abstained,
            detail="abstained" if s.abstained else "did not abstain (invented an answer)",
            answer=answer,
        )

    raise ValueError(f"unknown case kind {kind!r}")


def format_report(report: GateReport, mode: str = "") -> str:
    lines = [
        f"# Golden gate — {'PASS ✅' if report.passed else 'FAIL ❌'}"
        + (f"  ({mode})" if mode else ""),
        "",
        f"score {report.score:.0%} vs threshold {report.threshold:.0%}  "
        f"({sum(r.passed for r in report.results)}/{len(report.results)} cases)",
        "",
    ]
    for kind, (p, n) in sorted(report.by_kind().items()):
        lines.append(f"- {kind:9}: {p}/{n}")
    lines += ["", "| case | kind | result | detail |", "|---|---|---|---|"]
    for r in report.results:
        mark = "✅" if r.passed else "❌"
        lines.append(f"| {r.id} | {r.kind} | {mark} | {r.detail[:60]} |")
    if report.total_cost:
        lines += ["", f"cost: ${report.total_cost:.4f}"]
    return "\n".join(lines)


def run_golden_live(
    cases: list[dict],
    build_agent_fn,
    tenant: str,
    threshold: float,
    judge=None,
) -> tuple[GateReport, list[dict]]:
    """Live mode: run the agent on each case and score it. `build_agent_fn()` returns a fresh
    (agent, search_tool) per case so the citation whitelist does not leak across questions.

    A judge (optional) adds a subjective faithfulness score alongside the objective execution
    result — the two are recorded together so their agreement can be measured (M4 calibration).
    """
    report = GateReport(threshold=threshold)
    records = []
    for case in cases:
        agent, search = build_agent_fn()
        run = agent.run_detailed(case["question"], tenant=tenant)
        answer = run.answer

        # Map cited chunk ids -> source pages, for reference-case scoring.
        cited_pages = sorted(
            {search.seen[c].source_path for c in run.citation_report.cited_ids if c in search.seen}
        )
        result = score_case(case, answer, cited_pages, sorted(search.retrieved_ids))
        result.cost_usd = run.cost.get("total_usd", 0.0)

        rec: dict[str, Any] = {
            "id": case["id"],
            "kind": case["kind"],
            "question": case["question"],
            "answer": answer,
            "cited_pages": cited_pages,
            "retrieved": sorted(search.retrieved_ids),
            "passed": result.passed,
            "detail": result.detail,
            "cost_usd": run.cost.get("total_usd", 0.0),
        }
        if judge is not None and case["kind"] != "reference":
            passages = "\n\n".join(
                f"[{cid}] {' '.join(ch.text.split())[:400]}" for cid, ch in search.seen.items()
            )
            jm = judge.judge(case["question"], answer, passages or "(no passages retrieved)")
            rec["judge"] = jm.to_dict()
            result.cost_usd += jm.cost_usd
            rec["cost_usd"] += jm.cost_usd
        report.results.append(result)
        records.append(rec)
    return report, records


def load_records(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def gate_from_records(records: list[dict], cases: list[dict], threshold: float) -> GateReport:
    """Replay mode: score frozen answer records against the golden cases. Deterministic."""
    by_id = {c["id"]: c for c in cases}
    report = GateReport(threshold=threshold)
    for rec in records:
        case = by_id.get(rec["id"])
        if case is None:
            continue
        report.results.append(
            score_case(case, rec["answer"], rec.get("cited_pages", []), rec.get("retrieved", []))
        )
    return report
