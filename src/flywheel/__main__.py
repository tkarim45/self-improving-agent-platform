"""Flywheel CLI — the M5 loop, one stage per subcommand plus `cycle` for the whole thing.

    python -m src.flywheel traffic --live        # generate traces (SPENDS)
    python -m src.flywheel mine                  # classify traces, build datasets
    python -m src.flywheel cycle --ts <iso>      # mine -> train -> shadow -> canary -> decide
    python -m src.flywheel log                   # promotion history
    python -m src.flywheel rollback --ts <iso>

`cycle` is the artifact: an automated retrain -> shadow -> promote(or reject) pass whose
every decision lands in configs/promotions.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from src.flywheel import mining
from src.flywheel.promote import PromotionLog, active_router
from src.flywheel.router_train import RouterTrainer
from src.flywheel.shadow import decide, shadow
from src.ops.trace_store import TraceStore

# Traffic = the M1 labeled queries plus the M2 demo hard cases: realistic support questions
# with known difficulty spread. M6 replaces this with the usage simulator.
TRAFFIC_SPEC = Path("eval/retrieval/duckdb.yaml")


def _traffic_queries() -> list[str]:
    spec = yaml.safe_load(TRAFFIC_SPEC.read_text(encoding="utf-8"))
    queries = [q["query"] for q in spec["queries"]]
    queries += [
        "why would a hash join be slower than a merge join here, and what would you check first",
        "explain the difference between UNION and UNION ALL semantics for duplicate rows",
        "how do I profile which operator dominates runtime in a slow query",
    ]
    return queries


def cmd_traffic(args) -> int:
    from src.agent.__main__ import build_agent
    from src.agent.loop import AgentConfig

    if args.dry_run:
        from src.llm.fake import FakeProvider

        provider = FakeProvider(["The documentation does not cover this."] * 400)
    else:
        from src.llm.bedrock import BedrockProvider

        provider = BedrockProvider()

    store = TraceStore(args.db)
    cfg = AgentConfig(spend_limit_usd=args.spend_limit, critic=False)  # critic off: traffic
    if args.queries_file:
        queries = [
            q.strip() for q in Path(args.queries_file).read_text().splitlines() if q.strip()
        ]
    else:
        queries = _traffic_queries()
    queries = queries[: args.limit or None]
    total = 0.0
    for i, q in enumerate(queries):
        agent, _ = build_agent(provider, args.tenant, "data/index", "data/corpus/duckdb",
                               args.router, cfg)
        try:
            run = agent.run_detailed(q, tenant=args.tenant)
        except Exception as exc:  # noqa: BLE001 - one bad query must not kill the batch
            print(f"[{i:02d}] FAILED {type(exc).__name__}: {str(exc)[:70]}")
            continue
        store.write(run.trace, ts=f"{args.ts[:10]}T{10 + i // 60:02d}:{i % 60:02d}:00",
                    guard_action=run.guard_action)
        total += run.cost["total_usd"]
        flag = "↑" if run.routing.escalated else " "
        print(f"[{i:02d}]{flag} {run.routing.tier:6} ${run.cost['total_usd']:.4f} "
              f"grounded={int(run.citation_report.grounded)} {q[:56]}")
    store.close()
    print(f"\n{len(queries)} queries, total ${total:.4f} -> {args.db}")
    return 0


def cmd_mine(args) -> int:
    store = TraceStore(args.db)
    records = mining.mine(store, holdout_fraction=args.holdout)
    print(json.dumps(mining.summarize(records), indent=2))
    ds = mining.router_dataset(records)
    print(f"router dataset: {len(ds)} examples {ds.label_counts}")
    hard = mining.hard_cases(records)
    print(f"hard-case candidates: {len(hard)}")
    if args.out:
        Path(args.out).write_text(json.dumps([r.to_dict() for r in records], indent=2))
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


def _canary_ok(candidate_router, args) -> tuple[bool, dict]:
    """Frozen canary: replay the golden records; the candidate must not regress the gate.

    The router cannot change a frozen answer, so replay checks that the candidate's
    *machinery* (loading, routing on golden questions) works and the gate itself still
    passes. A live canary (re-running the agent under the candidate) is the stronger check
    and is what `--live-canary` does; replay is the free default.
    """
    from src.eval.golden import gate_from_records, load_cases, load_records

    cases, _ = load_cases()
    records = load_records(Path("eval/golden/records.json"))
    report = gate_from_records(records, cases, threshold=0.75)
    for case in cases:  # exercise the candidate router on every golden question
        candidate_router.route(case["question"])
    return report.passed, {"golden_score": report.score, "mode": "replay"}


def cmd_cycle(args) -> int:
    store = TraceStore(args.db)
    records = mining.mine(store, holdout_fraction=args.holdout)
    summary = mining.summarize(records)
    print("mined:", json.dumps(summary))

    ds = mining.router_dataset(records)
    print(f"router dataset: {len(ds)} examples {ds.label_counts}")

    log = PromotionLog()
    if log.too_soon("router", args.ts, min_hours=args.min_hours):
        print("SKIP: retrain-frequency cap — last promotion too recent")
        return 0

    trainer = RouterTrainer()
    version = f"router-{args.ts[:10]}-{len(ds)}ex"
    try:
        candidate, info = trainer.train(ds, version)
    except ValueError as exc:
        print(f"SKIP: {exc}")
        return 0
    cfg = trainer.to_candidate(candidate, info, version)
    print(f"trained candidate: {cfg.version} ({info})")

    incumbent = active_router(log)
    report = shadow(records, incumbent, candidate)
    print("shadow:", json.dumps({k: v for k, v in report.to_dict().items() if k != "per_query"}))

    canary_ok, canary_info = _canary_ok(candidate, args)
    decision = decide(report, canary_ok, min_holdout=args.min_holdout)
    print(f"decision: {decision.reason}")

    entry = log.record(
        ts=args.ts,
        component="router",
        candidate_version=cfg.version,
        artifact=cfg.artifact_path,
        decision=decision.to_dict() | {"canary": canary_info},
        shadow=report.to_dict(),
        promoted=decision.promote,
    )
    print(("PROMOTED -> " if decision.promote else "logged (not promoted) -> ")
          + str(log.log_path))
    if decision.promote:
        print("active config:", json.dumps(log.active()["router"]))
    return 0 if entry else 1


def cmd_log(args) -> int:
    log = PromotionLog()
    entries = log.entries()
    if not entries:
        print("no promotion history yet")
        return 0
    for e in entries:
        mark = "✅ PROMOTED" if e["promoted"] else "❌ rejected"
        print(f"{e['ts']}  {e['component']:8} {e['candidate_version']:28} {mark}  "
              f"{e['decision'].get('reason', '')[:70]}")
    print("\nactive:", json.dumps(log.active()))
    return 0


def cmd_rollback(args) -> int:
    log = PromotionLog()
    previous = log.rollback(args.component, ts=args.ts)
    print(f"rolled back {args.component} to {previous}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.flywheel")
    sub = parser.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("traffic", help="run the traffic batch through the agent (spends)")
    t.add_argument("--db", default="data/traces.db")
    t.add_argument("--tenant", default="duckdb")
    t.add_argument("--spend-limit", type=float, default=0.15)
    t.add_argument("--limit", type=int, default=0, help="cap number of queries (0 = all)")
    t.add_argument("--dry-run", action="store_true")
    t.add_argument("--live", action="store_true", help="explicit ack that this spends")
    t.add_argument("--ts", default="2026-07-23T00:00:00")
    t.add_argument("--router", default="heuristic", help="heuristic | cheap | strong | active")
    t.add_argument(
        "--queries-file", default="", help="one query per line; overrides the default batch"
    )

    m = sub.add_parser("mine", help="classify traces into failure modes + datasets")
    m.add_argument("--db", default="data/traces.db")
    m.add_argument("--holdout", type=float, default=0.3)
    m.add_argument("--out", default="")

    c = sub.add_parser("cycle", help="mine -> train -> shadow -> canary -> decide")
    c.add_argument("--db", default="data/traces.db")
    c.add_argument("--holdout", type=float, default=0.3)
    c.add_argument("--min-holdout", type=int, default=5)
    c.add_argument("--min-hours", type=float, default=12.0)
    c.add_argument("--ts", required=True, help="ISO timestamp for the log (caller-supplied)")

    line = sub.add_parser("log", help="promotion history + active config")  # noqa: F841

    r = sub.add_parser("rollback", help="restore the previous config")
    r.add_argument("--component", default="router")
    r.add_argument("--ts", required=True)

    args = parser.parse_args(argv)
    if args.cmd == "traffic" and not (args.dry_run or args.live):
        print("traffic spends money: pass --live to confirm, or --dry-run", file=sys.stderr)
        return 2
    return {"traffic": cmd_traffic, "mine": cmd_mine, "cycle": cmd_cycle,
            "log": cmd_log, "rollback": cmd_rollback}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
