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
from src.flywheel.promote import PromotionLog
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


def cmd_cycle(args) -> int:
    from src.flywheel.cycle import run_cycle

    store = TraceStore(args.db)
    log = PromotionLog()
    result = run_cycle(
        store,
        log,
        ts=args.ts,
        candidates_dir=Path("configs/candidates"),
        holdout_fraction=args.holdout,
        min_holdout=args.min_holdout,
        min_hours=args.min_hours,
    )
    print(json.dumps(result.to_dict(), indent=2))
    if result.unpriced_queries:
        print("unpriced candidate choices (run these live to price them):")
        for q, tier in result.unpriced_queries:
            print(f"  [{tier}] {q[:70]}")
    if result.promoted:
        print("active config:", json.dumps(log.active()["router"]))
    return 0


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
