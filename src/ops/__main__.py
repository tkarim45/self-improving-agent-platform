"""Trace viewer — the M3 artifact.

    python -m src.ops traces              # recent requests, one line each
    python -m src.ops summary             # cost / grounding / guard rollup
    python -m src.ops show <trace_id>     # full record for one request

Reads the SQLite trace store the agent writes to. This is the CLI that makes "what did every
request cost and how did it go" answerable without opening the database.
"""

from __future__ import annotations

import argparse
import json
import sys

from src.ops.trace_store import TraceStore


def _row_line(r: dict) -> str:
    flag = {"block": "⛔", "redact": "✂️ ", "allow": "  "}.get(r["guard_action"], "  ")
    grounded = "✓" if r["grounded"] else "✗"
    esc = " ↑esc" if r["escalated"] else ""
    return (
        f"{flag} {r['trace_id']}  {r['ts'][:19]}  {r['model_tier']:6} "
        f"${r['cost_usd']:.4f}  {r['latency_ms']:6.0f}ms  "
        f"grounded {grounded}  cite {r['citation_rate']:.0%}{esc}  {r['query'][:44]}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.ops")
    parser.add_argument("cmd", choices=["traces", "summary", "show"])
    parser.add_argument("trace_id", nargs="?", default="")
    parser.add_argument("--db", default="data/traces.db")
    parser.add_argument("--tenant", default=None)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args(argv)

    store = TraceStore(args.db)

    if args.cmd == "traces":
        rows = store.recent(limit=args.limit, tenant=args.tenant)
        if not rows:
            print("no traces yet — run the agent first")
            return 0
        for r in rows:
            print(_row_line(r))
        return 0

    if args.cmd == "summary":
        s = store.summary(tenant=args.tenant)
        if not s["n"]:
            print("no traces yet")
            return 0
        print(f"requests        : {s['n']}")
        print(f"total cost      : ${s['total_cost']:.4f}  (mean ${s['mean_cost']:.4f})")
        print(f"mean latency    : {s['mean_latency']:.0f} ms")
        print(f"grounded rate   : {s['grounded_rate']:.0%}")
        print(f"mean cite rate  : {s['mean_citation_rate']:.0%}")
        print(f"escalations     : {s['escalations']}")
        print(f"guard blocks    : {s['blocks']}")
        print(f"guard redactions: {s['redactions']}")
        print("by tier         :")
        for tier, v in sorted(s["by_tier"].items()):
            print(f"  {tier:6} {v['n']:4} requests  ${v['cost'] or 0:.4f}")
        return 0

    if args.cmd == "show":
        if not args.trace_id:
            print("show requires a trace_id", file=sys.stderr)
            return 1
        row = store.get(args.trace_id)
        if not row:
            print(f"no trace {args.trace_id}", file=sys.stderr)
            return 1
        payload = json.loads(row["payload"])
        print(json.dumps(payload, indent=2))
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
