"""Agent demo / smoke CLI — the Milestone 2 artifact.

    # creds must be in the environment first
    set -a; source ~/.env; set +a
    python -m src.agent demo
    python -m src.agent ask "how do I filter the output of a window function"

Every run prints what it cost and which tier served it. `--dry-run` uses the fake provider so
the plumbing can be exercised with no spend; its numbers are fabricated and labelled as such.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from src.agent.loop import AgentConfig, AgentRun, GroundedAgent
from src.agent.router import get_router
from src.agent.tools import RunSqlTool, SearchDocsTool
from src.index.graph import build_graph
from src.index.store import HybridIndex
from src.ingest.loaders import load_path
from src.llm.base import LLMProvider
from src.llm.pricing import CHEAP, PRICING_AS_OF, STRONG
from src.retrieval.pipeline import HybridRetriever, RetrievalConfig

# Five questions spanning the difficulty range, so the router has something to decide.
# Drawn from the M1 labeled set plus two deliberately harder multi-part ones.
DEMO_QUESTIONS = [
    "how do I filter the output of a window function",
    "what is the syntax for PIVOT",
    "match each trade to the most recent price recorded before it",
    "why would a hash join be slower than a merge join here, and what would you check first",
    "read a hive partitioned parquet dataset stored in an s3 bucket",
]


def build_agent(
    provider: LLMProvider,
    tenant: str,
    index_root: str,
    corpus: str,
    router_name: str,
    config: AgentConfig,
) -> tuple[GroundedAgent, SearchDocsTool]:
    index = HybridIndex.load(tenant, root=index_root)
    graph = build_graph(load_path(corpus, tenant)) if corpus and Path(corpus).exists() else None
    # M1's shipped default: dense first stage + a 0.05 link-graph nudge.
    retriever = HybridRetriever(
        index,
        RetrievalConfig(mode="dense", graph_boost=0.05, fetch_k=50),
        graph=graph,
    )
    search = SearchDocsTool(retriever, tenant=tenant)
    agent = GroundedAgent(
        provider, search, RunSqlTool(), config=config, router=get_router(router_name)
    )
    return agent, search


def report(question: str, run: AgentRun, fake: bool) -> None:
    rep = run.citation_report
    print("=" * 78)
    print(f"Q: {question}")
    print("-" * 78)
    print(run.answer.strip()[:1400])
    print("-" * 78)
    print(f"  tier        : {run.routing.tier}{'  (ESCALATED)' if run.routing.escalated else ''}")
    print(f"  routing     : {run.routing.reason}")
    print(f"  tools       : {', '.join(run.tool_calls) or 'none'}  ({run.iterations} iterations)")
    print(
        f"  grounding   : {len(rep.cited_ids)} cited, {len(rep.invalid_ids)} invalid, "
        f"{rep.citation_rate:.0%} of {rep.n_claims} claims cited"
    )
    if rep.invalid_ids:
        print(f"  INVALID IDS : {rep.invalid_ids}   <-- invented sources")
    if run.critique:
        head = run.critique.splitlines()[0][:80]
        print(f"  critic      : {head}{' (revised)' if run.revised else ''}")
    cost = f"${run.cost['total_usd']:.5f}"
    tokens = f"{run.cost['input_tokens']}in/{run.cost['output_tokens']}out"
    print(
        f"  cost        : {cost}{'  [FABRICATED - dry run]' if fake else ''}  "
        f"({run.cost['calls']} calls, {tokens})"
    )
    print(f"  latency     : {run.trace.latency_ms:.0f} ms")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.agent")
    parser.add_argument("cmd", choices=["demo", "ask"])
    parser.add_argument("question", nargs="?", default="")
    parser.add_argument("--tenant", default="duckdb")
    parser.add_argument("--index-root", default="data/index")
    parser.add_argument("--corpus", default="data/corpus/duckdb")
    parser.add_argument("--router", default="heuristic", help="heuristic | cheap | strong")
    parser.add_argument("--no-critic", action="store_true")
    parser.add_argument("--no-escalate", action="store_true")
    parser.add_argument("--spend-limit", type=float, default=0.25, help="USD per question")
    parser.add_argument("--dry-run", action="store_true", help="fake provider, no spend")
    parser.add_argument("--out", default="", help="write run records as JSON here")
    args = parser.parse_args(argv)

    config = AgentConfig(
        critic=not args.no_critic,
        escalate_on_ungrounded=not args.no_escalate,
        spend_limit_usd=args.spend_limit,
    )

    if args.dry_run:
        from src.llm.fake import FakeProvider

        provider: LLMProvider = FakeProvider(["The documentation does not cover this."] * 40)
    else:
        from src.llm.bedrock import BedrockProvider

        provider = BedrockProvider()

    questions = DEMO_QUESTIONS if args.cmd == "demo" else [args.question]
    if not questions or not questions[0]:
        print("ask requires a question", file=sys.stderr)
        return 1

    print(f"provider={provider.name}  router={args.router}  "
          f"cheap={CHEAP.model_id} strong={STRONG.model_id}  (pricing as of {PRICING_AS_OF})")
    if STRONG.note:
        print(f"note: {STRONG.note}")
    print()

    records, total, t0 = [], 0.0, time.perf_counter()
    for question in questions:
        # A fresh search tool per question — retrieved_ids is the citation whitelist and
        # must not leak passages from a previous question into this one's grounding check.
        agent, _ = build_agent(
            provider, args.tenant, args.index_root, args.corpus, args.router, config
        )
        try:
            run = agent.run_detailed(question, tenant=args.tenant)
        except Exception as exc:  # noqa: BLE001 - one bad question shouldn't kill the demo
            print(f"Q: {question}\n  FAILED: {type(exc).__name__}: {exc}\n")
            continue
        report(question, run, fake=args.dry_run)
        total += run.cost["total_usd"]
        records.append(
            {
                "question": question,
                "answer": run.answer,
                "tier": run.routing.tier,
                "escalated": run.routing.escalated,
                "routing_reason": run.routing.reason,
                "tools": run.tool_calls,
                "iterations": run.iterations,
                "citations": run.citation_report.to_dict(),
                "cost": run.cost,
                "latency_ms": run.trace.latency_ms,
                "revised": run.revised,
            }
        )

    grounded = sum(1 for r in records if r["citations"]["grounded"])
    invalid = sum(r["citations"]["n_uncited"] for r in records)
    print("=" * 78)
    print(
        f"{len(records)} questions | grounded {grounded}/{len(records)} | "
        f"{invalid} uncited claims | total ${total:.5f}"
        f"{'  [FABRICATED]' if args.dry_run else ''} | {time.perf_counter() - t0:.1f}s"
    )
    tiers = [r["tier"] for r in records]
    print(f"tiers: {', '.join(tiers)}  (escalations: {sum(r['escalated'] for r in records)})")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
