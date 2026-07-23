"""Eval CLI.

    python -m src.eval retrieval
    python -m src.eval golden --replay eval/golden/records.json   # free, CI gate
    python -m src.eval golden --live                              # real Bedrock, spends
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.eval.golden import (
    DEFAULT_SPEC as GOLDEN_SPEC,
)
from src.eval.golden import (
    format_report as golden_report,
)
from src.eval.golden import (
    gate_from_records,
    load_cases,
    load_records,
    run_golden_live,
)
from src.eval.retrieval import DEFAULT_SPEC, format_report, run
from src.index.store import HybridIndex


def _golden(args) -> int:
    cases, _ = load_cases(Path(args.spec))
    if args.replay:
        records = load_records(Path(args.replay))
        report = gate_from_records(records, cases, threshold=args.threshold)
        mode = "replay"
    else:
        from src.agent.__main__ import build_agent
        from src.agent.loop import AgentConfig

        if args.dry_run:
            from src.llm.fake import FakeProvider

            provider = FakeProvider(["The documentation does not cover this."] * 200)
        else:
            from src.llm.bedrock import BedrockProvider

            provider = BedrockProvider()

        prompt_override = args.prompt_override or None
        cfg = AgentConfig(
            spend_limit_usd=args.spend_limit,
            critic=not args.no_critic,
            escalate_on_ungrounded=not args.no_escalate,
        )

        def make():
            agent, search = build_agent(
                provider, args.tenant, args.index_root, args.corpus, "heuristic", cfg
            )
            if prompt_override:
                # Swap the answer system prompt to prove the gate reacts to a worse prompt.
                import src.agent.loop as loop_mod

                loop_mod.ANSWER_SYSTEM = Path(prompt_override).read_text(encoding="utf-8")
            return agent, search

        judge = None
        if args.judge and not args.dry_run:
            from src.eval.judge import LLMJudge

            judge = LLMJudge(provider)
        report, records = run_golden_live(
            cases, make, args.tenant, args.threshold, judge=judge
        )
        mode = "live"
        if args.records_out:
            Path(args.records_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.records_out).write_text(json.dumps(records, indent=2), encoding="utf-8")

    text = golden_report(report, mode=mode)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    return 0 if report.passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.eval")
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("golden", help="run the golden eval + CI gate")
    g.add_argument("--spec", default=str(GOLDEN_SPEC))
    g.add_argument("--replay", default="", help="score frozen records (free, no model)")
    g.add_argument("--live", action="store_true", help="run the agent for real (spends)")
    g.add_argument("--dry-run", action="store_true", help="live path with the fake provider")
    g.add_argument("--judge", action="store_true", help="add the LLM-judge (live only)")
    g.add_argument("--no-critic", action="store_true")
    g.add_argument("--no-escalate", action="store_true")
    g.add_argument("--prompt-override", default="", help="path to a system prompt to test")
    g.add_argument("--threshold", type=float, default=0.75)
    g.add_argument("--tenant", default="duckdb")
    g.add_argument("--index-root", default="data/index")
    g.add_argument("--corpus", default="data/corpus/duckdb")
    g.add_argument("--spend-limit", type=float, default=0.25)
    g.add_argument("--records-out", default="", help="write per-case records here")
    g.add_argument("--out", default="", help="write the gate report here")

    r = sub.add_parser("retrieval", help="run the labeled retrieval eval")
    r.add_argument("--tenant", default="duckdb")
    r.add_argument("--spec", default=str(DEFAULT_SPEC))
    r.add_argument("--index-root", default="data/index")
    r.add_argument("--reranker", default="none", help="'none', 'lexical', or a cross-encoder")
    r.add_argument(
        "--corpus",
        default="data/corpus/duckdb",
        help="corpus path, used to build the link graph (empty disables graph arms)",
    )
    r.add_argument("--out", default="", help="write the markdown report here")
    args = parser.parse_args(argv)

    if args.cmd == "golden":
        return _golden(args)

    if args.cmd == "retrieval":
        results, queries, spec = run(
            tenant=args.tenant,
            spec_path=Path(args.spec),
            index_root=args.index_root,
            reranker_name=args.reranker,
            corpus_path=args.corpus,
        )
        index = HybridIndex.load(args.tenant, root=args.index_root)
        report = format_report(results, queries, spec, index)
        print(report)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(report, encoding="utf-8")
            print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
