"""Simulator CLI.

    python -m src.sim run --weeks 6 --dry-run      # free, fake provider
    python -m src.sim run --weeks 6 --live         # real Bedrock (SPENDS ~$1.5-2)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.sim")
    sub = parser.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run N simulated weeks")
    r.add_argument("--weeks", type=int, default=6)
    r.add_argument("--k", type=int, default=12, help="queries per week")
    r.add_argument("--shadow-budget", type=int, default=4)
    r.add_argument("--state-dir", default="data/sim")
    r.add_argument("--spend-limit", type=float, default=0.15)
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--live", action="store_true", help="explicit ack that this spends")
    r.add_argument("--fresh", action="store_true", help="wipe the state dir first")
    args = parser.parse_args(argv)

    if not (args.dry_run or args.live):
        print("the simulator spends money: pass --live to confirm, or --dry-run",
              file=sys.stderr)
        return 2

    state_dir = Path(args.state_dir)
    if args.fresh and state_dir.exists():
        import shutil

        shutil.rmtree(state_dir)

    if args.dry_run:
        from src.llm.fake import FakeProvider

        provider = FakeProvider(["The documentation does not cover this."] * 5000)
    else:
        from src.llm.bedrock import BedrockProvider

        provider = BedrockProvider()

    from src.agent.__main__ import build_agent
    from src.agent.loop import AgentConfig

    cfg = AgentConfig(spend_limit_usd=args.spend_limit, critic=False)

    def agent_factory(prov, router):
        agent, _ = build_agent(prov, "duckdb", "data/index", "data/corpus/duckdb",
                               "heuristic", cfg)
        agent.router = router  # the simulator owns routing; build_agent's choice is replaced
        return agent

    from src.sim.simulator import Simulator

    sim = Simulator(
        state_dir=state_dir,
        provider=provider,
        agent_factory=agent_factory,
        k_per_week=args.k,
        shadow_budget=args.shadow_budget,
        spend_limit_usd=args.spend_limit,
    )
    reports = sim.run(args.weeks)
    total = sum(r.cost_usd + r.shadow_cost_usd for r in reports)
    print(f"\n{args.weeks} weeks simulated, total ${total:.4f} -> {state_dir}/weekly.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
