"""FastAPI app factory — the backend of the M7 product surface.

Design lines:

- **Dry by default.** `POST /api/query` uses the fake provider unless the request says
  `live: true` AND the server was started with live mode allowed (`SIAP_ALLOW_LIVE=1`).
  An HTTP endpoint that silently spends Bedrock money on every curious click would violate
  the project's own cost discipline, so spending requires both the caller and the operator
  to opt in.
- **Injectable provider.** Tests pass a `provider_factory`; production uses the default
  (FakeProvider for dry, BedrockProvider for live). Same pattern as every CLI here.
- **Timestamps at the boundary.** The deterministic core requires caller-supplied
  timestamps (`datetime.now()` is banned there). The API is the serving boundary, so it
  stamps wall-clock time here — or accepts `ts` in the request, which is what tests do.
- **Read endpoints read artifacts, not state.** Promotions, the golden report and the M6
  curve are served from the same committed files the CLIs write, so the dashboard shows
  exactly what the repo ships.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.agent.loop import AgentConfig
from src.llm.base import LLMProvider
from src.llm.pricing import CHEAP, PRICING_AS_OF, STRONG

ProviderFactory = Callable[[bool], LLMProvider]


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    live: bool = False
    router: str = "active"
    tenant: str = "duckdb"
    spend_limit_usd: float = Field(default=0.25, gt=0, le=1.0)
    ts: str | None = None  # tests inject; production stamps wall clock at the boundary


def _default_provider_factory(live: bool) -> LLMProvider:
    if live:
        from src.llm.bedrock import BedrockProvider

        return BedrockProvider()
    # Dry mode does REAL retrieval with templated prose (see retrieval_demo_provider):
    # a canned refusal here made every demo answer ungrounded, which then escalated —
    # a worst-case impression of a system whose whole point is grounding.
    from src.llm.fake import retrieval_demo_provider

    return retrieval_demo_provider()


def create_app(
    provider_factory: ProviderFactory = _default_provider_factory,
    trace_db: str | Path = "data/traces.db",
    index_root: str = "data/index",
    corpus: str = "data/corpus/duckdb",
    allow_live: bool | None = None,
    configs_dir: str | Path = "configs",
    sim_weekly: str | Path = "eval/sim/weekly.json",
    golden_records: str | Path = "eval/golden/records.json",
    golden_spec: str | Path = "eval/golden/duckdb.yaml",
    golden_threshold: float = 0.75,
) -> FastAPI:
    import os

    if allow_live is None:
        allow_live = os.environ.get("SIAP_ALLOW_LIVE", "") == "1"

    app = FastAPI(title="self-improving-agent-platform", version="0.7.0")
    app.add_middleware(
        CORSMiddleware,
        # The Next.js dev server. Not "*": this API can spend money when live is enabled.
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    configs_dir = Path(configs_dir)

    def promotion_log():
        from src.flywheel.promote import PromotionLog

        return PromotionLog(
            log_path=configs_dir / "promotions.jsonl",
            active_path=configs_dir / "active.json",
        )

    def trace_store():
        from src.ops.trace_store import TraceStore

        return TraceStore(trace_db)

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "cheap": CHEAP.model_id,
            "strong": STRONG.model_id,
            "pricing_as_of": PRICING_AS_OF,
            "live_enabled": allow_live,
            "active": promotion_log().active(),
        }

    @app.post("/api/query")
    def query(req: QueryRequest) -> dict[str, Any]:
        from src.agent.__main__ import build_agent

        if req.live and not allow_live:
            raise HTTPException(
                status_code=403,
                detail="live mode disabled — start the server with SIAP_ALLOW_LIVE=1",
            )
        live = req.live and allow_live
        provider = provider_factory(live)
        config = AgentConfig(spend_limit_usd=req.spend_limit_usd)
        try:
            agent, _ = build_agent(
                provider, req.tenant, index_root, corpus, req.router, config
            )
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=503, detail=f"index not built — run `make ingest` ({exc})"
            ) from exc
        t0 = time.perf_counter()
        run = agent.run_detailed(req.question, tenant=req.tenant)
        ts = req.ts or time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        with trace_store() as store:
            store.write(run.trace, ts=ts, guard_action=run.guard_action)
        rep = run.citation_report
        return {
            "trace_id": run.trace.trace_id,
            "answer": run.answer,
            "citations": rep.to_dict(),
            "grounded": rep.grounded,
            "tier": run.routing.tier,
            "escalated": run.routing.escalated,
            "routing_reason": run.routing.reason,
            "tools": run.tool_calls,
            "iterations": run.iterations,
            "guard_action": run.guard_action,
            "guard_events": run.guard_events,
            "cost": run.cost,
            "latency_ms": run.trace.latency_ms,
            "wall_s": round(time.perf_counter() - t0, 3),
            "live": live,
            "fabricated": not live,  # dry-run numbers are fake and say so, as everywhere here
        }

    @app.get("/api/traces")
    def traces(limit: int = 20, tenant: str | None = None) -> list[dict[str, Any]]:
        with trace_store() as store:
            return store.recent(limit=min(limit, 200), tenant=tenant)

    @app.get("/api/traces/{trace_id}")
    def trace(trace_id: str) -> dict[str, Any]:
        with trace_store() as store:
            row = store.get(trace_id)
        if row is None:
            raise HTTPException(status_code=404, detail="no such trace")
        return row

    @app.get("/api/summary")
    def summary(tenant: str | None = None) -> dict[str, Any]:
        with trace_store() as store:
            return store.summary(tenant=tenant)

    @app.get("/api/promotions")
    def promotions() -> dict[str, Any]:
        log = promotion_log()
        return {"active": log.active(), "entries": log.entries()}

    @app.get("/api/sim/weekly")
    def sim_weekly_data() -> list[dict[str, Any]]:
        path = Path(sim_weekly)
        if not path.exists():
            raise HTTPException(status_code=404, detail="no simulation has been run")
        return json.loads(path.read_text())

    @app.get("/api/golden")
    def golden() -> dict[str, Any]:
        from src.eval.golden import gate_from_records, load_cases, load_records

        records_path = Path(golden_records)
        if not records_path.exists():
            raise HTTPException(status_code=404, detail="no golden records recorded")
        cases, _meta = load_cases(Path(golden_spec))
        report = gate_from_records(load_records(records_path), cases, golden_threshold)
        return {
            "score": report.score,
            "passed": report.passed,
            "threshold": golden_threshold,
            "by_kind": {k: list(v) for k, v in report.by_kind().items()},
            "cases": [c.to_dict() for c in report.results],
        }

    return app
