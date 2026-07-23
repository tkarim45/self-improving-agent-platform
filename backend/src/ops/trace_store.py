"""Per-request trace persistence to SQLite.

The `Trace` dataclass (src/types.py) already carries everything one request did — prompt,
tokens, cost, latency, retrieval set, tier, scores. M3 makes it durable: every request is
written to a SQLite table, which becomes the flywheel's raw input in M5 (mine low-scoring
traces) and the source for the M6 improvement curve (quality and cost over time).

Two columns are stored both flat and as JSON. The flat columns (cost, latency, tier, tenant,
scores-as-json) are what dashboards and drift checks query; the full JSON blob is the
loss-free record, so a schema that grows in M4/M5 does not orphan older traces.

WHY IT WRITES REDACTED TEXT: the query stored here has already passed the input guard, so a
credential a user pasted was replaced with a placeholder before it reached this table. The
trace store trusts that boundary rather than re-implementing redaction — but it never sees
the raw secret in the first place because the guard runs first (see src/agent/loop.py).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.types import Trace

SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    trace_id       TEXT PRIMARY KEY,
    ts             TEXT NOT NULL,
    tenant         TEXT NOT NULL,
    query          TEXT NOT NULL,
    answer         TEXT NOT NULL,
    model_tier     TEXT NOT NULL,
    input_tokens   INTEGER NOT NULL,
    output_tokens  INTEGER NOT NULL,
    cost_usd       REAL NOT NULL,
    latency_ms     REAL NOT NULL,
    config_version TEXT NOT NULL,
    grounded       INTEGER NOT NULL,
    citation_rate  REAL NOT NULL,
    invalid_cites  INTEGER NOT NULL,
    escalated      INTEGER NOT NULL,
    guard_action   TEXT NOT NULL DEFAULT 'allow',
    error          TEXT,
    payload        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_traces_ts ON traces(ts);
CREATE INDEX IF NOT EXISTS idx_traces_tenant ON traces(tenant);
"""


class TraceStore:
    def __init__(self, path: str | Path = "data/traces.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> TraceStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def write(self, trace: Trace, ts: str, guard_action: str = "allow") -> None:
        """Persist one trace. `ts` is passed in rather than generated, because the workflow
        engine (and tests) must control time — `datetime.now()` would make runs non-repeatable
        and is banned in this codebase's deterministic paths."""
        scores = trace.scores or {}
        retrieved = set(trace.retrieved)
        invalid = sum(1 for c in trace.citations if c.chunk_id not in retrieved)
        self._conn.execute(
            """INSERT OR REPLACE INTO traces
               (trace_id, ts, tenant, query, answer, model_tier, input_tokens,
                output_tokens, cost_usd, latency_ms, config_version, grounded,
                citation_rate, invalid_cites, escalated, guard_action, error, payload)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                trace.trace_id,
                ts,
                trace.tenant,
                trace.query,
                trace.answer,
                trace.model_tier,
                trace.input_tokens,
                trace.output_tokens,
                trace.cost_usd,
                trace.latency_ms,
                trace.config_version,
                int(bool(scores.get("grounded"))),
                float(scores.get("citation_rate", 0.0)),
                invalid,
                int(bool(scores.get("escalated"))),
                guard_action,
                trace.error,
                json.dumps(asdict(trace), default=str),
            ),
        )
        self._conn.commit()

    def get(self, trace_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM traces WHERE trace_id = ?", (trace_id,)
        ).fetchone()
        return dict(row) if row else None

    def recent(self, limit: int = 20, tenant: str | None = None) -> list[dict[str, Any]]:
        if tenant:
            rows = self._conn.execute(
                "SELECT * FROM traces WHERE tenant = ? ORDER BY ts DESC LIMIT ?",
                (tenant, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM traces ORDER BY ts DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def summary(self, tenant: str | None = None) -> dict[str, Any]:
        where, args = ("WHERE tenant = ?", (tenant,)) if tenant else ("", ())
        row = self._conn.execute(
            f"""SELECT
                    COUNT(*) AS n,
                    COALESCE(SUM(cost_usd), 0) AS total_cost,
                    COALESCE(AVG(cost_usd), 0) AS mean_cost,
                    COALESCE(AVG(latency_ms), 0) AS mean_latency,
                    COALESCE(AVG(grounded), 0) AS grounded_rate,
                    COALESCE(AVG(citation_rate), 0) AS mean_citation_rate,
                    COALESCE(SUM(escalated), 0) AS escalations,
                    COALESCE(SUM(CASE WHEN guard_action='block' THEN 1 ELSE 0 END), 0) AS blocks,
                    COALESCE(SUM(CASE WHEN guard_action='redact' THEN 1 ELSE 0 END), 0)
                        AS redactions
                FROM traces {where}""",
            args,
        ).fetchone()
        d = dict(row)
        by_tier = self._conn.execute(
            f"SELECT model_tier, COUNT(*) n, SUM(cost_usd) cost FROM traces {where} "
            "GROUP BY model_tier",
            args,
        ).fetchall()
        d["by_tier"] = {r["model_tier"]: {"n": r["n"], "cost": r["cost"]} for r in by_tier}
        return d
