"""The M7 API surface, offline. FastAPI TestClient + fake provider — no network, no spend."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.index.store import HybridIndex
from src.llm.fake import FakeProvider, tool_turn
from src.types import Chunk, stable_id

QUALIFY = stable_id("duckdb", "qualify")


@pytest.fixture
def client(tmp_path) -> TestClient:
    index = HybridIndex("duckdb", embedder_name="hashing", root=tmp_path / "index")
    index.add(
        [
            Chunk(
                chunk_id=QUALIFY,
                doc_id="qualify",
                tenant="duckdb",
                text="QUALIFY filters the output of a window function.",
                source_path="sql/qualify.md",
                heading_path=("DuckDB", "QUALIFY"),
            )
        ]
    )
    index.save()

    def factory(live: bool) -> FakeProvider:
        assert not live, "tests must never construct a live provider"
        return FakeProvider(
            [
                tool_turn("search_docs", {"query": "qualify"}),
                f"QUALIFY filters window output [{QUALIFY}].",
                "LGTM",  # critic pass (anything not starting with REVISE keeps the answer)
            ]
        )

    app = create_app(
        provider_factory=factory,
        trace_db=tmp_path / "traces.db",
        index_root=str(tmp_path / "index"),
        corpus=str(tmp_path / "nocorpus"),  # absent -> no link graph, and that's fine
        allow_live=False,
        configs_dir=tmp_path / "configs",
        sim_weekly=tmp_path / "weekly.json",
        golden_records=tmp_path / "records.json",
    )
    return TestClient(app)


def test_health_reports_models_and_active_config(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["live_enabled"] is False
    assert body["active"]["router"]["kind"] == "heuristic"  # nothing promoted in tmp configs


def test_query_answers_grounded_and_persists_a_trace(client):
    r = client.post(
        "/api/query",
        json={"question": "how do I filter a window function", "ts": "2026-07-23T00:00:00"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["grounded"] is True
    assert body["fabricated"] is True  # dry numbers are labelled, as everywhere here
    assert QUALIFY in body["citations"]["cited_ids"]

    # the trace is durable and retrievable
    row = client.get(f"/api/traces/{body['trace_id']}").json()
    assert row["query"] == "how do I filter a window function"
    listed = client.get("/api/traces").json()
    assert listed and listed[0]["trace_id"] == body["trace_id"]
    assert client.get("/api/summary").json()["n"] == 1


def test_live_requires_operator_opt_in(client):
    r = client.post("/api/query", json={"question": "q", "live": True})
    assert r.status_code == 403  # SIAP_ALLOW_LIVE unset -> the server refuses to spend


def test_unknown_trace_404s(client):
    assert client.get("/api/traces/nope").status_code == 404


def test_sim_weekly_404s_before_any_simulation(client):
    assert client.get("/api/sim/weekly").status_code == 404


def test_promotions_empty_log_serves_default_active(client):
    body = client.get("/api/promotions").json()
    assert body["entries"] == []
    assert body["active"]["router"]["version"] == "heuristic-v0"


def test_golden_replays_committed_records(tmp_path):
    # Uses the real committed spec + records — the same artifact the CI gate scores.
    app = create_app(
        provider_factory=lambda live: (_ for _ in ()).throw(AssertionError("no provider")),
        trace_db=tmp_path / "t.db",
        index_root=str(tmp_path / "no-index"),
        configs_dir=tmp_path / "configs",
    )
    body = TestClient(app).get("/api/golden").json()
    assert body["passed"] is True
    assert body["score"] >= body["threshold"]
    assert len(body["cases"]) == 12  # the committed 12-case golden set


def test_sim_weekly_serves_written_artifact(tmp_path):
    weekly = tmp_path / "weekly.json"
    weekly.write_text(json.dumps([{"week": 0, "grounded": 11, "n_queries": 12}]))
    app = create_app(
        trace_db=tmp_path / "t.db",
        index_root=str(tmp_path / "no-index"),
        configs_dir=tmp_path / "configs",
        sim_weekly=weekly,
    )
    body = TestClient(app).get("/api/sim/weekly").json()
    assert body[0]["week"] == 0


def test_query_without_index_is_503_not_500(tmp_path):
    app = create_app(
        provider_factory=lambda live: FakeProvider(["x"]),
        trace_db=tmp_path / "t.db",
        index_root=str(tmp_path / "no-index"),
        configs_dir=tmp_path / "configs",
    )
    r = TestClient(app).post("/api/query", json={"question": "q"})
    assert r.status_code == 503


def test_demo_provider_grounds_answers_on_real_retrieval(tmp_path):
    """The dry-mode default must produce a grounded, cited answer from the real index —
    a canned refusal here made every demo answer ungrounded (and then escalate)."""
    from src.llm.fake import retrieval_demo_provider

    index = HybridIndex("duckdb", embedder_name="hashing", root=tmp_path / "index")
    index.add(
        [
            Chunk(
                chunk_id=QUALIFY,
                doc_id="qualify",
                tenant="duckdb",
                text="QUALIFY filters the output of a window function. Use it after WINDOW.",
                source_path="sql/qualify.md",
                heading_path=("DuckDB", "QUALIFY"),
            )
        ]
    )
    index.save()
    app = create_app(
        provider_factory=lambda live: retrieval_demo_provider(),
        trace_db=tmp_path / "t.db",
        index_root=str(tmp_path / "index"),
        corpus=str(tmp_path / "nocorpus"),
        allow_live=False,
        configs_dir=tmp_path / "configs",
    )
    body = TestClient(app).post(
        "/api/query",
        json={"question": "how do I filter a window function", "ts": "2026-07-23T00:00:00"},
    ).json()
    assert body["grounded"] is True
    assert body["escalated"] is False
    assert QUALIFY in body["citations"]["cited_ids"]
    assert "QUALIFY" in body["answer"]
