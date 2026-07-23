"""Trace store + provider failover. Offline."""

from __future__ import annotations

import pytest

from src.llm.base import LLMResponse, SpendLimitExceeded
from src.llm.fake import FakeProvider
from src.ops.failover import AllProvidersFailed, FailoverProvider
from src.ops.trace_store import TraceStore
from src.types import Citation, Trace


def make_trace(tid="t1", tier="cheap", cost=0.01, grounded=True, retrieved=("c1",), cited=("c1",)):
    return Trace(
        trace_id=tid,
        tenant="duckdb",
        query="q",
        answer="a",
        retrieved=list(retrieved),
        citations=[Citation(chunk_id=c) for c in cited],
        model_tier=tier,
        input_tokens=100,
        output_tokens=25,
        cost_usd=cost,
        latency_ms=120.0,
        config_version="m3",
        scores={"grounded": float(grounded), "citation_rate": 0.8 if grounded else 0.0},
    )


# --- trace store ----------------------------------------------------------------------


def test_write_and_read_back(tmp_path):
    store = TraceStore(tmp_path / "t.db")
    store.write(make_trace(), ts="2026-07-22T00:00:00")
    row = store.get("t1")
    assert row["model_tier"] == "cheap"
    assert row["grounded"] == 1
    assert row["cost_usd"] == 0.01


def test_full_payload_survives_as_json(tmp_path):
    """The flat columns are for queries; the JSON blob is the loss-free record."""
    import json

    store = TraceStore(tmp_path / "t.db")
    store.write(make_trace(), ts="2026-07-22T00:00:00")
    payload = json.loads(store.get("t1")["payload"])
    assert payload["citations"][0]["chunk_id"] == "c1"


def test_invalid_citations_are_counted(tmp_path):
    store = TraceStore(tmp_path / "t.db")
    store.write(make_trace(retrieved=("c1",), cited=("c1", "ghost")), ts="2026-07-22T00:00:00")
    assert store.get("t1")["invalid_cites"] == 1


def test_summary_aggregates_cost_and_grounding(tmp_path):
    store = TraceStore(tmp_path / "t.db")
    store.write(make_trace("a", tier="cheap", cost=0.01, grounded=True), ts="2026-07-22T00:00:00")
    store.write(make_trace("b", tier="strong", cost=0.10, grounded=False), ts="2026-07-22T00:01:00")
    s = store.summary()
    assert s["n"] == 2
    assert s["total_cost"] == pytest.approx(0.11)
    assert s["grounded_rate"] == pytest.approx(0.5)
    assert s["by_tier"]["strong"]["cost"] == pytest.approx(0.10)


def test_guard_action_is_persisted(tmp_path):
    store = TraceStore(tmp_path / "t.db")
    store.write(make_trace("a"), ts="2026-07-22T00:00:00", guard_action="block")
    store.write(make_trace("b"), ts="2026-07-22T00:01:00", guard_action="redact")
    s = store.summary()
    assert s["blocks"] == 1 and s["redactions"] == 1


def test_recent_orders_newest_first(tmp_path):
    store = TraceStore(tmp_path / "t.db")
    store.write(make_trace("old"), ts="2026-07-22T00:00:00")
    store.write(make_trace("new"), ts="2026-07-22T00:05:00")
    assert [r["trace_id"] for r in store.recent()] == ["new", "old"]


def test_write_is_idempotent_on_trace_id(tmp_path):
    store = TraceStore(tmp_path / "t.db")
    store.write(make_trace("t1", cost=0.01), ts="2026-07-22T00:00:00")
    store.write(make_trace("t1", cost=0.02), ts="2026-07-22T00:00:00")
    assert store.summary()["n"] == 1
    assert store.get("t1")["cost_usd"] == 0.02


# --- failover -------------------------------------------------------------------------


class DeadProvider:
    """A provider that always raises a failover-eligible error."""

    def __init__(self, exc=None):
        self.exc = exc or ConnectionError("primary is down")
        self.calls = 0

    name = "dead"

    def generate(self, *a, **k):
        self.calls += 1
        raise self.exc


def test_dead_primary_falls_through_to_working_secondary():
    dead = DeadProvider()
    live = FakeProvider(["ok"])
    fo = FailoverProvider([dead, live], labels=["primary", "backup"])
    resp = fo.generate("s", [{"role": "user", "content": "q"}], tier="cheap")
    assert resp.text == "ok"
    assert dead.calls == 1
    assert fo.events[0].provider == "primary"


def test_response_records_which_provider_served_after_failover():
    fo = FailoverProvider([DeadProvider(), FakeProvider(["ok"])], labels=["primary", "backup"])
    resp = fo.generate("s", [], tier="cheap")
    assert resp.raw["served_by"] == "backup"
    assert resp.raw["skipped"] == ["primary"]


def test_working_primary_is_used_and_records_nothing():
    live = FakeProvider(["fine"])
    fo = FailoverProvider([live, DeadProvider()])
    resp = fo.generate("s", [], tier="cheap")
    assert resp.text == "fine"
    assert fo.events == []


def test_all_providers_dead_raises_with_the_trail():
    fo = FailoverProvider([DeadProvider(), DeadProvider()], labels=["a", "b"])
    with pytest.raises(AllProvidersFailed) as exc:
        fo.generate("s", [], tier="cheap")
    assert len(exc.value.events) == 2


def test_spend_limit_is_not_failed_over():
    """A budget decision is the caller's, not a provider fault — don't spend elsewhere."""

    class Broke:
        name = "broke"

        def generate(self, *a, **k):
            raise SpendLimitExceeded("over budget")

    fo = FailoverProvider([Broke(), FakeProvider(["ok"])])
    with pytest.raises(SpendLimitExceeded):
        fo.generate("s", [], tier="cheap")


def test_non_failover_error_propagates_immediately():
    """A bug (ValueError) is not a provider outage — don't mask it by trying the next one."""

    class Buggy:
        name = "buggy"

        def generate(self, *a, **k):
            raise ValueError("real bug")

    live = FakeProvider(["ok"])
    fo = FailoverProvider([Buggy(), live])
    with pytest.raises(ValueError):
        fo.generate("s", [], tier="cheap")


def test_bedrock_style_runtime_error_fails_over():
    """BedrockProvider raises RuntimeError when no path connects — that must fail over."""
    fo = FailoverProvider(
        [DeadProvider(RuntimeError("no working Bedrock path")), FakeProvider(["ok"])]
    )
    assert fo.generate("s", [], tier="cheap").text == "ok"


def test_failover_needs_a_provider():
    with pytest.raises(ValueError):
        FailoverProvider([])


def test_failover_generate_returns_llmresponse_type():
    fo = FailoverProvider([FakeProvider(["hi"])])
    assert isinstance(fo.generate("s", [], tier="cheap"), LLMResponse)
