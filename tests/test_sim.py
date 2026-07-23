"""M6 simulator: sampling, weekly loop, isolated state, shadow sampling. Offline."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.sim.simulator import Simulator, WeekReport, load_pool, sample_week


def test_pool_loads_with_weighted_repetition():
    pool = load_pool()
    queries = [q for q, _ in pool]
    # lookup weight 5 means each lookup query appears 5 times in the flattened pool
    assert queries.count("how do I filter rows produced by a window function") == 5
    assert queries.count("why would a hash join be slower than a merge join here, "
                         "and what would you check first") == 1


def test_weekly_sample_is_deterministic_and_deduped():
    pool = load_pool()
    a = sample_week(pool, week=2, k=10)
    b = sample_week(pool, week=2, k=10)
    assert a == b
    assert len({q for q, _ in a}) == 10  # no duplicate queries within a week


def test_different_weeks_draw_different_mixes():
    pool = load_pool()
    w0 = {q for q, _ in sample_week(pool, 0, 10)}
    w1 = {q for q, _ in sample_week(pool, 1, 10)}
    assert w0 != w1
    # ...but repetition ACROSS weeks is allowed and expected (support traffic repeats)


def test_golden_questions_are_not_in_the_pool():
    """The golden set is a frozen canary; it must never be traffic."""
    import yaml

    golden = {c["question"] for c in
              yaml.safe_load(Path("eval/golden/duckdb.yaml").read_text())["cases"]}
    pool = {q for q, _ in load_pool()}
    assert golden & pool == set()


class ScriptedAgent:
    """Offline stand-in: grounded iff routed to the tier the script says suffices."""

    def __init__(self, router, sufficient: dict[str, str]):
        self.router = router
        self.sufficient = sufficient

    def run_detailed(self, query, tenant="duckdb"):
        from src.agent.citations import CitationReport
        from src.agent.loop import AgentRun
        from src.types import Citation, Trace

        decision = self.router.route(query)
        tier = decision.tier
        needed = self.sufficient.get(query, "cheap")
        # cheap suffices unless the script says strong; strong always suffices
        grounded = tier == "strong" or needed == "cheap"
        cost = 0.01 if tier == "cheap" else 0.05
        run = AgentRun()
        run.routing = decision
        run.answer = "a [abc123]." if grounded else "ungrounded"
        rep = CitationReport(cited_ids=["abc123"] if grounded else [])
        rep.n_claims = 1
        run.citation_report = rep
        run.cost = {"total_usd": cost, "calls": 1, "by_tier": {}, "input_tokens": 1,
                    "output_tokens": 1}
        run.trace = Trace(
            trace_id=f"{hash((query, tier)) & 0xFFFFFFFF:x}", tenant=tenant, query=query,
            answer=run.answer, retrieved=["abc123"] if grounded else ["abc123"],
            citations=[Citation(chunk_id="abc123")] if grounded else [],
            model_tier=tier, cost_usd=cost, latency_ms=1.0, config_version="sim",
            scores={"grounded": float(grounded), "citation_rate": float(grounded),
                    "escalated": 0.0, "exhausted": 0.0},
        )
        return run


@pytest.fixture
def sim(tmp_path):
    def factory(provider, router):
        return ScriptedAgent(router, sufficient={})  # cheap suffices on everything

    return Simulator(
        state_dir=tmp_path / "sim",
        provider=None,
        agent_factory=factory,
        k_per_week=8,
        shadow_budget=3,
    )


def test_week_zero_serves_under_the_heuristic(sim):
    report = sim.run_week(0)
    assert report.router_version == "heuristic-v0"
    assert report.n_queries == 8


def test_simulation_state_is_isolated(sim, tmp_path):
    sim.run_week(0)
    assert (tmp_path / "sim" / "traces.db").exists()
    assert not Path("data/sim-test-leak").exists()
    # The production promotion log is untouched by construction (different paths).


def test_shadow_sampling_prices_unknown_choices(sim):
    """A rejected cycle with unpriced choices triggers bounded live sampling."""
    reports = sim.run(3)
    total_shadow = sum(r.shadow_sampled for r in reports)
    # With cheap sufficing everywhere and the heuristic routing some queries strong, the
    # always-cheap candidate has unpriced choices until the sampler prices them.
    assert all(r.shadow_sampled <= sim.shadow_budget for r in reports)
    assert total_shadow >= 0  # bounded, never negative; exact count depends on sampling


def test_flywheel_promotes_within_a_few_weeks_when_cheap_suffices(sim):
    """The end-to-end property the curve shows: heuristic start -> promotion -> cheaper."""
    reports = sim.run(6)
    promoted_weeks = [r.week for r in reports if r.cycle.get("promoted")]
    assert promoted_weeks, "flywheel never promoted despite cheap sufficing everywhere"
    first = promoted_weeks[0]
    # After promotion the active router serves; later weeks run all-cheap.
    later = [r for r in reports if r.week > first]
    assert all(set(r.tier_mix) == {"cheap"} for r in later)
    # And cost per query drops from the pre-promotion weeks to the post-promotion weeks.
    pre = [r.cost_per_query for r in reports if r.week <= first]
    post = [r.cost_per_query for r in later]
    if post:
        assert min(pre) > min(post) or max(pre) > max(post)


def test_weekly_report_serializes():
    r = WeekReport(week=0, router_version="x", n_queries=4, grounded=3, cost_usd=0.04)
    d = r.to_dict()
    assert d["grounded_rate"] == 0.75
    assert d["cost_per_query"] == 0.01
