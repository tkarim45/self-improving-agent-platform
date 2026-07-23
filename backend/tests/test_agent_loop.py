"""The full agent loop, offline. No credentials, no spend."""

from __future__ import annotations

import pytest

from src.agent.loop import AgentConfig, GroundedAgent
from src.agent.router import AlwaysRouter, HeuristicRouter
from src.agent.tools import RunSqlTool, SearchDocsTool
from src.index.store import HybridIndex
from src.llm.base import CostMeter, SpendLimitExceeded
from src.llm.fake import FakeProvider, tool_turn
from src.llm.pricing import spec_for
from src.retrieval.pipeline import HybridRetriever, RetrievalConfig
from src.types import Chunk, stable_id

DOCS = {
    "qualify": "QUALIFY filters the output of a window function, like HAVING does for GROUP BY.",
    "pivot": "The PIVOT statement rotates rows into columns using an aggregate expression.",
}


@pytest.fixture
def search_tool(tmp_path) -> SearchDocsTool:
    index = HybridIndex("duckdb", embedder_name="hashing", root=tmp_path)
    index.add(
        [
            Chunk(
                chunk_id=stable_id("duckdb", name),
                doc_id=name,
                tenant="duckdb",
                text=text,
                source_path=f"sql/{name}.md",
                heading_path=("DuckDB", name.upper()),
            )
            for name, text in DOCS.items()
        ]
    )
    return SearchDocsTool(HybridRetriever(index, RetrievalConfig(mode="bm25")), tenant="duckdb")


def cid(name: str) -> str:
    return stable_id("duckdb", name)


# --- tool dispatch --------------------------------------------------------------------


def test_agent_calls_search_then_answers(search_tool):
    provider = FakeProvider(
        [
            tool_turn("search_docs", {"query": "qualify window filter"}),
            lambda msgs: f"QUALIFY filters window function output [{cid('qualify')}].",
        ]
    )
    agent = GroundedAgent(provider, search_tool, config=AgentConfig(critic=False))
    run = agent.run_detailed("how do I filter a window function")

    assert run.tool_calls == ["search_docs"]
    assert run.citation_report.grounded
    assert run.iterations == 2


def test_search_results_are_fed_back_as_tool_result(search_tool):
    provider = FakeProvider(
        [tool_turn("search_docs", {"query": "qualify"}), f"Answer [{cid('qualify')}]."]
    )
    GroundedAgent(provider, search_tool, config=AgentConfig(critic=False)).run_detailed("q")

    second = provider.requests[1]["messages"]
    tool_result = second[-1]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert "QUALIFY filters" in tool_result["content"]


def test_parallel_tool_results_go_back_in_one_message(search_tool):
    """Splitting them across messages trains the model out of parallel calls."""
    from src.llm.base import LLMResponse, ToolCall

    both = LLMResponse(
        text="",
        tool_calls=[
            ToolCall(id="t1", name="search_docs", arguments={"query": "qualify"}),
            ToolCall(id="t2", name="run_sql", arguments={"sql": "SELECT 1"}),
        ],
        stop_reason="tool_use",
    )
    provider = FakeProvider([both, f"Done [{cid('qualify')}]."])
    agent = GroundedAgent(provider, search_tool, RunSqlTool(), config=AgentConfig(critic=False))
    agent.run_detailed("q")

    user_msgs = [m for m in provider.requests[1]["messages"] if m["role"] == "user"]
    assert len(user_msgs[-1]["content"]) == 2


DISPATCH_ONLY = AgentConfig(critic=False, escalate_on_ungrounded=False)


def test_unknown_tool_returns_an_error_the_model_can_recover_from(search_tool):
    provider = FakeProvider([tool_turn("nope", {}), "Sorry, I could not look that up."])
    agent = GroundedAgent(provider, search_tool, config=DISPATCH_ONLY)
    agent.run_detailed("q")
    assert "no tool named" in provider.requests[1]["messages"][-1]["content"][0]["content"]


def test_bad_tool_arguments_do_not_crash_the_loop(search_tool):
    provider = FakeProvider([tool_turn("search_docs", {"wrong": 1}), "Could not search."])
    agent = GroundedAgent(provider, search_tool, config=DISPATCH_ONLY)
    run = agent.run_detailed("q")
    assert "bad arguments" in provider.requests[1]["messages"][-1]["content"][0]["content"]
    assert run.answer == "Could not search."


def test_iteration_budget_is_enforced_per_attempt(search_tool):
    provider = FakeProvider([tool_turn("search_docs", {"query": "x"})] * 10)
    agent = GroundedAgent(
        provider,
        search_tool,
        config=AgentConfig(critic=False, escalate_on_ungrounded=False, max_iterations=3),
    )
    run = agent.run_detailed("q")
    assert run.iterations == 3
    assert run.exhausted
    assert "could not finish" in run.answer.lower()


def test_run_wide_call_ceiling_bounds_escalation_and_critic(search_tool):
    """max_iterations bounds one conversation; max_llm_calls bounds the whole run.

    Without the second ceiling a run could spend max_iterations*2 + 1 + max_iterations
    calls — observed at 10 calls / $0.22 on a real question the corpus cannot answer.
    """
    provider = FakeProvider([tool_turn("search_docs", {"query": "x"})] * 30)
    agent = GroundedAgent(
        provider,
        search_tool,
        config=AgentConfig(max_iterations=6, max_llm_calls=4, critic=True),
        router=AlwaysRouter("cheap"),
    )
    run = agent.run_detailed("q")
    assert provider.call_count <= 4
    assert run.exhausted


def test_exhaustion_escalates_and_the_trace_says_why(search_tool):
    """Budget exhaustion runs the whole loop again on the dearer tier — roughly double the
    spend. Pinned so that cost is a recorded decision, not a surprise on the bill."""
    provider = FakeProvider(
        [tool_turn("search_docs", {"query": "x"})] * 3
        + [tool_turn("search_docs", {"query": "x"}), f"Found it [{cid('qualify')}]."]
    )
    agent = GroundedAgent(
        provider,
        search_tool,
        config=AgentConfig(critic=False, max_iterations=3),
        router=AlwaysRouter("cheap"),
    )
    run = agent.run_detailed("q")

    assert run.routing.escalated
    assert "exhausted its tool budget" in run.routing.reason
    assert run.iterations == 5  # 3 on cheap, then 2 on strong
    assert set(provider.tiers_used()) == {"cheap", "strong"}
    assert not run.exhausted  # the retry finished


# --- routing --------------------------------------------------------------------------


def test_heuristic_routes_lookup_cheap_and_reasoning_strong():
    router = HeuristicRouter()
    assert router.route("what is the syntax for PIVOT").tier == "cheap"
    assert router.route("why is my hash join slower than a merge join here").tier == "strong"


def test_routing_decision_explains_itself():
    decision = HeuristicRouter().route("why is this slow and how do I compare the two plans")
    assert "score" in decision.reason and decision.score > 0


def test_ungrounded_cheap_answer_escalates_to_strong(search_tool):
    """Escalation is triggered by a measured grounding failure, not a guess."""
    provider = FakeProvider(
        [
            "QUALIFY does something I made up.",  # cheap tier: no citations
            tool_turn("search_docs", {"query": "qualify"}),
            f"QUALIFY filters window output [{cid('qualify')}].",
        ]
    )
    agent = GroundedAgent(
        provider, search_tool, config=AgentConfig(critic=False), router=AlwaysRouter("cheap")
    )
    run = agent.run_detailed("what does QUALIFY do")

    assert run.routing.escalated
    assert run.routing.tier == "strong"
    assert provider.tiers_used() == ["cheap", "strong", "strong"]
    assert run.citation_report.grounded


def test_grounded_cheap_answer_does_not_escalate(search_tool):
    provider = FakeProvider(
        [tool_turn("search_docs", {"query": "qualify"}), f"Filters windows [{cid('qualify')}]."]
    )
    agent = GroundedAgent(
        provider, search_tool, config=AgentConfig(critic=False), router=AlwaysRouter("cheap")
    )
    run = agent.run_detailed("q")
    assert not run.routing.escalated
    assert set(provider.tiers_used()) == {"cheap"}


def test_strong_tier_does_not_escalate_further(search_tool):
    provider = FakeProvider(["An ungrounded answer with no citations at all."])
    agent = GroundedAgent(
        provider, search_tool, config=AgentConfig(critic=False), router=AlwaysRouter("strong")
    )
    run = agent.run_detailed("q")
    assert not run.routing.escalated
    assert provider.call_count == 1


# --- critic ---------------------------------------------------------------------------


def test_critic_pass_leaves_the_answer_alone(search_tool):
    provider = FakeProvider(
        [
            tool_turn("search_docs", {"query": "qualify"}),
            f"QUALIFY filters window output [{cid('qualify')}].",
            "PASS",
        ]
    )
    agent = GroundedAgent(provider, search_tool, config=AgentConfig(critic=True))
    run = agent.run_detailed("q")
    assert not run.revised
    assert run.answer.startswith("QUALIFY filters")


def test_critic_revise_triggers_a_rewrite(search_tool):
    provider = FakeProvider(
        [
            tool_turn("search_docs", {"query": "qualify"}),
            f"QUALIFY filters output [{cid('qualify')}]. It also does something uncited.",
            "REVISE\n- the second sentence is uncited",
            f"QUALIFY filters the output of a window function [{cid('qualify')}].",
        ]
    )
    agent = GroundedAgent(provider, search_tool, config=AgentConfig(critic=True))
    run = agent.run_detailed("q")
    assert run.revised
    assert "uncited" not in run.answer


def test_critic_is_skipped_when_nothing_was_cited(search_tool):
    """No citations means grounding already failed — a critic pass would just cost money."""
    provider = FakeProvider(["No citations here at all in this reply."])
    agent = GroundedAgent(
        provider,
        search_tool,
        config=AgentConfig(critic=True, escalate_on_ungrounded=False),
        router=AlwaysRouter("cheap"),
    )
    agent.run_detailed("q")
    assert provider.call_count == 1


# --- cost -----------------------------------------------------------------------------


def test_cost_is_computed_from_reported_tokens():
    meter = CostMeter(spend_limit_usd=10.0)
    provider = FakeProvider(["hi"], tokens_per_call=1_000_000)
    response = provider.generate("s", [{"role": "user", "content": "q"}], tier="cheap")
    meter.record(response)
    # 1M input @ $1/MTok + 250k output @ $5/MTok = 1.00 + 1.25
    assert response.cost_usd == pytest.approx(2.25)
    assert meter.summary()["calls"] == 1


def test_strong_tier_costs_more_than_cheap_for_identical_usage():
    assert spec_for("strong").cost(1000, 1000) > spec_for("cheap").cost(1000, 1000)


def test_spend_limit_raises_rather_than_continuing():
    meter = CostMeter(spend_limit_usd=0.001)
    provider = FakeProvider(["a", "b"], tokens_per_call=1_000_000)
    with pytest.raises(SpendLimitExceeded):
        for _ in range(2):
            meter.record(provider.generate("s", [], tier="strong"))


def test_run_records_cost_and_trace(search_tool):
    provider = FakeProvider(
        [tool_turn("search_docs", {"query": "qualify"}), f"Answer [{cid('qualify')}]."]
    )
    agent = GroundedAgent(provider, search_tool, config=AgentConfig(critic=False))
    run = agent.run_detailed("q")

    assert run.cost["calls"] == 2
    assert run.trace.cost_usd > 0
    assert run.trace.model_tier in ("cheap", "strong")
    assert run.trace.citations and run.trace.retrieved


def test_fake_provider_refuses_to_improvise_when_script_runs_dry(search_tool):
    provider = FakeProvider([tool_turn("search_docs", {"query": "x"})])
    agent = GroundedAgent(provider, search_tool, config=AgentConfig(critic=False))
    with pytest.raises(AssertionError, match="script exhausted"):
        agent.run_detailed("q")


def test_fake_token_counts_are_fabricated_not_measured():
    """Pinned so a fake-run number can never be quoted as a real cost."""
    provider = FakeProvider(["x"], tokens_per_call=100)
    r = provider.generate("s", [], tier="cheap")
    assert r.model_id.startswith("fake:")
    assert (r.input_tokens, r.output_tokens) == (100, 25)
