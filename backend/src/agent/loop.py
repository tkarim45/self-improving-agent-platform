"""The agent: plan -> retrieve -> tool -> critic -> answer.

A manual tool-use loop rather than the SDK's beta tool runner. Three reasons: it runs on
Bedrock (where the beta namespace is not a safe assumption), it avoids a beta dependency in
a repo whose point is showing how the loop works, and M5 needs to instrument the loop
directly to mine failures from it.

The loop is bounded on three axes, because an agent that spends money needs a ceiling on all
of them: `max_iterations` (tool-call rounds), the CostMeter's spend limit, and the SQL tool's
own timeout. A runaway loop is the failure mode that costs real money.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from src.agent import citations as cite
from src.agent.prompts import (
    ANSWER_SYSTEM,
    CRITIC_SYSTEM,
    critic_user_prompt,
    revision_user_prompt,
)
from src.agent.router import HeuristicRouter, RoutingDecision
from src.agent.tools import RunSqlTool, SearchDocsTool, Tool
from src.guardrails.policy import GuardDecision, InputGuard, OutputGuard, ToolGuard
from src.interfaces import Agent
from src.llm.base import CostMeter, LLMProvider
from src.types import Answer, Citation, Trace


@dataclass
class AgentConfig:
    # Bounds ONE conversation. It does not bound the run: escalation restarts the loop and
    # the critic's revision pass runs it a third time, so the worst case is
    # max_iterations * 2 + 1 + max_iterations calls. Observed at 10 calls / $0.22 on a
    # question the corpus cannot answer, which is why max_llm_calls exists below.
    max_iterations: int = 6
    # Hard ceiling on model calls for the whole run, across every phase.
    max_llm_calls: int = 9
    max_tokens: int = 4096
    critic: bool = True
    escalate_on_ungrounded: bool = True
    spend_limit_usd: float = 1.00
    router: str = "heuristic"
    guardrails: bool = True
    version: str = "m3"


@dataclass
class AgentRun:
    """Everything one run did — the unit M4 scores and M5 mines."""

    answer: str = ""
    trace: Trace | None = None
    citation_report: cite.CitationReport | None = None
    routing: RoutingDecision | None = None
    critique: str = ""
    revised: bool = False
    iterations: int = 0
    tool_calls: list[str] = field(default_factory=list)
    cost: dict[str, Any] = field(default_factory=dict)
    guard_input: GuardDecision | None = None
    guard_output: GuardDecision | None = None
    guard_events: list[dict] = field(default_factory=list)
    guard_action: str = "allow"  # allow | redact | block, whichever the boundary applied
    # Set when a conversation hit `max_iterations` without the model finishing. Tracked
    # separately from "ungrounded" because they are different failures: an ungrounded answer
    # is a reasoning problem, exhaustion is a control problem. Escalating on exhaustion runs
    # the whole loop again on the dearer tier, so it needs to be visible in the trace rather
    # than showing up only as a doubled bill.
    exhausted: bool = False


class GroundedAgent(Agent):
    def __init__(
        self,
        provider: LLMProvider,
        search_tool: SearchDocsTool,
        sql_tool: RunSqlTool | None = None,
        config: AgentConfig | None = None,
        router: Any = None,
    ) -> None:
        self.provider = provider
        self.search = search_tool
        self.sql = sql_tool
        self.config = config or AgentConfig()
        self.router = router or HeuristicRouter()
        self.tools: dict[str, Tool] = {search_tool.name: search_tool}
        if sql_tool is not None:
            self.tools[sql_tool.name] = sql_tool
        self.input_guard = InputGuard()
        self.tool_guard = ToolGuard()
        self.output_guard = OutputGuard()
        self._run: AgentRun | None = None  # current run, for guard-event logging in dispatch

    def _schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self.tools.values()]

    def _dispatch(self, name: str, arguments: dict[str, Any]) -> str:
        tool = self.tools.get(name)
        if tool is None:
            return f"Error: no tool named {name!r}. Available: {sorted(self.tools)}."
        if self.config.guardrails:
            decision = self.tool_guard.check(name, arguments)
            if decision.blocked:
                if self._run is not None:
                    self._run.guard_events.append(
                        {"stage": "tool", "tool": name} | decision.to_dict()
                    )
                # Returned as a tool result the model can react to, not an exception — the
                # model should learn to stop reaching for blocked SQL.
                return f"Blocked by policy: {decision.reason}"
        try:
            return tool.run(**arguments).content
        except TypeError as exc:
            # Wrong arguments are the model's mistake to correct, not a crash.
            return f"Error: bad arguments for {name}: {exc}"

    def _converse(
        self, messages: list[dict[str, Any]], tier: str, meter: CostMeter, run: AgentRun
    ) -> str:
        """Run the tool loop until the model stops asking for tools. Returns final text."""
        for _ in range(self.config.max_iterations):
            if meter.calls >= self.config.max_llm_calls:
                run.exhausted = True
                return (
                    "I ran out of budget for this question before reaching an answer. "
                    "The documentation I reached did not settle it."
                )
            run.iterations += 1
            response = self.provider.generate(
                system=ANSWER_SYSTEM,
                messages=messages,
                tools=self._schemas(),
                tier=tier,
                max_tokens=self.config.max_tokens,
            )
            meter.record(response)

            if not response.wants_tools:
                return response.text

            assistant_content: list[dict[str, Any]] = []
            if response.text:
                assistant_content.append({"type": "text", "text": response.text})
            for call in response.tool_calls:
                assistant_content.append(
                    {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
                )
            messages.append({"role": "assistant", "content": assistant_content})

            # All tool results go back in ONE user message — splitting them teaches the
            # model to stop making parallel calls.
            results = []
            for call in response.tool_calls:
                run.tool_calls.append(call.name)
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": self._dispatch(call.name, call.arguments),
                    }
                )
            messages.append({"role": "user", "content": results})

        run.exhausted = True
        return (
            "I could not finish within the tool-call budget for this question. "
            "The documentation I reached did not settle it."
        )

    def run(self, query: str, tenant: str = "duckdb") -> Answer:
        result = self.run_detailed(query, tenant)
        cited = result.citation_report.cited_ids if result.citation_report else []
        return Answer(
            text=result.answer,
            citations=[Citation(chunk_id=c) for c in cited],
            trace=result.trace,
        )

    def run_detailed(self, query: str, tenant: str = "duckdb") -> AgentRun:
        run = AgentRun()
        self._run = run
        meter = CostMeter(spend_limit_usd=self.config.spend_limit_usd)
        t0 = time.perf_counter()

        # INPUT GUARD. Redaction happens here, before the query reaches the model OR the
        # trace, so a pasted credential never enters the context or the persisted record.
        agent_query = query
        if self.config.guardrails:
            run.guard_input = self.input_guard.check(query)
            agent_query = run.guard_input.text  # redacted; equals query when nothing matched
            if run.guard_input.blocked:
                return self._blocked_run(run, query, agent_query, tenant, t0)

        run.routing = self.router.route(agent_query)
        tier = run.routing.tier

        messages: list[dict[str, Any]] = [{"role": "user", "content": agent_query}]
        answer = self._converse(messages, tier, meter, run)

        report = cite.check(answer, self.search.retrieved_ids)

        # Escalation: a grounding failure is measured, not guessed. Only escalate upward.
        if (
            self.config.escalate_on_ungrounded
            and not report.grounded
            and tier != "strong"
        ):
            was_exhausted = run.exhausted
            run.exhausted = False
            cause = (
                "exhausted its tool budget"
                if was_exhausted
                else f"answered ungrounded ({len(report.invalid_ids)} invalid citations, "
                f"{len(report.cited_ids)} cited)"
            )
            run.routing = RoutingDecision(
                tier="strong",
                reason=f"escalated: {run.routing.reason} -> cheap tier {cause}",
                score=run.routing.score,
                escalated=True,
            )
            tier = "strong"
            messages = [{"role": "user", "content": agent_query}]
            answer = self._converse(messages, tier, meter, run)
            report = cite.check(answer, self.search.retrieved_ids)

        if self.config.critic and report.cited_ids and meter.calls < self.config.max_llm_calls:
            answer, report, run.critique, run.revised = self._critique(
                agent_query, answer, messages, tier, meter, run
            )

        # OUTPUT GUARD. Strips any secret the model echoed and catches a system-prompt leak,
        # before the answer is returned or persisted.
        guard_action = "allow"
        if run.guard_input and run.guard_input.redacted:
            guard_action = "redact"
        if self.config.guardrails:
            run.guard_output = self.output_guard.check(answer)
            if run.guard_output.action != "allow":
                answer = run.guard_output.text
                report = cite.check(answer, self.search.retrieved_ids)
                guard_action = run.guard_output.action
                run.guard_events.append({"stage": "output"} | run.guard_output.to_dict())

        run.answer = answer
        run.citation_report = report
        run.cost = meter.summary()
        run.trace = Trace(
            trace_id=uuid.uuid4().hex[:12],
            tenant=tenant,
            query=agent_query,  # redacted; the raw query is never persisted
            answer=answer,
            retrieved=sorted(self.search.retrieved_ids),
            citations=[Citation(chunk_id=c) for c in report.cited_ids],
            model_tier=tier,
            input_tokens=meter.input_tokens,
            output_tokens=meter.output_tokens,
            cost_usd=meter.total_usd,
            latency_ms=(time.perf_counter() - t0) * 1000,
            config_version=self.config.version,
            scores=report.to_dict()
            | {"escalated": float(run.routing.escalated), "exhausted": float(run.exhausted)},
            steps=[{"tool": t} for t in run.tool_calls],
        )
        run.guard_action = guard_action
        run.trace.scores["guard_action"] = 1.0 if guard_action != "allow" else 0.0
        return run

    def _blocked_run(
        self, run: AgentRun, raw_query: str, redacted: str, tenant: str, t0: float
    ) -> AgentRun:
        """A query blocked at the input guard never reaches the model. It still produces a
        trace — a blocked attempt is exactly what M5 wants to see, and the trace stores the
        redacted text so the record of the block cannot itself leak a secret."""
        decision = run.guard_input
        run.answer = (
            "I can't process that request. It looked like an attempt to override my "
            "instructions. Ask me a DuckDB question instead and I'm glad to help."
        )
        run.routing = RoutingDecision(tier="none", reason="blocked at input guard", score=0.0)
        run.citation_report = cite.check("", set())
        run.cost = {
            "calls": 0, "total_usd": 0.0, "by_tier": {}, "input_tokens": 0, "output_tokens": 0
        }
        run.guard_events.append({"stage": "input"} | (decision.to_dict() if decision else {}))
        run.trace = Trace(
            trace_id=uuid.uuid4().hex[:12],
            tenant=tenant,
            query=redacted,
            answer=run.answer,
            model_tier="none",
            latency_ms=(time.perf_counter() - t0) * 1000,
            config_version=self.config.version,
            scores={"grounded": 0.0, "citation_rate": 0.0, "blocked": 1.0},
        )
        run.guard_action = "block"
        return run

    def _critique(
        self,
        query: str,
        answer: str,
        messages: list[dict[str, Any]],
        tier: str,
        meter: CostMeter,
        run: AgentRun,
    ):
        """One critic pass. Revises only if the critic asks for it."""
        passages = "\n\n".join(
            f"[{cid}] {' '.join(chunk.text.split())[:400]}"
            for cid, chunk in self.search.seen.items()
        )
        verdict = self.provider.generate(
            system=CRITIC_SYSTEM,
            messages=[{"role": "user", "content": critic_user_prompt(query, answer, passages)}],
            tier=tier,
            max_tokens=1024,
        )
        meter.record(verdict)
        critique = verdict.text.strip()

        if not critique.upper().startswith("REVISE"):
            return answer, cite.check(answer, self.search.retrieved_ids), critique, False

        messages.append({"role": "assistant", "content": answer})
        messages.append({"role": "user", "content": revision_user_prompt(critique)})
        revised = self._converse(messages, tier, meter, run)
        return revised, cite.check(revised, self.search.retrieved_ids), critique, True
