"""A deterministic fake model.

This is not a stub that returns a fixed string. It is scriptable: tests hand it a sequence of
turns (text, tool calls, or a callable that inspects the conversation) and it replays them,
recording every request it saw. That makes the entire agent loop — planning, tool dispatch,
citation checking, the critic pass, router accounting — testable offline with no credentials
and no spend.

It reports plausible token counts so the cost plumbing is exercised too. Those numbers are
**fabricated by construction** and must never appear in a report; a test pins that fact.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.llm.base import LLMProvider, LLMResponse, ToolCall
from src.llm.pricing import spec_for

Turn = str | LLMResponse | Callable[[list[dict[str, Any]]], "str | LLMResponse"]


class FakeProvider(LLMProvider):
    """Replays scripted turns. Raises if the script runs dry — a silent default would let a
    test pass while the agent looped more times than the author expected."""

    def __init__(self, turns: list[Turn] | None = None, tokens_per_call: int = 100) -> None:
        self.turns: list[Turn] = list(turns or [])
        self.tokens_per_call = tokens_per_call
        self.requests: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "fake"

    @property
    def call_count(self) -> int:
        return len(self.requests)

    def tiers_used(self) -> list[str]:
        return [r["tier"] for r in self.requests]

    def generate(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tier: str = "cheap",
        max_tokens: int = 4096,
    ) -> LLMResponse:
        self.requests.append(
            {
                "system": system,
                "messages": messages,
                "tools": [t["name"] for t in (tools or [])],
                "tier": tier,
                "max_tokens": max_tokens,
            }
        )
        if not self.turns:
            raise AssertionError(
                f"FakeProvider script exhausted after {len(self.requests)} calls — "
                "the agent asked for more turns than the test scripted"
            )

        turn = self.turns.pop(0)
        if callable(turn):
            turn = turn(messages)
        if isinstance(turn, LLMResponse):
            response = turn
        else:
            response = LLMResponse(text=turn, stop_reason="end_turn")

        spec = spec_for(tier)
        response.tier = tier
        response.model_id = f"fake:{spec.model_id}"
        if not response.input_tokens:
            response.input_tokens = self.tokens_per_call
        if not response.output_tokens:
            response.output_tokens = self.tokens_per_call // 4
        response.cost_usd = spec.cost(response.input_tokens, response.output_tokens)
        return response


def tool_turn(name: str, arguments: dict[str, Any], call_id: str = "t1") -> LLMResponse:
    """Convenience for scripting a tool-calling turn."""
    return LLMResponse(
        text="",
        tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
        stop_reason="tool_use",
    )
