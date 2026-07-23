"""Provider interface and message types for the agent loop.

One interface, three implementations: a deterministic fake (offline tests), Bedrock (real),
and later a local GGUF tier. The agent never learns which it is talking to — that is what
makes the whole M2 loop testable without spending a cent, and what lets M5 swap a
fine-tuned router in without touching the agent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.llm.pricing import ModelSpec, cost_at_list_price


@dataclass
class ToolCall:
    """A tool the model asked to run."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """One model turn, with the accounting attached.

    `usage` numbers come from the API response, never estimated — see src/llm/pricing.py.
    """

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    tier: str = ""
    model_id: str = ""
    latency_ms: float = 0.0
    raw: Any = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)

    @property
    def list_price_cost_usd(self) -> float:
        """Cost without the Sonnet introductory discount, for restating a claim at list."""
        return cost_at_list_price(self.tier, self.input_tokens, self.output_tokens)


class LLMProvider(ABC):
    """Generate one turn, optionally with tools available."""

    @abstractmethod
    def generate(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tier: str = "cheap",
        max_tokens: int = 4096,
    ) -> LLMResponse: ...

    @property
    def name(self) -> str:
        return type(self).__name__


class CostMeter:
    """Running spend, so a run can be capped before it surprises anyone.

    Every project in this portfolio that spends money learns the same lesson late; this one
    starts with the ceiling. `spend_limit_usd` raises rather than silently continuing,
    because a runaway agent loop is the failure mode that actually costs real money.
    """

    def __init__(self, spend_limit_usd: float = 1.00) -> None:
        self.spend_limit_usd = spend_limit_usd
        self.total_usd = 0.0
        self.calls = 0
        self.by_tier: dict[str, float] = {}
        self.input_tokens = 0
        self.output_tokens = 0

    def record(self, response: LLMResponse) -> None:
        self.total_usd += response.cost_usd
        self.calls += 1
        self.by_tier[response.tier] = self.by_tier.get(response.tier, 0.0) + response.cost_usd
        self.input_tokens += response.input_tokens
        self.output_tokens += response.output_tokens
        if self.total_usd > self.spend_limit_usd:
            raise SpendLimitExceeded(
                f"spend ${self.total_usd:.4f} exceeded limit ${self.spend_limit_usd:.2f} "
                f"after {self.calls} calls"
            )

    def summary(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "total_usd": round(self.total_usd, 6),
            "by_tier": {k: round(v, 6) for k, v in self.by_tier.items()},
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


class SpendLimitExceeded(RuntimeError):
    pass


def build_cost(spec: ModelSpec, input_tokens: int, output_tokens: int) -> float:
    return spec.cost(input_tokens, output_tokens)
