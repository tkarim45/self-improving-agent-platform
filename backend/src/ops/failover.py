"""Ordered provider failover.

Wraps an ordered list of `LLMProvider`s: try the primary, and on a provider-level failure
fall through to the next. Returns the response from the first that succeeds, tagged with which
one served — because a cost or latency number means something different depending on the
provider behind it.

What counts as a failure worth failing over: connection errors, 5xx, throttling, and
"model not available" (403) — the transient or entitlement problems a second provider might
not have. A `SpendLimitExceeded` is NOT failed over: it is the caller's own budget decision,
and retrying on another provider would just spend more. That distinction is the whole point of
failover being selective rather than a blanket try/except.

M3 verifies this with a dead-primary test (src/ops, tests): a provider that always raises,
followed by a working one, must produce a working answer and record that the primary was
skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.llm.base import LLMProvider, LLMResponse, SpendLimitExceeded


@dataclass
class FailoverEvent:
    provider: str
    error: str


class AllProvidersFailed(RuntimeError):
    def __init__(self, events: list[FailoverEvent]) -> None:
        self.events = events
        super().__init__(
            "every provider failed: " + "; ".join(f"{e.provider}: {e.error}" for e in events)
        )


# Error names that mean "this provider can't serve right now, try the next one". Matched by
# class name so the failover layer needs no import from anthropic/botocore.
_FAILOVER_ERRORS = {
    "APIConnectionError",
    "APIConnectionTimeoutError",
    "InternalServerError",
    "ServiceUnavailableError",
    "RateLimitError",
    "PermissionDeniedError",  # Bedrock 403 "model not available" — a sibling may have it
    "ClientError",
    "EndpointConnectionError",
    "ConnectionError",
    "TimeoutError",
    "RuntimeError",  # BedrockProvider raises this when no path connects
}


def _should_failover(exc: Exception) -> bool:
    return type(exc).__name__ in _FAILOVER_ERRORS


class FailoverProvider(LLMProvider):
    def __init__(self, providers: list[LLMProvider], labels: list[str] | None = None) -> None:
        if not providers:
            raise ValueError("failover needs at least one provider")
        self.providers = providers
        self.labels = labels or [p.name for p in providers]
        self.events: list[FailoverEvent] = []

    @property
    def name(self) -> str:
        return "failover(" + " -> ".join(self.labels) + ")"

    def generate(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tier: str = "cheap",
        max_tokens: int = 4096,
    ) -> LLMResponse:
        failures: list[FailoverEvent] = []
        for provider, label in zip(self.providers, self.labels, strict=True):
            try:
                response = provider.generate(system, messages, tools, tier, max_tokens)
                if failures:
                    # Record on the response so a trace can show it was not the primary.
                    response.raw = {"served_by": label, "skipped": [f.provider for f in failures]}
                return response
            except SpendLimitExceeded:
                # The caller's budget, not a provider fault — do not spend more elsewhere.
                raise
            except Exception as exc:  # noqa: BLE001 - deciding failover vs re-raise by policy
                if not _should_failover(exc):
                    raise
                event = FailoverEvent(
                    provider=label, error=f"{type(exc).__name__}: {str(exc)[:80]}"
                )
                failures.append(event)
                self.events.append(event)
        raise AllProvidersFailed(failures)
