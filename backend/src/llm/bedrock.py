"""Real Claude via AWS Bedrock.

Two client shapes exist and the SDK exposes both: `AnthropicBedrockMantle` (the Messages-API
Bedrock endpoint, the one to prefer) and the legacy `AnthropicBedrock` InvokeModel path.
Which one an account can actually reach varies, so this tries Mantle first and falls back —
and records which one served, because a cost or latency number means something different
depending on the path it took. On the account this project runs against, Mantle returns 403
and the legacy path serves; the probe is what establishes that rather than an assumption.

Thinking is **disabled by default** on both tiers. Sonnet 5 runs adaptive thinking when the
field is omitted, which would make the router's cost comparison measure two different things
(one tier thinking, one not). Disabling it on both is the controlled comparison. The tradeoff
is documented rather than hidden: with thinking off, Sonnet 5 reaches for tools less
readily, so the system prompt carries an explicit tool-use nudge (see src/agent/prompts.py).
"""

from __future__ import annotations

import os
import time
from typing import Any

from src.llm.base import LLMProvider, LLMResponse, ToolCall
from src.llm.pricing import spec_for

# The legacy InvokeModel path needs fully-qualified, region-prefixed inference-profile IDs,
# and the suffix convention is NOT uniform: Haiku 4.5 carries a dated `-v1:0` suffix while
# Sonnet 4.6 carries none. Both were read from `bedrock.list_inference_profiles()` rather
# than constructed — guessing `claude-sonnet-5-20260514-v1:0` returned "The provided model
# identifier is invalid", which is what a fabricated ID looks like.
LEGACY_IDS = {
    "anthropic.claude-haiku-4-5": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "anthropic.claude-sonnet-4-6": "us.anthropic.claude-sonnet-4-6",
}


class BedrockProvider(LLMProvider):
    def __init__(
        self,
        region: str | None = None,
        thinking: bool = False,
        spend_limit_usd: float = 1.00,
    ) -> None:
        self.region = region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        self.thinking = thinking
        self.spend_limit_usd = spend_limit_usd
        self._client = None
        self._path = ""

    @property
    def name(self) -> str:
        return f"bedrock({self._path or 'connecting'})"

    @property
    def path(self) -> str:
        """Which Bedrock path served — 'mantle' or 'legacy'. Empty until first call."""
        return self._path

    def _connect(self):
        if self._client is not None:
            return self._client
        from anthropic import AnthropicBedrock, AnthropicBedrockMantle

        errors = []
        for label, ctor in (("mantle", AnthropicBedrockMantle), ("legacy", AnthropicBedrock)):
            try:
                client = ctor(aws_region=self.region)
                # A tiny probe is cheaper than discovering the path is dead mid-eval.
                client.messages.create(
                    model=self._model_id("anthropic.claude-haiku-4-5", label),
                    max_tokens=4,
                    messages=[{"role": "user", "content": "hi"}],
                )
                self._client, self._path = client, label
                return client
            except Exception as exc:  # noqa: BLE001 - probing both paths on purpose
                errors.append(f"{label}: {type(exc).__name__}: {str(exc)[:120]}")
        raise RuntimeError(
            "no working Bedrock path. Refresh credentials (`aws login`) and retry.\n  "
            + "\n  ".join(errors)
        )

    def _model_id(self, model_id: str, path: str | None = None) -> str:
        path = path or self._path
        return LEGACY_IDS.get(model_id, model_id) if path == "legacy" else model_id

    def generate(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tier: str = "cheap",
        max_tokens: int = 4096,
    ) -> LLMResponse:
        client = self._connect()
        spec = spec_for(tier)

        kwargs: dict[str, Any] = {
            "model": self._model_id(spec.model_id),
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
        # Explicit either way: omitting `thinking` means adaptive on Sonnet 5 but off on
        # Haiku 4.5, which would silently make the two tiers non-comparable.
        kwargs["thinking"] = {"type": "adaptive"} if self.thinking else {"type": "disabled"}

        t0 = time.perf_counter()
        response = client.messages.create(**kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000

        text_parts, tool_calls = [], []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                )

        return LLMResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=response.stop_reason or "",
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost_usd=spec.cost(response.usage.input_tokens, response.usage.output_tokens),
            tier=tier,
            model_id=spec.model_id,
            latency_ms=latency_ms,
            raw=response,
        )
