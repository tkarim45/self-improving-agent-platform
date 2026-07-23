"""Model tiers and per-token pricing.

Cost is computed from the token counts the API actually reports, not estimated from
character counts. Everything downstream — the router's saving claim in M2, the cost line on
the M6 improvement curve — is only as honest as this table, so it is pinned with a date and
a source rather than hardcoded inline.

Prices are USD per million tokens, captured 2026-07-21. They change; re-check before
quoting a cost figure in a report.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """A routable model tier."""

    tier: str
    model_id: str  # Bedrock ID (carries the `anthropic.` prefix)
    input_per_mtok: float
    output_per_mtok: float
    note: str = ""

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok
        ) / 1_000_000


# The two arms of the cost-aware router (M2 step 4).
#
# The strong tier is Sonnet 4.6, NOT Sonnet 5. Sonnet 5 was the intended choice, but this
# Bedrock account returns 403 "not available for this account" for it — as it does for Opus
# 4.8 and Opus 4.5. Probed on 2026-07-21; enabled here are Haiku 4.5, Sonnet 4.6 and Sonnet
# 4.5. Sonnet 4.6 is the nearest available tier, one generation back.
#
# A consequence worth stating: Sonnet 4.6 has no introductory discount, so the price gap is a
# clean 3x on both input and output rather than the 2x Sonnet 5's promotion would have given.
# That makes the router's cost story easier to read, not harder.
CHEAP = ModelSpec(
    tier="cheap",
    model_id="anthropic.claude-haiku-4-5",
    input_per_mtok=1.00,
    output_per_mtok=5.00,
)

STRONG = ModelSpec(
    tier="strong",
    model_id="anthropic.claude-sonnet-4-6",
    input_per_mtok=3.00,
    output_per_mtok=15.00,
    note="substituted for Sonnet 5, which is not enabled on this Bedrock account",
)

# Same as live pricing here (no promotion in play). Kept so a future tier on introductory
# pricing can still have its claim restated at list.
LIST_PRICING = {
    "cheap": (1.00, 5.00),
    "strong": (3.00, 15.00),
}

TIERS: dict[str, ModelSpec] = {"cheap": CHEAP, "strong": STRONG}

PRICING_AS_OF = "2026-07-21"


def spec_for(tier: str) -> ModelSpec:
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; expected one of {sorted(TIERS)}")
    return TIERS[tier]


def cost_at_list_price(tier: str, input_tokens: int, output_tokens: int) -> float:
    """What the same call would cost without the introductory discount."""
    in_rate, out_rate = LIST_PRICING[tier]
    return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000
