"""Cost-aware model routing.

Two mechanisms, deliberately separated:

1. **Pre-routing** — a transparent heuristic guesses the tier from the query alone, before
   any tokens are spent. Cheap and explainable, but it is guessing.
2. **Escalation** — if the cheap tier's answer fails the grounding check (invented a
   citation, or cited nothing), the query is retried on the strong tier. This is not a
   guess: it is a measured failure, caught by src/agent/citations.py.

The escalation path is the interesting half. A pre-router can only be as good as its
correlation with difficulty; an escalation policy observes the actual failure. `llm-router`
found that a structural heuristic beat a learned router on mock data and *lost* to it on
real Bedbrock traffic, because the learned one adapted to where the small model actually
failed. Escalation is that adaptation without the training step — and the (query, tier that
worked) pairs it produces are exactly the supervision M5 distills into a learned router.

Every decision carries a `reason` string so a routing choice can be audited rather than
trusted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Signals that a question needs more than a single lookup.
_MULTI_PART = re.compile(
    r"\b(?:and then|as well as|both .+ and|after that|followed by|in addition)\b", re.IGNORECASE
)
_REASONING = re.compile(
    r"\b(?:why|compare|difference between|trade[- ]?off|when should|which is better|"
    r"explain how|under the hood|performance of|faster|slower|instead of)\b",
    re.IGNORECASE,
)
_LOOKUP = re.compile(
    r"^\s*(?:what(?:'s| is) (?:the )?(?:syntax|default)|how do i|what does|list |show me)\b",
    re.IGNORECASE,
)

CHEAP, STRONG = "cheap", "strong"


@dataclass
class RoutingDecision:
    tier: str
    reason: str
    score: float
    escalated: bool = False

    def to_dict(self) -> dict:
        return {
            "tier": self.tier,
            "reason": self.reason,
            "score": round(self.score, 3),
            "escalated": self.escalated,
        }


class HeuristicRouter:
    """Score a query for difficulty; route above the threshold to the strong tier.

    The weights are hand-set, not fitted — with no labeled difficulty data yet, a fitted
    threshold would be overfitting to nothing. M2 measures whether this heuristic actually
    correlates with which tier answers correctly; M5 replaces it with a classifier trained
    on the escalation record.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def score(self, query: str) -> tuple[float, list[str]]:
        score, reasons = 0.0, []
        words = len(query.split())

        if _REASONING.search(query):
            score += 0.5
            reasons.append("reasoning/comparison language")
        if _MULTI_PART.search(query):
            score += 0.4
            reasons.append("multi-part question")
        if words > 25:
            score += 0.3
            reasons.append(f"long query ({words} words)")
        elif words > 15:
            score += 0.15
            reasons.append(f"medium query ({words} words)")
        if query.count("?") > 1:
            score += 0.3
            reasons.append("several questions")
        if _LOOKUP.match(query) and words <= 15:
            score -= 0.3
            reasons.append("short factual lookup")

        return max(0.0, score), reasons

    def route(self, query: str) -> RoutingDecision:
        score, reasons = self.score(query)
        tier = STRONG if score >= self.threshold else CHEAP
        why = "; ".join(reasons) if reasons else "no difficulty signals"
        return RoutingDecision(tier=tier, reason=f"score {score:.2f} ({why})", score=score)


class AlwaysRouter:
    """Fixed-tier control arm. Without always-cheap and always-strong baselines, a router's
    saving is unfalsifiable — it needs both a floor and a ceiling to be measured against."""

    def __init__(self, tier: str = CHEAP) -> None:
        self.tier = tier

    def route(self, query: str) -> RoutingDecision:
        return RoutingDecision(tier=self.tier, reason=f"always-{self.tier} baseline", score=0.0)


def get_router(name: str, threshold: float = 0.5):
    if name == "heuristic":
        return HeuristicRouter(threshold=threshold)
    if name in (CHEAP, STRONG):
        return AlwaysRouter(name)
    if name == "active":
        # Whatever the flywheel last promoted (configs/active.json); falls back to the
        # heuristic when nothing has been promoted. This is the hook that makes a promotion
        # change production behaviour rather than just writing a log line.
        from src.flywheel.promote import PromotionLog, active_router

        return active_router(PromotionLog())
    if name.startswith("learned:"):
        from pathlib import Path

        from src.flywheel.router_train import LearnedRouter

        return LearnedRouter.load(Path(name.split(":", 1)[1]))
    raise ValueError(
        f"unknown router {name!r}; expected 'heuristic', 'cheap', 'strong', 'active', "
        "or 'learned:<path>'"
    )
