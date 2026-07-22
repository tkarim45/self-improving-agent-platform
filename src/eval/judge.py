"""LLM-judge — Claude scoring an answer against the passages it was given (G-Eval, CoT).

Runs on a *sampled* fraction of traffic (the deterministic scorers cover 100%), because it
costs a model call. The judge scores three axes on a 1–5 scale with chain-of-thought, against
the retrieved passages only:
  - faithfulness : does every claim follow from the passages? (the anti-hallucination axis)
  - relevance    : does the answer actually address the question?
  - completeness : does it cover what the passages support, or leave the user hanging?

The judge is deliberately given ONLY the passages, not outside knowledge, so it grades
grounding rather than its own opinion of DuckDB. It returns structured scores plus its
reasoning, so a verdict is auditable and — critically for M5 — so its faithfulness score can
be checked against the *execution* oracle. A judge that calls a wrong-answer faithful is a
judge M5's reward-hacking would exploit; M4 measures that agreement rather than assuming it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.llm.base import LLMProvider

JUDGE_SYSTEM = """You are a strict evaluator of a DuckDB support answer. You grade ONLY \
against the reference passages provided — never against your own knowledge of DuckDB, which \
may differ from this documentation version.

Score three axes, each an integer 1 to 5:

- faithfulness: Does every factual claim in the answer follow from the passages? 5 = every \
claim is supported; 1 = the answer asserts things the passages do not say, or contradicts \
them. This is the most important axis.
- relevance: Does the answer address the user's actual question? 5 = directly answers it; \
1 = answers a different question or rambles.
- completeness: Does the answer cover what the passages support for this question? 5 = \
thorough; 1 = omits the key point the passages contain.

If the answer correctly says the documentation does not cover the question, and the passages \
indeed do not, score faithfulness 5, relevance 5, completeness 5 — a correct abstention is a \
good answer.

Think step by step, then end with a single JSON line and nothing after it:
{"faithfulness": N, "relevance": N, "completeness": N, "verdict": "pass"|"fail"}

verdict is "pass" only if faithfulness >= 4 and relevance >= 4."""

_JSON = re.compile(r"\{[^{}]*\"faithfulness\"[^{}]*\}")


@dataclass
class Judgment:
    faithfulness: int
    relevance: int
    completeness: int
    verdict: str
    reasoning: str = ""
    cost_usd: float = 0.0

    @property
    def passed(self) -> bool:
        return self.verdict == "pass"

    def to_dict(self) -> dict:
        return {
            "faithfulness": self.faithfulness,
            "relevance": self.relevance,
            "completeness": self.completeness,
            "verdict": self.verdict,
            "cost_usd": round(self.cost_usd, 6),
        }


def _user_prompt(question: str, answer: str, passages: str) -> str:
    return (
        f"# Question\n{question}\n\n"
        f"# Reference passages (the ONLY source of truth)\n{passages}\n\n"
        f"# Answer to grade\n{answer}\n\n"
        "Grade the answer against the passages. Reason step by step, then output the JSON line."
    )


class LLMJudge:
    def __init__(self, provider: LLMProvider, tier: str = "strong") -> None:
        # Judge on the strong tier by default: a weak judge is a weak sensor, and the judge
        # runs on a sample so the cost is bounded.
        self.provider = provider
        self.tier = tier

    def judge(self, question: str, answer: str, passages: str) -> Judgment:
        response = self.provider.generate(
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": _user_prompt(question, answer, passages)}],
            tier=self.tier,
            max_tokens=1024,
        )
        return parse_judgment(response.text, cost_usd=response.cost_usd)


def parse_judgment(text: str, cost_usd: float = 0.0) -> Judgment:
    """Pull the trailing JSON verdict out of the judge's chain-of-thought.

    A judge that returns malformed JSON is a fail-safe, not a crash: default to the harshest
    verdict so a parsing failure can never silently pass a bad answer through the gate.
    """
    matches = _JSON.findall(text)
    if not matches:
        return Judgment(1, 1, 1, "fail", reasoning=text, cost_usd=cost_usd)
    try:
        data = json.loads(matches[-1])
    except json.JSONDecodeError:
        return Judgment(1, 1, 1, "fail", reasoning=text, cost_usd=cost_usd)

    def clamp(v) -> int:
        try:
            return max(1, min(5, int(v)))
        except (TypeError, ValueError):
            return 1

    faith = clamp(data.get("faithfulness"))
    rel = clamp(data.get("relevance"))
    comp = clamp(data.get("completeness"))
    # Recompute the verdict from the scores rather than trusting the model's own "verdict"
    # field — the rubric defines pass, not the model's mood.
    verdict = "pass" if faith >= 4 and rel >= 4 else "fail"
    return Judgment(faith, rel, comp, verdict, reasoning=text, cost_usd=cost_usd)
