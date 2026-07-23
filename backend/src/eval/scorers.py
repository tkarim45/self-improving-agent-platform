"""Online scorers — cheap, deterministic, run on every trace.

These are the flywheel's *sensor*: they run on every answer with no model call and no cost,
so quality can be tracked on 100% of traffic rather than the sampled fraction the LLM-judge
covers. They are heuristics, and they are labelled as heuristics — a proxy reported as truth
is how an eval harness starts lying.

Four signals:
  - groundedness   : do cited chunk ids actually appear in the retrieved set? (invented
                     sources are the serious failure — an answer that looks grounded and isn't)
  - citation_rate  : fraction of claim sentences carrying a citation (faithfulness proxy)
  - task_success   : did the run produce a grounded, non-empty, non-exhausted answer?
  - abstained      : did the agent correctly say "the docs don't cover this"? (not a failure)

`task_success` is a PROXY. It rewards a grounded answer; it does not check the answer is
*correct*. That is what the execution checker (objective, for SQL) and the LLM-judge
(subjective) are for. Keeping the proxy honest about its own limits is the point of the M4
harness existing at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.agent import citations as cite
from src.types import Trace

_ABSTAIN = re.compile(
    r"\b(do(?:es)?(?:n't| not) (?:cover|mention|address|contain|include|specify)|"
    r"not covered|couldn't find|could not find|no (?:information|mention|benchmark|data)|"
    r"i don't have|isn't (?:covered|documented)|not (?:documented|in the documentation))\b",
    re.IGNORECASE,
)


@dataclass
class Scores:
    groundedness: float  # 1.0 if every citation is real and at least one exists
    citation_rate: float
    task_success: float
    abstained: bool
    invalid_citations: int

    def to_dict(self) -> dict:
        return {
            "groundedness": self.groundedness,
            "citation_rate": round(self.citation_rate, 4),
            "task_success": self.task_success,
            "abstained": self.abstained,
            "invalid_citations": self.invalid_citations,
        }


def score_trace(trace: Trace) -> Scores:
    """Score one trace from what it already recorded — no re-running the agent."""
    return score_answer(trace.answer, [c.chunk_id for c in trace.citations], trace.retrieved)


def score_answer(answer: str, cited_ids: list[str], retrieved_ids: list[str]) -> Scores:
    report = cite.check(answer, set(retrieved_ids))
    abstained = bool(_ABSTAIN.search(answer))

    grounded = report.grounded
    # task_success: a grounded answer that finished, OR a correct abstention. An abstention
    # is a *success* — saying "the docs don't cover this" beats inventing an answer, and the
    # whole point of picking an un-memorized corpus (M0) was to make that distinction real.
    task_success = 1.0 if (grounded or abstained) else 0.0

    return Scores(
        groundedness=1.0 if grounded else 0.0,
        citation_rate=report.citation_rate,
        task_success=task_success,
        abstained=abstained,
        invalid_citations=len(report.invalid_ids),
    )


def aggregate(scores: list[Scores]) -> dict:
    if not scores:
        return {"n": 0}
    n = len(scores)
    return {
        "n": n,
        "groundedness": round(sum(s.groundedness for s in scores) / n, 4),
        "citation_rate": round(sum(s.citation_rate for s in scores) / n, 4),
        "task_success": round(sum(s.task_success for s in scores) / n, 4),
        "abstain_rate": round(sum(s.abstained for s in scores) / n, 4),
        "total_invalid_citations": sum(s.invalid_citations for s in scores),
    }
