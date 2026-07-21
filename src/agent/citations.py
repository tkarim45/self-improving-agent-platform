"""Citation extraction and grounding checks.

The agent is instructed to cite every claim as `[chunk_id]`. This module is the cheap
post-check that verifies it actually did — run on every answer, before the answer is
returned, so a grounding failure is caught by the system rather than by a reader.

Two failures are worth separating, because they mean different things:

- **Invalid citation** — the answer cites a chunk id that was never retrieved. The model
  invented a source. This is the serious one: it looks grounded and isn't.
- **Uncited claim** — a sentence asserts something with no citation at all. Less alarming
  (it may be a transition or a hedge) but it is where ungrounded content hides.

Sentence segmentation and claim detection are heuristics, not semantics. The counts here
are a **cheap online signal**, deliberately not a substitute for the M4 LLM-judge. Stated
plainly because a heuristic reported as a quality metric is how eval harnesses start lying.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Chunk ids are 12 hex chars (src.types.stable_id).
CITATION = re.compile(r"\[([0-9a-f]{6,32})\]")

# Split on sentence-ending punctuation followed by whitespace + a capital/digit/backtick.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`\[])")

# Models write the citation AFTER the closing period ("...already been applied. [403bd...]"),
# which is what the prompt asks for. Segmenting naively strands the citation as its own
# fragment and marks the sentence it supports as uncited — that bug reported a 0% citation
# rate on a real answer carrying three valid citations. Under-reporting is as harmful as
# over-reporting: M5 would mine well-grounded answers as failures.
#
# Rather than refuse to split there (which merges two sentences whenever a citation sits
# between them), normalize the citation to just *before* the period. Segmentation then works
# normally and the citation stays attached to the claim it supports.
_TRAILING_CITATION = re.compile(r"([.!?])(\s*)(\[[0-9a-f]{6,32}\])")


def _attach_trailing_citations(text: str) -> str:
    return _TRAILING_CITATION.sub(r" \3\1", text)

# Lines that carry no assertion of their own and so need no citation.
_NON_CLAIM = re.compile(
    r"^\s*(?:#{1,6}\s|[-*+]\s*$|\d+\.\s*$|```|\||>\s*$)",
)


@dataclass
class CitationReport:
    cited_ids: list[str] = field(default_factory=list)
    invalid_ids: list[str] = field(default_factory=list)
    uncited_claims: list[str] = field(default_factory=list)
    n_claims: int = 0

    @property
    def has_invalid(self) -> bool:
        return bool(self.invalid_ids)

    @property
    def grounded(self) -> bool:
        """No invented sources and at least one real one."""
        return not self.invalid_ids and bool(self.cited_ids)

    @property
    def citation_rate(self) -> float:
        """Fraction of claim sentences carrying at least one citation."""
        if not self.n_claims:
            return 0.0
        return (self.n_claims - len(self.uncited_claims)) / self.n_claims

    def to_dict(self) -> dict:
        return {
            "cited_ids": self.cited_ids,
            "invalid_ids": self.invalid_ids,
            "n_claims": self.n_claims,
            "n_uncited": len(self.uncited_claims),
            "citation_rate": round(self.citation_rate, 4),
            "grounded": self.grounded,
        }


def extract_citations(text: str) -> list[str]:
    """Cited chunk ids in order of first appearance."""
    seen: list[str] = []
    for match in CITATION.finditer(text):
        cid = match.group(1)
        if cid not in seen:
            seen.append(cid)
    return seen


def strip_citations(text: str) -> str:
    """The answer as a reader would hear it, for prose-level checks."""
    return re.sub(r"\s*\[[0-9a-f]{6,32}\]", "", text)


def split_sentences(text: str) -> list[str]:
    out: list[str] = []
    # Fenced code blocks are not prose — drop them before segmenting.
    prose = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    prose = _attach_trailing_citations(prose)
    for line in prose.splitlines():
        if _NON_CLAIM.match(line) or not line.strip():
            continue
        out.extend(s.strip() for s in _SENTENCE_SPLIT.split(line.strip()) if s.strip())
    return out


def is_claim(sentence: str) -> bool:
    """Whether a sentence asserts something that ought to be cited.

    Excludes questions and very short fragments ("Yes.", "In short:") — those carry no
    standalone assertion, and demanding a citation on them would inflate the uncited count
    with noise.
    """
    stripped = sentence.strip()
    if len(stripped) < 25 or stripped.endswith("?"):
        return False
    return len(strip_citations(stripped).split()) >= 5


def check(text: str, retrieved_ids: list[str] | set[str]) -> CitationReport:
    """Verify an answer's citations against the chunks that were actually retrieved."""
    available = set(retrieved_ids)
    cited = extract_citations(text)

    report = CitationReport(
        cited_ids=cited,
        invalid_ids=[c for c in cited if c not in available],
    )
    for sentence in split_sentences(text):
        if not is_claim(sentence):
            continue
        report.n_claims += 1
        if not CITATION.search(sentence):
            report.uncited_claims.append(sentence)
    return report
