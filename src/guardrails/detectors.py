"""Deterministic detectors: secrets, PII, prompt injection.

No external service, no model call — regex and structural checks only, so a guardrail never
costs a token and never adds latency worth measuring. Adapted from the `llm-guardrails` repo
(macro-F1 1.0 on its labeled set), narrowed to what this domain actually sees.

The domain shapes what matters. A DuckDB support question routinely carries a real credential
("here is my `CREATE SECRET` with my AWS keys, why does it fail") — and the trace it produces
is persisted to SQLite and mined by the M5 flywheel, so an un-redacted secret would end up in
training data. Secret redaction on the input path is therefore the load-bearing guardrail
here, not injection defense.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# --- secrets: high-confidence, low false-positive credential shapes -------------------

_SECRETS = {
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "aws_secret_key": re.compile(r"\b(?<![A-Za-z0-9/+])[A-Za-z0-9/+]{40}(?![A-Za-z0-9/+])\b"),
    "anthropic_key": re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b"),
    "openai_key": re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    "github_token": re.compile(r"\bghp_[A-Za-z0-9]{36}\b"),
    "slack_token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b"),
}

# --- PII ------------------------------------------------------------------------------

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CREDIT_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
# IPv4 is deliberately NOT redacted: a DuckDB support question is full of them (host
# addresses, S3 endpoints, `SET http_proxy`), and redacting them would mangle legitimate
# technical content. Redact what is a secret, not what merely looks like a number.


@dataclass
class Span:
    kind: str
    start: int
    end: int
    text: str


def _luhn(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    checksum, parity = 0, len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        checksum += d
    return checksum % 10 == 0


def detect_secrets(text: str) -> list[Span]:
    spans: list[Span] = []
    for kind, rx in _SECRETS.items():
        for m in rx.finditer(text):
            spans.append(Span(kind, m.start(), m.end(), m.group()))
    return spans


def detect_pii(text: str) -> list[Span]:
    spans: list[Span] = []
    for kind, rx in (("email", _EMAIL), ("ssn", _SSN)):
        for m in rx.finditer(text):
            spans.append(Span(kind, m.start(), m.end(), m.group()))
    for m in _CREDIT_CARD.finditer(text):
        if _luhn(m.group()):  # Luhn kills most false positives on stray digit runs
            spans.append(Span("credit_card", m.start(), m.end(), m.group()))
    return spans


def redact(text: str, spans: list[Span]) -> str:
    """Replace each span with a typed placeholder, right-to-left so offsets stay valid."""
    out = text
    # De-overlap first: two detectors can hit the same region (an AWS secret key is 40
    # base64 chars, which a card regex might graze). Keep the longest.
    ordered = sorted(spans, key=lambda s: (s.start, -(s.end - s.start)))
    kept: list[Span] = []
    for s in ordered:
        if not any(s.start < k.end and s.end > k.start for k in kept):
            kept.append(s)
    for s in sorted(kept, key=lambda s: s.start, reverse=True):
        out = out[: s.start] + f"[{s.kind.upper()}]" + out[s.end :]
    return out


# --- prompt injection -----------------------------------------------------------------
#
# (weight, pattern, label). Returns a score + the matched signals so a block is explainable.
# Tuned conservatively: this corpus is technical, and an over-eager detector will flag
# legitimate SQL. `SET`, `PRAGMA` and SQL comments are NOT injection signals here.

_INJECTION = [
    (0.8, re.compile(r"ignore\s+(all|any|the|your)?\s*(previous|prior|above|earlier)\s+"
                     r"(instructions?|prompts?|rules)", re.I), "instruction_override"),
    (0.8, re.compile(r"disregard\s+(all|the|your)?\s*(previous|prior|above)?\s*"
                     r"(instructions?|rules|guidelines)", re.I), "instruction_override"),
    (0.7, re.compile(r"forget\s+(everything|all|what).{0,20}(said|told|instructed)", re.I),
     "instruction_override"),
    (0.8, re.compile(r"\b(you are now|act as|pretend to be|roleplay as)\b.{0,40}\b"
                     r"(dan|developer mode|jailbroken|unrestricted|no restrictions)\b", re.I),
     "jailbreak_persona"),
    (0.7, re.compile(r"(reveal|print|repeat|show|output).{0,25}"
                     r"(system prompt|your instructions|initial prompt|the prompt above)", re.I),
     "prompt_exfiltration"),
    (0.7, re.compile(r"(ignore|bypass|override|disable).{0,20}"
                     r"(safety|guardrails?|content policy|filters?|restrictions?)", re.I),
     "safety_bypass"),
]


def injection_score(text: str) -> tuple[float, list[str]]:
    matched: list[str] = []
    total = 0.0
    for weight, rx, label in _INJECTION:
        if rx.search(text):
            matched.append(label)
            total = max(total, weight) + 0.1 * (len(set(matched)) - 1)
    return round(min(total, 1.0), 3), sorted(set(matched))
