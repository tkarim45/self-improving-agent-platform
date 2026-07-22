"""The guardrail boundary: input, tool-call, and output gates.

Three gates, each returning a `GuardDecision` that says allow / redact / block plus the
signals that drove it, so every boundary action is auditable in the trace rather than opaque.

The design choice that matters: **redaction happens before anything is stored or sent.** The
query is redacted before it reaches the model and before it is written to a trace, so a
credential a user pastes never lands in the model context, the SQLite trace, or the M5
training data drawn from it. Blocking is reserved for injection — you cannot redact an
instruction-override, you can only refuse it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.guardrails import detectors as d

# Same statements the run_sql sandbox refuses, screened one layer earlier so a block is
# attributed to policy and logged, not swallowed as a runtime exception.
_UNSAFE_SQL = re.compile(
    r"\b(?:ATTACH|DETACH|INSTALL|LOAD|COPY|EXPORT|IMPORT|SET\s+enable_external_access)\b",
    re.IGNORECASE,
)


@dataclass
class GuardDecision:
    action: str  # "allow" | "redact" | "block"
    text: str  # the (possibly redacted) text to use downstream
    signals: list[str] = field(default_factory=list)
    reason: str = ""

    @property
    def blocked(self) -> bool:
        return self.action == "block"

    @property
    def redacted(self) -> bool:
        return self.action == "redact"

    def to_dict(self) -> dict:
        return {"action": self.action, "signals": self.signals, "reason": self.reason}


class InputGuard:
    """Screens a user query before it enters the agent.

    Order matters: redact secrets/PII first (so they never reach the model or the trace),
    then evaluate injection on the *redacted* text. A blocked query is still redacted, so the
    logged attempt cannot itself leak a credential.
    """

    def __init__(self, injection_threshold: float = 0.7) -> None:
        self.injection_threshold = injection_threshold

    def check(self, query: str) -> GuardDecision:
        spans = d.detect_secrets(query) + d.detect_pii(query)
        text = d.redact(query, spans) if spans else query
        redaction_signals = sorted({s.kind for s in spans})

        score, inj_signals = d.injection_score(text)
        if score >= self.injection_threshold:
            return GuardDecision(
                action="block",
                text=text,
                signals=redaction_signals + inj_signals,
                reason=f"prompt injection (score {score}, {', '.join(inj_signals)})",
            )
        if spans:
            return GuardDecision(
                action="redact",
                text=text,
                signals=redaction_signals,
                reason=f"redacted {len(spans)} sensitive span(s): {', '.join(redaction_signals)}",
            )
        return GuardDecision(action="allow", text=query)


class ToolGuard:
    """Screens a tool call before it is dispatched.

    The `run_sql` sandbox already blocks filesystem and network access (verified in M2). This
    is a second, policy-level gate whose job is to make an unsafe tool call an *auditable
    event* in the trace, not merely a caught exception. Defense in depth plus a record the
    flywheel can mine: a model repeatedly reaching for blocked SQL is a signal.
    """

    def check(self, name: str, arguments: dict) -> GuardDecision:
        if name == "run_sql":
            sql = str(arguments.get("sql", ""))
            if _UNSAFE_SQL.search(sql):
                return GuardDecision(
                    action="block",
                    text=sql,
                    signals=["unsafe_sql"],
                    reason="tool call would touch the filesystem/extensions; refused at policy",
                )
        return GuardDecision(action="allow", text="")


class OutputGuard:
    """Screens the final answer before it is returned or stored.

    Two jobs: strip any secret the model echoed (it should not, but the answer is persisted),
    and catch an answer that reproduces the system prompt — the tail end of a prompt-leak
    attack that slipped past the input gate.
    """

    # A short, distinctive phrase from the system prompt. If it appears verbatim in an
    # answer, the model is reciting its instructions.
    _LEAK_MARKER = "You are a DuckDB support engineer"

    def check(self, answer: str) -> GuardDecision:
        spans = d.detect_secrets(answer)
        if spans:
            return GuardDecision(
                action="redact",
                text=d.redact(answer, spans),
                signals=sorted({s.kind for s in spans}),
                reason="redacted secret(s) from model output",
            )
        if self._LEAK_MARKER.lower() in answer.lower():
            return GuardDecision(
                action="block",
                text=("I can't share my internal instructions. "
                      "What DuckDB question can I help with?"),
                signals=["system_prompt_leak"],
                reason="answer reproduced the system prompt",
            )
        return GuardDecision(action="allow", text=answer)
