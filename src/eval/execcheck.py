"""Execution-based verification — the objective oracle this domain makes possible.

Because DuckDB answers are SQL, a golden case can be checked by *running* the query and
comparing rows, not by asking another model whether it looks right. That is the whole reason
M0 committed to this corpus: execution is ground truth for "does this query return the right
result", and a judge-only signal is exactly what reward-hacking exploits in M5.

The check: pull the SQL out of an answer's fenced code blocks, run each candidate against a
setup script in the sandboxed in-memory DuckDB, and see whether any of them reproduces the
expected result. "Any" rather than "the last" because an answer often shows two equivalent
forms — the check passes if the answer contains a working one.

Reuses the same sandbox contract as the agent's run_sql tool: `enable_external_access=false`,
no filesystem, no network. Model/answer SQL is untrusted input.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field

_FENCE = re.compile(r"```(?:sql)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_BLOCKED = re.compile(
    r"\b(?:ATTACH|DETACH|INSTALL|LOAD|COPY|EXPORT|IMPORT|SET\s+enable_external_access)\b",
    re.IGNORECASE,
)
TIMEOUT_S = 5.0


@dataclass
class ExecResult:
    checked: bool  # was execution attempted (did the answer contain runnable SQL)
    passed: bool  # did any candidate reproduce the expected result
    n_candidates: int = 0
    detail: str = ""
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "checked": self.checked,
            "passed": self.passed,
            "n_candidates": self.n_candidates,
            "detail": self.detail,
        }


def extract_sql_blocks(answer: str) -> list[str]:
    """SQL from fenced code blocks. A block may contain several statements."""
    blocks = []
    for m in _FENCE.finditer(answer):
        block = m.group(1).strip()
        if block and not _BLOCKED.search(block):
            blocks.append(block)
    return blocks


def _run(setup: str, query: str, expected) -> tuple[bool, str]:
    """Run setup then query in a fresh sandbox; compare rows to `expected` (a list of tuples).
    Returns (matched, detail)."""
    import duckdb

    result: dict = {}

    def _exec() -> None:
        try:
            conn = duckdb.connect(":memory:")
            conn.execute("SET enable_external_access=false")
            if setup:
                conn.execute(setup)
            rows = conn.execute(query).fetchall()
            result["rows"] = [tuple(r) for r in rows]
        except Exception as exc:  # noqa: BLE001 - the error text is the useful output
            result["err"] = f"{type(exc).__name__}: {exc}"

    t = threading.Thread(target=_exec, daemon=True)
    t.start()
    t.join(timeout=TIMEOUT_S)
    if t.is_alive():
        return False, f"timeout >{TIMEOUT_S:.0f}s"
    if "err" in result:
        return False, result["err"]

    got = result["rows"]
    want = [tuple(r) for r in expected]
    # Order-insensitive by default: a support answer's SELECT need not match row order unless
    # the question is about ordering. Compare as multisets.
    if sorted(map(str, got)) == sorted(map(str, want)):
        return True, f"matched {len(got)} row(s)"
    return False, f"got {got[:5]}, want {want[:5]}"


def check_answer(answer: str, setup: str, query_expectation: list, expected: list) -> ExecResult:
    """Verify an answer against an execution expectation.

    `setup` builds the fixture tables. Each SQL block in the answer is tried as the query
    (statements before the final SELECT are run as setup within that block). The check passes
    if any block's final statement reproduces `expected`.
    """
    blocks = extract_sql_blocks(answer)
    if not blocks:
        return ExecResult(checked=False, passed=False, detail="no runnable SQL in answer")

    errors = []
    for block in blocks:
        # A block may itself create tables then select; run the whole block as setup+query by
        # splitting the trailing statement off.
        stmts = [s.strip() for s in block.split(";") if s.strip()]
        if not stmts:
            continue
        block_setup = ";\n".join(stmts[:-1])
        query = stmts[-1]
        full_setup = ";\n".join(s for s in (setup, block_setup) if s)
        matched, detail = _run(full_setup, query, expected)
        if matched:
            return ExecResult(
                checked=True, passed=True, n_candidates=len(blocks), detail=detail
            )
        errors.append(detail)

    return ExecResult(
        checked=True,
        passed=False,
        n_candidates=len(blocks),
        detail=f"no candidate matched ({len(blocks)} tried)",
        errors=errors,
    )
