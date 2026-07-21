"""Agent tools.

Two tools, chosen because they map to how a real DuckDB maintainer answers a question:
look it up in the docs, and run the SQL to check.

`run_sql` is the one that makes this domain worth picking. The agent can execute the query
it is about to recommend and see whether it actually works — grounding the answer in
execution rather than in the model's confidence. It is also the mechanism M5 needs: a golden
case can be verified by *running* it, so the flywheel has a signal that reward-hacking a
judge cannot fake.

SECURITY: `run_sql` executes model-generated SQL. It runs in a fresh in-memory database with
`enable_external_access=false`, which was verified to block `COPY ... TO` with a
PermissionException — so the model cannot read or write the filesystem, attach a database, or
install an extension. Statements are also screened before execution and the call is bounded
by a timeout. The sandbox is the load-bearing control; the screen is defense in depth.
"""

from __future__ import annotations

import re
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from src.index.store import HybridIndex
from src.retrieval.pipeline import HybridRetriever
from src.types import Chunk

# Screened even though the sandbox already blocks these — a clear tool error teaches the
# model to stop trying, where a raw PermissionException reads like a transient failure.
_BLOCKED = re.compile(
    r"\b(?:ATTACH|DETACH|INSTALL|LOAD|COPY|EXPORT|IMPORT|SET\s+enable_external_access)\b",
    re.IGNORECASE,
)

SQL_TIMEOUT_S = 5.0
MAX_ROWS = 20


@dataclass
class ToolResult:
    content: str
    is_error: bool = False


class Tool(ABC):
    name: str
    description: str
    input_schema: dict[str, Any]

    @abstractmethod
    def run(self, **kwargs: Any) -> ToolResult: ...

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class SearchDocsTool(Tool):
    """Retrieval, exposed as a tool so the agent can search more than once.

    The agent controls the query text, which matters: a user's phrasing is often not the
    phrasing that retrieves well, and letting the model re-query is the cheapest fix for the
    multi-hop gap M1 measured (coverage@10 stuck at 0.667). M1's finding was that a bridge
    page can't be reached by ranking alone — an agent that searches twice can reach it.
    """

    name = "search_docs"
    description = (
        "Search the DuckDB documentation and return matching passages with their citation "
        "ids. Use this before answering any question about DuckDB behaviour, syntax, or "
        "configuration. Search again with different wording if the first results do not "
        "contain the answer, and search separately for each distinct sub-question."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to look up. Use the terminology the docs would use.",
            },
            "k": {
                "type": "integer",
                "description": "How many passages to return (default 5, max 10).",
            },
        },
        "required": ["query"],
    }

    def __init__(
        self, retriever: HybridRetriever, tenant: str = "duckdb", max_searches: int = 6
    ) -> None:
        self.retriever = retriever
        self.tenant = tenant
        self.seen: dict[str, Chunk] = {}
        self.queries: list[str] = []
        self.max_searches = max_searches

    def run(self, query: str, k: int = 5) -> ToolResult:
        # Measured need, not a precaution: on "why would a hash join be slower than a merge
        # join", the strong tier issued 12 searches over 6 turns, burned 35k input tokens and
        # $0.115, and produced no answer at all. It kept searching because nothing told it to
        # stop. The budget converts an open-ended loop into a forced decision.
        if len(self.queries) >= self.max_searches:
            return ToolResult(
                f"Search budget exhausted ({self.max_searches} searches). Stop searching and "
                "answer now using the passages you already have. If they do not cover the "
                "question, say so explicitly — that is a valid and useful answer.",
                is_error=True,
            )
        self.queries.append(query)
        hits = self.retriever.search(query, self.tenant, k=max(1, min(int(k), 10)))
        if not hits:
            return ToolResult(f"No passages matched {query!r}. Try different wording.")

        lines = []
        for hit in hits:
            chunk = hit.chunk
            self.seen[chunk.chunk_id] = chunk
            where = " > ".join(chunk.heading_path) or chunk.source_path
            body = " ".join(chunk.text.split())[:700]
            lines.append(f"[{chunk.chunk_id}] {where}\n{body}")
        return ToolResult("\n\n".join(lines))

    @property
    def retrieved_ids(self) -> set[str]:
        """Every chunk id the agent has actually been shown — the citation whitelist."""
        return set(self.seen)


class RunSqlTool(Tool):
    """Execute DuckDB SQL in a sandboxed in-memory database."""

    name = "run_sql"
    description = (
        "Execute a DuckDB SQL statement in a scratch in-memory database and return the "
        "result or the exact error. Use this to VERIFY that syntax you are about to "
        "recommend actually works, and to check what a query returns. You may CREATE and "
        "populate temporary tables to build an example. No filesystem or network access."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": "The SQL to run. Multiple statements separated by ';' are allowed.",
            }
        },
        "required": ["sql"],
    }

    def __init__(self) -> None:
        self._conn = None
        self.executions: list[tuple[str, bool]] = []

    def _connect(self):
        if self._conn is None:
            import duckdb

            self._conn = duckdb.connect(":memory:")
            # Verified to raise PermissionException on `COPY ... TO` — see module docstring.
            self._conn.execute("SET enable_external_access=false")
        return self._conn

    def run(self, sql: str) -> ToolResult:
        if _BLOCKED.search(sql):
            self.executions.append((sql, False))
            return ToolResult(
                "Blocked: this sandbox has no filesystem, network, or extension access. "
                "Use only in-memory SQL (CREATE TABLE / INSERT / SELECT).",
                is_error=True,
            )

        conn = self._connect()
        result: dict[str, Any] = {}

        def _execute() -> None:
            try:
                rows = conn.execute(sql).fetchmany(MAX_ROWS + 1)
                cols = [d[0] for d in (conn.description or [])]
                result["ok"] = (cols, rows)
            except Exception as exc:  # noqa: BLE001 - the error text is the useful output
                result["err"] = f"{type(exc).__name__}: {exc}"

        thread = threading.Thread(target=_execute, daemon=True)
        thread.start()
        thread.join(timeout=SQL_TIMEOUT_S)
        if thread.is_alive():
            # The connection is left behind rather than interrupted: DuckDB has no safe
            # cross-thread cancel here, and a wedged scratch DB is cheaper than corrupting
            # one. A fresh connection is made on the next call.
            self._conn = None
            self.executions.append((sql, False))
            return ToolResult(f"Query exceeded the {SQL_TIMEOUT_S:.0f}s timeout.", is_error=True)

        if "err" in result:
            self.executions.append((sql, False))
            return ToolResult(result["err"], is_error=True)

        cols, rows = result["ok"]
        self.executions.append((sql, True))
        if not rows:
            return ToolResult("OK (0 rows).")

        truncated = len(rows) > MAX_ROWS
        body = "\n".join(" | ".join(str(v) for v in row) for row in rows[:MAX_ROWS])
        header = " | ".join(cols) if cols else ""
        out = f"{header}\n{body}" if header else body
        return ToolResult(out + (f"\n... ({MAX_ROWS}+ rows, truncated)" if truncated else ""))


def build_tools(index: HybridIndex, retriever: HybridRetriever, tenant: str = "duckdb"):
    search = SearchDocsTool(retriever, tenant=tenant)
    sql = RunSqlTool()
    return search, sql, {t.name: t for t in (search, sql)}
