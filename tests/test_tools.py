"""Tool behaviour, including the SQL sandbox. Offline."""

from __future__ import annotations

import pytest

from src.agent.tools import RunSqlTool, SearchDocsTool
from src.index.store import HybridIndex
from src.retrieval.pipeline import HybridRetriever, RetrievalConfig
from src.types import Chunk, stable_id


@pytest.fixture
def search_tool(tmp_path) -> SearchDocsTool:
    index = HybridIndex("duckdb", embedder_name="hashing", root=tmp_path)
    index.add(
        [
            Chunk(
                chunk_id=stable_id("duckdb", "qualify"),
                doc_id="qualify",
                tenant="duckdb",
                text="QUALIFY filters the output of a window function.",
                source_path="sql/qualify.md",
                heading_path=("DuckDB", "QUALIFY"),
            )
        ]
    )
    return SearchDocsTool(HybridRetriever(index, RetrievalConfig(mode="bm25")), tenant="duckdb")


# --- search ---------------------------------------------------------------------------


def test_search_returns_passages_with_citation_ids(search_tool):
    out = search_tool.run(query="qualify window").content
    assert stable_id("duckdb", "qualify") in out
    assert "QUALIFY filters" in out


def test_search_tracks_what_the_agent_was_shown(search_tool):
    """retrieved_ids is the citation whitelist — it must reflect only what was returned."""
    assert search_tool.retrieved_ids == set()
    search_tool.run(query="qualify")
    assert search_tool.retrieved_ids == {stable_id("duckdb", "qualify")}


def test_search_miss_tells_the_model_to_rephrase(search_tool):
    assert "different wording" in search_tool.run(query="zzzz").content


def test_search_k_is_clamped(search_tool):
    search_tool.run(query="qualify", k=999)  # must not raise
    assert search_tool.queries == ["qualify"]


# --- sql sandbox ----------------------------------------------------------------------


def test_select_runs_and_returns_rows():
    result = RunSqlTool().run(sql="SELECT 42 AS answer")
    assert not result.is_error
    assert "42" in result.content


def test_syntax_error_is_returned_verbatim_for_the_model_to_fix():
    result = RunSqlTool().run(sql="SELEKT 1")
    assert result.is_error
    assert "Error" in result.content or "error" in result.content.lower()


def test_agent_can_build_a_worked_example_across_calls():
    tool = RunSqlTool()
    tool.run(sql="CREATE TABLE t (a INT); INSERT INTO t VALUES (1), (2), (3)")
    result = tool.run(sql="SELECT sum(a) AS total FROM t")
    assert "6" in result.content


@pytest.mark.parametrize(
    "sql",
    [
        "COPY (SELECT 1) TO '/tmp/pwned.csv'",
        "ATTACH '/etc/passwd' AS leak",
        "INSTALL httpfs",
        "LOAD httpfs",
    ],
)
def test_filesystem_and_extension_access_is_refused(sql):
    """Model-generated SQL is untrusted input. The screen gives a clear, teachable error."""
    result = RunSqlTool().run(sql=sql)
    assert result.is_error
    assert "sandbox" in result.content.lower()


def test_sandbox_blocks_writes_even_without_the_screen():
    """Defense in depth: the DuckDB setting is the real control, not the regex."""
    tool = RunSqlTool()
    conn = tool._connect()
    with pytest.raises(Exception, match="Permission|permission|not allowed"):
        conn.execute("COPY (SELECT 1) TO '/tmp/should_not_exist.csv'")


def test_row_output_is_capped():
    result = RunSqlTool().run(sql="SELECT * FROM range(500)")
    assert "truncated" in result.content
    assert len(result.content.splitlines()) < 40


def test_zero_row_result_is_reported_clearly():
    assert "0 rows" in RunSqlTool().run(sql="SELECT 1 WHERE false").content


def test_executions_are_recorded_for_the_trace():
    tool = RunSqlTool()
    tool.run(sql="SELECT 1")
    tool.run(sql="SELEKT 1")
    assert [ok for _, ok in tool.executions] == [True, False]
