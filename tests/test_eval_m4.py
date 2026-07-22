"""M4 eval harness: online scorers, execution checker, LLM-judge, golden gate. Offline."""

from __future__ import annotations

from src.eval import execcheck, golden, judge, scorers
from src.types import Citation, Trace

# --- online scorers -------------------------------------------------------------------


def test_grounded_answer_scores_success():
    s = scorers.score_answer("QUALIFY filters windows [abc123].", ["abc123"], ["abc123"])
    assert s.groundedness == 1.0
    assert s.task_success == 1.0
    assert s.invalid_citations == 0


def test_invented_citation_is_not_grounded():
    s = scorers.score_answer("Filters windows [deadbeef].", ["deadbeef"], ["abc123"])
    assert s.groundedness == 0.0
    assert s.invalid_citations == 1


def test_correct_abstention_counts_as_success():
    """Saying 'the docs don't cover this' beats inventing an answer."""
    s = scorers.score_answer("The documentation does not cover that benchmark.", [], [])
    assert s.abstained
    assert s.task_success == 1.0  # abstention is a success, not a failure
    assert s.groundedness == 0.0


def test_ungrounded_non_abstention_fails():
    s = scorers.score_answer("QUALIFY does something I made up.", [], ["abc123"])
    assert s.task_success == 0.0


def test_score_trace_reads_from_the_trace():
    tr = Trace(
        trace_id="t", tenant="duckdb", query="q",
        answer="Filters windows [abc123].",
        retrieved=["abc123"], citations=[Citation(chunk_id="abc123")],
    )
    assert scorers.score_trace(tr).groundedness == 1.0


def test_aggregate_averages():
    s = [
        scorers.score_answer("Answer text here [abc123].", ["abc123"], ["abc123"]),
        scorers.score_answer("made up", [], ["x"]),
    ]
    agg = scorers.aggregate(s)
    assert agg["n"] == 2
    assert agg["groundedness"] == 0.5


def test_aggregate_of_empty():
    assert scorers.aggregate([])["n"] == 0


# --- execution checker (the objective oracle) -----------------------------------------


def test_correct_sql_passes_execution():
    ans = "Use:\n```sql\nSELECT unnest([10,20,30]) AS x\n```"
    r = execcheck.check_answer(ans, "", [], [[10], [20], [30]])
    assert r.checked and r.passed


def test_wrong_sql_fails_execution():
    ans = "```sql\nSELECT unnest([1,2]) AS x\n```"
    r = execcheck.check_answer(ans, "", [], [[10], [20], [30]])
    assert r.checked and not r.passed


def test_execution_is_order_insensitive():
    ans = "```sql\nSELECT unnest([30,10,20]) AS x\n```"
    assert execcheck.check_answer(ans, "", [], [[10], [20], [30]]).passed


def test_answer_with_no_sql_is_unchecked():
    r = execcheck.check_answer("Just use QUALIFY, no example.", "", [], [[1]])
    assert not r.checked
    assert not r.passed


def test_setup_tables_are_available_to_the_query():
    ans = "```sql\nSELECT sum(score) AS total FROM t\n```"
    setup = "CREATE TABLE t(score INT); INSERT INTO t VALUES (1),(2),(3)"
    assert execcheck.check_answer(ans, setup, [], [[6]]).passed


def test_answer_can_build_its_own_example():
    ans = "```sql\nCREATE TABLE x(a INT); INSERT INTO x VALUES (5),(7); SELECT sum(a) FROM x\n```"
    assert execcheck.check_answer(ans, "", [], [[12]]).passed


def test_any_of_several_blocks_passing_is_enough():
    ans = "Option A:\n```sql\nSELECT 1\n```\nOption B:\n```sql\nSELECT unnest([10,20,30]) AS x\n```"
    assert execcheck.check_answer(ans, "", [], [[10], [20], [30]]).passed


def test_filesystem_sql_in_answer_is_skipped():
    """A COPY inside an answer must not be executed by the checker."""
    ans = "```sql\nCOPY (SELECT 1) TO '/tmp/x.csv'\n```"
    r = execcheck.check_answer(ans, "", [], [[1]])
    assert not r.checked  # the only block was blocked → nothing runnable


def test_broken_sql_is_reported_not_crashed():
    r = execcheck.check_answer("```sql\nSELEKT 1\n```", "", [], [[1]])
    assert r.checked and not r.passed
    assert r.errors


# --- LLM judge (parsing, offline) -----------------------------------------------------


def test_judge_parses_trailing_json():
    text = 'The answer is well supported.\n{"faithfulness": 5, "relevance": 5, "completeness": 4}'
    j = judge.parse_judgment(text)
    assert j.faithfulness == 5 and j.passed


def test_judge_recomputes_verdict_from_scores():
    """A model that says pass but scores low is overridden — the rubric defines pass."""
    text = '{"faithfulness": 2, "relevance": 5, "completeness": 5, "verdict": "pass"}'
    assert not judge.parse_judgment(text).passed  # faith < 4 → fail regardless


def test_malformed_json_fails_safe():
    """A parsing failure must never silently pass a bad answer."""
    assert judge.parse_judgment("no json here at all").verdict == "fail"
    assert judge.parse_judgment('{"faithfulness": "broken').verdict == "fail"


def test_judge_clamps_out_of_range_scores():
    j = judge.parse_judgment('{"faithfulness": 9, "relevance": 0, "completeness": 3}')
    assert j.faithfulness == 5 and j.relevance == 1


def test_judge_uses_provider():
    from src.llm.base import LLMResponse
    from src.llm.fake import FakeProvider

    fake = FakeProvider([LLMResponse(
        text='ok\n{"faithfulness": 5, "relevance": 5, "completeness": 5}', stop_reason="end_turn"
    )])
    jm = judge.LLMJudge(fake).judge("q", "a", "passages")
    assert jm.passed
    assert fake.requests[0]["tier"] == "strong"


# --- golden gate ----------------------------------------------------------------------


EXEC_CASE = {
    "id": "g", "kind": "exec", "question": "q", "setup": "", "expected": [[10], [20], [30]]
}
REF_CASE = {"id": "r", "kind": "reference", "question": "q", "expect_pages": ["sql/pivot.md"]}
ABSTAIN_CASE = {"id": "a", "kind": "abstain", "question": "q"}


def test_score_exec_case():
    r = golden.score_case(EXEC_CASE, "```sql\nSELECT unnest([10,20,30]) AS x\n```", [], [])
    assert r.passed


def test_score_reference_case_by_cited_page():
    r = golden.score_case(REF_CASE, "Use PIVOT [x].", ["sql/pivot.md"], [])
    assert r.passed
    bad = golden.score_case(REF_CASE, "Use PIVOT [x].", ["sql/other.md"], [])
    assert not bad.passed


def test_score_abstain_case():
    good = golden.score_case(ABSTAIN_CASE, "The documentation does not cover this.", [], [])
    assert good.passed
    bad = golden.score_case(ABSTAIN_CASE, "The P99 latency is 4.2ms.", [], [])
    assert not bad.passed


def test_gate_passes_above_threshold():
    report = golden.GateReport(threshold=0.75)
    report.results = [
        golden.score_case(EXEC_CASE, "```sql\nSELECT unnest([10,20,30]) AS x\n```", [], []),
        golden.score_case(ABSTAIN_CASE, "The docs do not cover this.", [], []),
    ]
    assert report.passed
    assert report.score == 1.0


def test_gate_fails_below_threshold():
    report = golden.GateReport(threshold=0.75)
    report.results = [
        golden.score_case(EXEC_CASE, "```sql\nSELECT 1\n```", [], []),  # wrong
        golden.score_case(ABSTAIN_CASE, "The P99 is 4ms.", [], []),  # invented
    ]
    assert not report.passed


def test_gate_by_kind_breakdown():
    report = golden.GateReport()
    report.results = [
        golden.score_case(EXEC_CASE, "```sql\nSELECT unnest([10,20,30]) AS x\n```", [], []),
        golden.score_case(ABSTAIN_CASE, "not covered", [], []),
    ]
    bk = report.by_kind()
    assert bk["exec"] == (1, 1)
    assert bk["abstain"] == (1, 1)


def test_replay_scores_frozen_records():
    cases = [EXEC_CASE, ABSTAIN_CASE]
    records = [
        {"id": "g", "answer": "```sql\nSELECT unnest([10,20,30]) AS x\n```",
         "cited_pages": [], "retrieved": []},
        {"id": "a", "answer": "The documentation does not cover this.",
         "cited_pages": [], "retrieved": []},
    ]
    report = golden.gate_from_records(records, cases, threshold=0.75)
    assert report.passed and report.score == 1.0
