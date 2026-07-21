from __future__ import annotations

from src.agent import citations as cite

IDS = {"a1b2c3d4e5f6", "0011223344ff"}


def test_extracts_ids_in_order_without_duplicates():
    text = "One [a1b2c3d4e5f6]. Two [0011223344ff]. Again [a1b2c3d4e5f6]."
    assert cite.extract_citations(text) == ["a1b2c3d4e5f6", "0011223344ff"]


def test_invalid_citation_is_flagged():
    """An id that was never retrieved is an invented source — the serious failure."""
    report = cite.check("QUALIFY filters window output [deadbeefcafe].", IDS)
    assert report.invalid_ids == ["deadbeefcafe"]
    assert not report.grounded


def test_grounded_requires_at_least_one_real_citation():
    assert not cite.check("QUALIFY filters the output of a window function.", IDS).grounded
    assert cite.check("QUALIFY filters window output [a1b2c3d4e5f6].", IDS).grounded


def test_uncited_claim_is_counted():
    text = (
        "QUALIFY filters the results of a window function [a1b2c3d4e5f6]. "
        "It was added to DuckDB in a version I did not verify anywhere."
    )
    report = cite.check(text, IDS)
    assert len(report.uncited_claims) == 1
    assert report.n_claims == 2
    assert report.citation_rate == 0.5


def test_short_fragments_and_questions_are_not_claims():
    """Demanding a citation on 'Yes.' would inflate the uncited count with noise."""
    assert not cite.is_claim("Yes.")
    assert not cite.is_claim("What about window functions and their frames?")
    assert cite.is_claim("QUALIFY filters the results of a window function in DuckDB.")


def test_code_blocks_are_not_treated_as_prose():
    text = (
        "Use QUALIFY to filter window output [a1b2c3d4e5f6].\n"
        "```sql\n"
        "SELECT name FROM t QUALIFY row_number() OVER () <= 3;\n"
        "```\n"
    )
    report = cite.check(text, IDS)
    assert report.uncited_claims == []


def test_headings_and_bullets_are_skipped():
    text = (
        "## Summary\n"
        "- \n"
        "QUALIFY filters the results of a window function in DuckDB [a1b2c3d4e5f6]."
    )
    report = cite.check(text, IDS)
    assert report.n_claims == 1
    assert report.uncited_claims == []


def test_citation_after_the_period_still_counts():
    """Regression: the model puts the citation after the closing period, as instructed.

    A splitter that breaks on ". [" strands the citation as its own fragment and reports the
    sentence as uncited — this returned 0% on a real answer that carried three valid
    citations.
    """
    text = (
        "You cannot use WHERE to filter window function results, because they are "
        "evaluated after that clause. [a1b2c3d4e5f6]"
    )
    report = cite.check(text, IDS)
    assert report.n_claims == 1
    assert report.uncited_claims == []
    assert report.citation_rate == 1.0


def test_multi_sentence_answer_with_trailing_citations_scores_fully():
    text = (
        "QUALIFY filters the output of a window function in DuckDB. [a1b2c3d4e5f6] "
        "It behaves the way HAVING behaves for aggregate functions. [0011223344ff]"
    )
    report = cite.check(text, IDS)
    assert report.n_claims == 2
    assert report.citation_rate == 1.0


def test_strip_citations_gives_reader_text():
    assert cite.strip_citations("Filters output [a1b2c3d4e5f6].") == "Filters output."


def test_empty_answer_is_not_grounded():
    report = cite.check("", IDS)
    assert not report.grounded
    assert report.citation_rate == 0.0


def test_report_serializes_for_the_trace():
    report = cite.check("QUALIFY filters window function output [a1b2c3d4e5f6].", IDS)
    d = report.to_dict()
    assert d["grounded"] is True and d["n_uncited"] == 0
