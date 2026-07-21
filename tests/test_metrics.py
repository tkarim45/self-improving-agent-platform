from __future__ import annotations

import math

from src.eval.metrics import mean, mrr, ndcg_at_k, precision_at_k, recall_at_k

RANKED = ["a", "b", "c", "d"]


def test_recall_counts_only_within_k():
    assert recall_at_k(RANKED, ["c"], 2) == 0.0
    assert recall_at_k(RANKED, ["c"], 3) == 1.0


def test_recall_with_several_relevant_is_a_fraction():
    assert recall_at_k(RANKED, ["a", "z"], 4) == 0.5


def test_recall_with_no_labels_is_zero_not_a_crash():
    assert recall_at_k(RANKED, [], 3) == 0.0


def test_precision_divides_by_k_not_by_hits():
    assert precision_at_k(RANKED, ["a", "b"], 4) == 0.5
    assert precision_at_k(RANKED, ["a"], 0) == 0.0


def test_mrr_is_reciprocal_of_first_hit_rank():
    assert mrr(RANKED, ["b"]) == 0.5
    assert mrr(RANKED, ["z"]) == 0.0


def test_mrr_respects_a_cutoff():
    assert mrr(RANKED, ["d"], k=2) == 0.0
    assert mrr(RANKED, ["d"], k=4) == 0.25


def test_ndcg_is_one_when_ranking_is_ideal():
    assert math.isclose(ndcg_at_k(["a", "b", "c"], ["a", "b"], 3), 1.0)


def test_ndcg_penalises_a_worse_ordering():
    good = ndcg_at_k(["a", "x", "y"], ["a"], 3)
    bad = ndcg_at_k(["x", "y", "a"], ["a"], 3)
    assert good > bad


def test_ndcg_credits_every_relevant_hit_unlike_mrr():
    """Against a fixed relevant set, finding both hits beats finding one — MRR cannot see it.

    The relevant set has to be held constant to show this. nDCG normalizes by the ideal
    ranking for that set, so "one of one" and "two of two" are both a perfect 1.0.
    """
    relevant = ["a", "b"]
    found_one = ndcg_at_k(["a", "x"], relevant, 2)
    found_both = ndcg_at_k(["a", "b"], relevant, 2)
    assert found_both > found_one
    # MRR only looks at the first hit, so it calls these two rankings identical.
    assert mrr(["a", "x"], relevant) == mrr(["a", "b"], relevant)


def test_mean_of_empty_is_zero():
    assert mean([]) == 0.0
