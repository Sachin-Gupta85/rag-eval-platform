import math

import pytest

from rageval import retrieval as r

RETRIEVED = ["d1", "d2", "d3", "d4", "d5"]
RELEVANT = {"d2", "d4"}


def test_precision_and_recall_at_k():
    assert r.precision_at_k(RETRIEVED, RELEVANT, 3) == pytest.approx(1 / 3)
    assert r.recall_at_k(RETRIEVED, RELEVANT, 3) == pytest.approx(0.5)
    assert r.precision_at_k(RETRIEVED, RELEVANT, 5) == pytest.approx(0.4)
    assert r.recall_at_k(RETRIEVED, RELEVANT, 5) == pytest.approx(1.0)


def test_hit_and_reciprocal_rank():
    assert r.hit_at_k(RETRIEVED, RELEVANT, 1) == 0.0      # d1 not relevant
    assert r.hit_at_k(RETRIEVED, RELEVANT, 2) == 1.0      # d2 is
    assert r.reciprocal_rank(RETRIEVED, RELEVANT) == pytest.approx(0.5)   # first at rank 2


def test_average_precision():
    # relevant at ranks 2 and 4: (1/2 + 2/4) / 2 = 0.5
    assert r.average_precision(RETRIEVED, RELEVANT) == pytest.approx(0.5)


def test_ndcg_at_k():
    actual = 1 / math.log2(3) + 1 / math.log2(5)         # gains at ranks 2 and 4
    ideal = 1 / math.log2(2) + 1 / math.log2(3)          # two relevant up front
    assert r.ndcg_at_k(RETRIEVED, RELEVANT, 5) == pytest.approx(actual / ideal)


def test_perfect_ranking_scores_one():
    perfect = ["d2", "d4", "d1", "d3"]
    assert r.average_precision(perfect, RELEVANT) == pytest.approx(1.0)
    assert r.ndcg_at_k(perfect, RELEVANT, 4) == pytest.approx(1.0)
    assert r.reciprocal_rank(perfect, RELEVANT) == pytest.approx(1.0)


def test_empty_relevant_is_zero_not_crash():
    assert r.recall_at_k(RETRIEVED, set(), 3) == 0.0
    assert r.average_precision(RETRIEVED, set()) == 0.0
    assert r.ndcg_at_k(RETRIEVED, set(), 3) == 0.0


def test_k_must_be_positive():
    with pytest.raises(ValueError):
        r.precision_at_k(RETRIEVED, RELEVANT, 0)
