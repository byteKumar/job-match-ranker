import math

import numpy as np

from jobmatch.lexical import job_keywords, keyword_score, tokenize
from jobmatch.metrics import evaluate, ndcg_at_k, rank, recall_at_k, reciprocal_rank


def test_ndcg_perfect_and_reversed():
    grades = np.array([2, 1, 0, 0])
    assert ndcg_at_k(np.array([0, 1, 2, 3]), grades) == 1.0
    worst = ndcg_at_k(np.array([3, 2, 1, 0]), grades)
    # DCG = 1/log2(4) + 3/log2(5); IDCG = 3/log2(2) + 1/log2(3)
    expected = (1 / math.log2(4) + 3 / math.log2(5)) / (3 + 1 / math.log2(3))
    assert abs(worst - expected) < 1e-12


def test_mrr_counts_only_grade_2():
    grades = np.array([1, 0, 2])
    assert reciprocal_rank(np.array([0, 1, 2]), grades) == 1 / 3


def test_recall_at_k():
    grades = np.array([2, 2, 0, 2])
    assert recall_at_k(np.array([0, 2, 1, 3]), grades, k=2) == 1 / 3


def test_rank_is_stable_on_ties():
    assert rank(np.array([1.0, 1.0, 2.0])).tolist() == [2, 0, 1]


def test_evaluate_shapes():
    res = evaluate(np.array([[3.0, 2.0, 1.0]]), np.array([[2, 0, 0]]))
    assert res["mean"] == {"ndcg@10": 1.0, "mrr": 1.0, "recall@50": 1.0}


def test_keyword_score():
    kw = job_keywords("Staff Accountant", "", "Prepare journal entries and reconciliations in Excel.")
    assert kw[:2] == ["staff", "accountant"]
    assert keyword_score(kw, set(tokenize("accountant excel reconciliations"))) == 100 * 3 / len(kw)
