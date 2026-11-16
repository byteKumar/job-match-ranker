"""Ranking metrics over a fully judged corpus.

Definitions (fixed before any model was run):
* nDCG@10 uses graded gains 2^g - 1 (grade 2 -> 3, grade 1 -> 1, grade 0 -> 0) and a log2
  discount. The ideal DCG is computed from all of the query's grades in the corpus.
* MRR and Recall@50 count a job as relevant only at grade 2 (same occupation).
  MRR is 1 / rank of the first grade-2 job over the full ranking (no cutoff).
  Recall@50 = grade-2 jobs in the top 50 / all grade-2 jobs for the query.
Ties in model scores are broken by corpus order (stable sort), never by the labels.
"""

from __future__ import annotations

import numpy as np


def rank(scores: np.ndarray) -> np.ndarray:
    """Indices of corpus items sorted by descending score, stable for ties."""
    return np.argsort(-scores, kind="stable")


def dcg(grades_in_rank_order: np.ndarray, k: int) -> float:
    g = grades_in_rank_order[:k].astype(float)
    return float(((2.0 ** g - 1) / np.log2(np.arange(2, len(g) + 2))).sum())


def ndcg_at_k(order: np.ndarray, grades: np.ndarray, k: int = 10) -> float:
    ideal = dcg(np.sort(grades)[::-1], k)
    return dcg(grades[order], k) / ideal if ideal > 0 else 0.0


def reciprocal_rank(order: np.ndarray, grades: np.ndarray, rel: int = 2) -> float:
    hits = np.nonzero(grades[order] >= rel)[0]
    return 1.0 / (hits[0] + 1) if len(hits) else 0.0


def recall_at_k(order: np.ndarray, grades: np.ndarray, k: int = 50, rel: int = 2) -> float:
    total = int((grades >= rel).sum())
    return float((grades[order[:k]] >= rel).sum()) / total if total else 0.0


def evaluate(score_matrix: np.ndarray, grade_matrix: np.ndarray) -> dict:
    """score_matrix, grade_matrix: [n_queries, n_corpus]. Returns mean metrics + per-query rows."""
    per_query = []
    for s, g in zip(score_matrix, grade_matrix):
        o = rank(s)
        per_query.append({
            "ndcg@10": ndcg_at_k(o, g, 10),
            "mrr": reciprocal_rank(o, g),
            "recall@50": recall_at_k(o, g, 50),
        })
    mean = {m: float(np.mean([q[m] for q in per_query])) for m in ("ndcg@10", "mrr", "recall@50")}
    return {"mean": mean, "per_query": per_query}
