"""Load the test queries, eval corpus and the dense grade matrix in a fixed order."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import PROCESSED


@dataclass
class EvalSet:
    queries: pd.DataFrame   # test resumes, full cleaned rows
    corpus: pd.DataFrame    # eval corpus jobs, full cleaned rows
    grades: np.ndarray      # [n_queries, n_corpus] int8 weak grades


def load_evalset() -> EvalSet:
    resumes = pd.read_parquet(PROCESSED / "resumes.parquet")
    jobs = pd.read_parquet(PROCESSED / "jobs.parquet")
    tq = pd.read_parquet(PROCESSED / "test_resumes.parquet")
    ec = pd.read_parquet(PROCESSED / "eval_corpus.parquet")
    qrels = pd.read_parquet(PROCESSED / "test_qrels.parquet")
    queries = tq[["resume_id"]].merge(resumes, on="resume_id", how="left")
    corpus = ec[["job_id"]].merge(jobs, on="job_id", how="left")
    g = qrels.pivot(index="resume_id", columns="job_id", values="grade")
    g = g.loc[queries.resume_id, corpus.job_id]
    return EvalSet(queries, corpus, g.to_numpy().astype(np.int8))
