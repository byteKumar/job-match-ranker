"""Checks on the built evaluation set. Skipped until scripts/prepare_data.py has run."""
import pandas as pd
import pytest

from handcheck import cohen_kappa
from jobmatch.categories import grade
from jobmatch.data import PROCESSED

needs_data = pytest.mark.skipif(not (PROCESSED / "core_pairs.parquet").exists(),
                                reason="run scripts/prepare_data.py first")


def load(name):
    return pd.read_parquet(PROCESSED / f"{name}.parquet")


@needs_data
def test_no_resume_or_job_leaks_between_train_and_test():
    core, corpus = load("core_pairs"), load("eval_corpus")
    train, test = core[core.split == "train"], core[core.split == "test"]
    assert set(train.resume_id).isdisjoint(set(load("test_resumes").resume_id))
    assert set(train.job_id).isdisjoint(set(corpus.job_id))
    assert set(test.resume_id) <= set(load("test_resumes").resume_id)


@needs_data
def test_qrels_are_dense_and_consistent():
    q, tr, corpus = load("test_qrels"), load("test_resumes"), load("eval_corpus")
    assert len(q) == len(tr) * len(corpus)          # every (query, corpus job) pair is labeled
    cat_r = dict(zip(tr.resume_id, tr.category))
    cat_j = dict(zip(corpus.job_id, corpus.category))
    sample = q.sample(500, random_state=0)
    assert all(grade(cat_r[r], cat_j[j]) == g for r, j, g in sample.itertuples(index=False))


@needs_data
def test_core_and_handcheck_sizes():
    core, hc = load("core_pairs"), load("handcheck_sample")
    assert len(core) == 500 and (core.split == "train").sum() == 300
    assert len(hc) == 100 and sorted(hc.grade.value_counts()) == [33, 33, 34]
    assert not core.duplicated(["resume_id", "job_id"]).any()


def test_cohen_kappa():
    assert cohen_kappa([0, 1, 2, 0], [0, 1, 2, 0]) == 1.0
    assert abs(cohen_kappa([0, 0, 1, 1], [0, 1, 0, 1])) < 1e-9   # agreement at chance level
