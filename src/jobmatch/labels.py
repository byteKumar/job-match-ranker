"""Build the labeled evaluation set.

Design (agreed with Chaman before coding):

* Queries are resumes. 2 resumes per category (48) are held out as TEST queries and never
  used for training or tuning anything.
* The EVAL CORPUS is ~40 jobs per category whose title maps to a category. Every test
  query is scored against every corpus job with the weak grade (2 same / 1 related / 0),
  so every job a model ranks has a label. That avoids the "unjudged = irrelevant" bias you
  get when only a handful of jobs per query are labeled.
* The CORE labeled set is 500 graded pairs: 300 TRAIN pairs (train resumes x jobs outside
  the eval corpus, 100 per grade) used for fine-tuning, and 200 TEST pairs sampled from
  the test grid. Resumes and jobs never cross the train/test line.
* The HAND-CHECK sample is 100 pairs drawn at random from the core set, stratified by weak
  grade so each grade gets ~33 checks (a plain random sample would be ~85% grade 0 in the
  test grid and would make agreement look better than it is).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .categories import grade

SEED = 42
TEST_RESUMES_PER_CAT = 2
EVAL_JOBS_PER_CAT = 40
TRAIN_PAIRS_PER_GRADE = 100
TEST_CORE_PAIRS = 200
HANDCHECK_N = 100


def split_resumes(resumes: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_idx = []
    for _, g in resumes.groupby("category"):
        test_idx += list(rng.choice(g.index.to_numpy(), size=min(TEST_RESUMES_PER_CAT, len(g)), replace=False))
    test = resumes.loc[sorted(test_idx)]
    train = resumes.drop(index=test.index)
    return train, test


def pick_eval_corpus(jobs: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    mapped = jobs[jobs["category"].notna()]
    idx = []
    for _, g in mapped.groupby("category"):
        idx += list(rng.choice(g.index.to_numpy(), size=min(EVAL_JOBS_PER_CAT, len(g)), replace=False))
    return jobs.loc[sorted(idx)]


def dense_qrels(test_resumes: pd.DataFrame, corpus: pd.DataFrame) -> pd.DataFrame:
    rows = [
        (r.resume_id, j.job_id, grade(r.category, j.category))
        for r in test_resumes.itertuples() for j in corpus.itertuples()
    ]
    return pd.DataFrame(rows, columns=["resume_id", "job_id", "grade"])


def sample_train_pairs(train_resumes: pd.DataFrame, jobs: pd.DataFrame,
                       rng: np.random.Generator) -> pd.DataFrame:
    """Sample TRAIN_PAIRS_PER_GRADE pairs per grade from train resumes x non-eval jobs."""
    pool = jobs[jobs["category"].notna()]
    by_cat = {c: g for c, g in pool.groupby("category")}
    out, seen = [], set()
    want = {0: TRAIN_PAIRS_PER_GRADE, 1: TRAIN_PAIRS_PER_GRADE, 2: TRAIN_PAIRS_PER_GRADE}
    from .categories import RELATED, CATEGORIES
    tries = 0
    while any(want.values()):
        tries += 1
        if tries > 100_000:
            raise RuntimeError(f"could not fill train pairs, still need {want}")
        g = int(rng.choice([k for k, v in want.items() if v]))
        r = train_resumes.iloc[int(rng.integers(len(train_resumes)))]
        if g == 2:
            cats = [r.category]
        elif g == 1:
            cats = sorted(RELATED[r.category])
        else:
            cats = [c for c in CATEGORIES if c != r.category and c not in RELATED[r.category]]
        cats = [c for c in cats if c in by_cat]
        if not cats:
            continue
        jg = by_cat[str(rng.choice(cats))]
        j = jg.iloc[int(rng.integers(len(jg)))]
        if (r.resume_id, j.job_id) in seen:
            continue
        seen.add((r.resume_id, j.job_id))
        out.append((r.resume_id, j.job_id, grade(r.category, j.category), "train"))
        want[g] -= 1
    return pd.DataFrame(out, columns=["resume_id", "job_id", "grade", "split"])


def sample_stratified(df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Sample n rows split as evenly as possible across grades 0/1/2."""
    per = [n // 3 + (1 if i < n % 3 else 0) for i in range(3)]  # e.g. 100 -> 34/33/33
    parts = []
    for g, k in zip((0, 1, 2), per):
        sub = df[df["grade"] == g]
        parts.append(sub.loc[rng.choice(sub.index.to_numpy(), size=min(k, len(sub)), replace=False)])
    return pd.concat(parts).sample(frac=1, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)


def build(resumes: pd.DataFrame, jobs: pd.DataFrame, seed: int = SEED) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train_res, test_res = split_resumes(resumes, rng)
    corpus = pick_eval_corpus(jobs, rng)
    qrels = dense_qrels(test_res, corpus)
    train_pairs = sample_train_pairs(train_res, jobs.drop(index=corpus.index), rng)
    test_core = sample_stratified(qrels, TEST_CORE_PAIRS, rng).assign(split="test")
    core = pd.concat([train_pairs, test_core], ignore_index=True)
    handcheck = sample_stratified(core, HANDCHECK_N, rng)
    handcheck.insert(0, "pair_id", range(1, len(handcheck) + 1))
    return {
        "test_resumes": test_res[["resume_id", "category"]].reset_index(drop=True),
        "train_resumes": train_res[["resume_id", "category"]].reset_index(drop=True),
        "eval_corpus": corpus[["job_id", "category"]].reset_index(drop=True),
        "test_qrels": qrels,
        "core_pairs": core,
        "handcheck_sample": handcheck,
    }
