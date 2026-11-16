"""Lexical baselines: a PepperUni-style keyword score and BM25."""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[a-z][a-z0-9+#]*")
STOPWORDS = set("""
a about above after again against all also am an and any are as at be because been before being
below between both but by can could did do does doing down during each few for from further had has
have having he her here hers herself him himself his how i if in into is it its itself just me more
most my myself no nor not now of off on once only or other our ours ourselves out over own same she
should so some such than that the their theirs them themselves then there these they this those
through to too under until up very was we were what when where which while who whom why will with
would you your yours yourself yourselves etc per via within without across including include
includes ability able must may well work working works job jobs role position team company new
experience experienced years year strong excellent skills skill required requirements preferred
qualifications responsibilities responsible using use used one two three plus make ensure provide
including within others every like get help need needs needed us our we're you'll you're will
""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall((text or "").lower()) if len(t) >= 3 and t not in STOPWORDS]


def job_keywords(title: str, requirements: str, description: str, n: int = 25) -> list[str]:
    """Top-n keywords of a posting: title words first, then the most frequent body words.

    Mirrors the PepperUni idea of a 0-100 score from "job-description keywords and skills":
    the requirements section is used when one was found, otherwise the full description.
    """
    title_toks = list(dict.fromkeys(tokenize(title)))
    body = Counter(tokenize(requirements or description))
    for t in title_toks:
        body.pop(t, None)
    return (title_toks + [t for t, _ in body.most_common(n)])[:n]


def keyword_score(job_kw: list[str], resume_tokens: set[str]) -> float:
    """Share of the job's keywords that appear anywhere in the resume, 0-100."""
    return 100.0 * sum(k in resume_tokens for k in job_kw) / len(job_kw) if job_kw else 0.0


def matched_keywords(job_kw: list[str], resume_tokens: set[str]) -> list[str]:
    return [k for k in job_kw if k in resume_tokens]


class KeywordRanker:
    def __init__(self, jobs):
        self.kw = [job_keywords(j.title, j.requirements, j.description) for j in jobs.itertuples()]

    def scores(self, resume_texts: list[str]) -> np.ndarray:
        out = np.zeros((len(resume_texts), len(self.kw)), dtype=np.float32)
        for i, txt in enumerate(resume_texts):
            toks = set(tokenize(txt))
            out[i] = [keyword_score(kw, toks) for kw in self.kw]
        return out


class BM25Ranker:
    """BM25 (Okapi, k1=1.5, b=0.75 library defaults) over job title + description.

    The resume is the query. Query terms are de-duplicated: a long resume repeats words
    many times and rank_bm25 would otherwise add each repeat's score again.
    """

    def __init__(self, jobs):
        self.bm25 = BM25Okapi([tokenize(j.full) for j in jobs.itertuples()])

    def scores(self, resume_texts: list[str]) -> np.ndarray:
        return np.stack([self.bm25.get_scores(list(dict.fromkeys(tokenize(t)))) for t in resume_texts]).astype(np.float32)
