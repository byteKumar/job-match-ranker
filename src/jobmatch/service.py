"""FastAPI ranking service: POST /rank with resume text, returns top-k jobs with matched skills.

Run:  .venv/bin/uvicorn jobmatch.service:app --app-dir src --port 8010
The index must be built first: .venv/bin/python scripts/build_index.py

Pipeline per request (all on CPU by default, JOBMATCH_DEVICE overrides):
  1. clean the resume text and put title/skills/summary/experience first (section-aware)
  2. bi-encoder embedding (all-MiniLM-L6-v2) -> FAISS exact inner-product search over 10k jobs
  3. optional: re-rank the top 50 with the cross-encoder (JOBMATCH_RERANKER path)
  4. matched skills = the job's keywords that also appear in the resume
"""

from __future__ import annotations

import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .clean import normalize
from .lexical import matched_keywords, tokenize
from .neural import ROOT, BiEncoder, CrossEncoderReranker  # neural sets HF_HOME before model loads

INDEX_DIR = Path(os.environ.get("JOBMATCH_INDEX_DIR", ROOT / "indexes" / "jobs10k"))
DEVICE = os.environ.get("JOBMATCH_DEVICE", "cpu")
RERANKER = os.environ.get("JOBMATCH_RERANKER", str(ROOT / "models" / "ce_seed0"))
RERANK_K = 50

_HEADINGS = {
    "skills": r"skills|skill highlights|highlights|technical skills|core qualifications|qualifications|certifications",
    "summary": r"summary|professional summary|profile|objective|career overview",
    "experience": r"experience|work experience|professional experience|work history|employment history",
    "education": r"education|education and training",
}
_HEAD_RX = re.compile(r"^\s*(%s)\s*:?\s*$" % "|".join(_HEADINGS.values()), re.I)


def plain_resume_embed_text(text: str) -> str:
    """Section-aware ordering for plain-text resumes (the API gets text, not the dataset's HTML).

    Lines that look like section headings split the text; the first non-empty line is taken as
    the title. Output order matches training/eval: title, skills, summary, experience. If no
    headings are found the normalized text is used as is.
    """
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return ""
    title, sections, current = normalize(lines[0]), {}, None
    for ln in lines[1:]:
        m = _HEAD_RX.match(ln)
        if m:
            head = m.group(1).lower()
            current = next(k for k, rx in _HEADINGS.items() if re.fullmatch(rx, head))
            continue
        if current:
            sections.setdefault(current, []).append(normalize(ln))
    if not sections:
        return normalize(text)
    parts = [title] + [" ; ".join(sections[k]) for k in ("skills", "summary", "experience") if k in sections]
    return " . ".join(p for p in parts if p)


class RankRequest(BaseModel):
    resume_text: str = Field(min_length=1, max_length=100_000)
    k: int = Field(10, ge=1, le=50)
    rerank: bool = False


class JobHit(BaseModel):
    job_id: str
    title: str
    score: float
    matched_skills: list[str]


class RankResponse(BaseModel):
    results: list[JobHit]
    timings_ms: dict[str, float]


class _State:
    jobs: pd.DataFrame
    keywords: list[list[str]]
    index: object
    encoder: BiEncoder
    reranker: CrossEncoderReranker | None = None


state = _State()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    import faiss
    state.jobs = pd.read_parquet(INDEX_DIR / "jobs.parquet")
    state.keywords = state.jobs["keywords"].tolist()
    state.index = faiss.read_index(str(INDEX_DIR / "jobs.faiss"))
    state.encoder = BiEncoder(device=DEVICE)
    if Path(RERANKER).exists():
        state.reranker = CrossEncoderReranker(RERANKER, device=DEVICE)
    yield


app = FastAPI(title="job-match-ranker", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"jobs": len(state.jobs), "device": DEVICE, "reranker": state.reranker is not None}


@app.post("/rank", response_model=RankResponse)
def rank(req: RankRequest) -> RankResponse:
    t = {}
    t0 = time.perf_counter()
    text = plain_resume_embed_text(req.resume_text)
    q = state.encoder.encode([text])
    t["embed"] = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    n = RERANK_K if req.rerank else req.k
    scores, ids = state.index.search(q, n)
    ids, scores = ids[0], scores[0]
    t["search"] = (time.perf_counter() - t1) * 1000

    if req.rerank:
        if state.reranker is None:
            raise HTTPException(503, "no re-ranker model loaded")
        t2 = time.perf_counter()
        ce = state.reranker.score(text, state.jobs["embed_text"].iloc[ids].tolist())
        order = np.argsort(-ce, kind="stable")[: req.k]
        ids, scores = ids[order], ce[order]
        t["rerank"] = (time.perf_counter() - t2) * 1000

    resume_tokens = set(tokenize(req.resume_text))
    hits = [
        JobHit(job_id=state.jobs.job_id.iat[i], title=state.jobs.title.iat[i], score=float(s),
               matched_skills=matched_keywords(state.keywords[i], resume_tokens))
        for i, s in zip(ids[: req.k], scores[: req.k])
    ]
    t["total"] = (time.perf_counter() - t0) * 1000
    return RankResponse(results=hits, timings_ms={k: round(v, 3) for k, v in t.items()})
