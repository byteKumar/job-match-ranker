"""Load the raw CSVs, clean them, and write processed parquet files."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from .categories import title_to_category
from .clean import job_embed_text, resume_embed_text, structure_job, structure_resume

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

MIN_JOB_DESC_CHARS = 300
MIN_RESUME_CHARS = 500


def load_resumes(path: Path = RAW / "resumes" / "Resume.csv") -> pd.DataFrame:
    raw = pd.read_csv(path, dtype={"ID": str})
    rows = []
    for rec in raw.itertuples(index=False):
        s = structure_resume(rec.Resume_html, rec.Resume_str)
        if len(s["full"]) < MIN_RESUME_CHARS:
            continue
        rows.append({"resume_id": rec.ID, "category": rec.Category, **s, "embed_text": resume_embed_text(s)})
    df = pd.DataFrame(rows).drop_duplicates("full").reset_index(drop=True)
    return df


def load_jobs(path: Path = RAW / "jobs" / "job_postings.csv") -> pd.DataFrame:
    raw = pd.read_csv(path, dtype={"job_id": str}, usecols=[
        "job_id", "title", "description", "skills_desc", "formatted_experience_level"])
    raw = raw.fillna("")
    rows, seen = [], set()
    for rec in raw.itertuples(index=False):
        s = structure_job(rec.title, rec.description, rec.skills_desc)
        if len(s["description"]) < MIN_JOB_DESC_CHARS:
            continue
        # Many postings are reposted verbatim (same title + text); keep one copy.
        key = hashlib.sha1((s["title"].lower() + "|" + s["description"][:2000]).encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "job_id": rec.job_id, **s, "embed_text": job_embed_text(s),
            "experience_level": rec.formatted_experience_level,
            "category": title_to_category(s["title"]),
        })
    return pd.DataFrame(rows).reset_index(drop=True)


def load_processed() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (pd.read_parquet(PROCESSED / "resumes.parquet"),
            pd.read_parquet(PROCESSED / "jobs.parquet"))
