"""Milestone 1: clean the raw data and build the labeled evaluation set.

Usage: .venv/bin/python scripts/prepare_data.py
Writes data/processed/*.parquet (git-ignored) and results/dataset_stats.json (committed).
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jobmatch import labels  # noqa: E402
from jobmatch.data import PROCESSED, ROOT, load_jobs, load_resumes  # noqa: E402


def main() -> None:
    t0 = time.perf_counter()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    resumes = load_resumes()
    jobs = load_jobs()
    resumes.to_parquet(PROCESSED / "resumes.parquet", index=False)
    jobs.to_parquet(PROCESSED / "jobs.parquet", index=False)

    sets = labels.build(resumes, jobs)
    for name, df in sets.items():
        df.to_parquet(PROCESSED / f"{name}.parquet", index=False)

    core, qrels = sets["core_pairs"], sets["test_qrels"]
    stats = {
        "resumes_after_cleaning": len(resumes),
        "jobs_after_cleaning_and_dedup": len(jobs),
        "jobs_with_mapped_category": int(jobs["category"].notna().sum()),
        "jobs_per_category_mapped": jobs["category"].value_counts().to_dict(),
        "test_queries": len(sets["test_resumes"]),
        "train_resumes": len(sets["train_resumes"]),
        "eval_corpus_jobs": len(sets["eval_corpus"]),
        "eval_corpus_per_category": sets["eval_corpus"]["category"].value_counts().to_dict(),
        "test_qrels_pairs": len(qrels),
        "test_qrels_grade_counts": qrels["grade"].value_counts().sort_index().to_dict(),
        "core_pairs": len(core),
        "core_split_counts": core["split"].value_counts().to_dict(),
        "core_grade_counts": core["grade"].value_counts().sort_index().to_dict(),
        "handcheck_sample": len(sets["handcheck_sample"]),
        "handcheck_grade_counts": sets["handcheck_sample"]["grade"].value_counts().sort_index().to_dict(),
        "seed": labels.SEED,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    stats = json.loads(json.dumps(stats, default=int))
    out = ROOT / "results" / "dataset_stats.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
