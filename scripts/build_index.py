"""Build the service index: 10,000 jobs -> bi-encoder embeddings -> FAISS flat inner-product index.

Usage: .venv/bin/python scripts/build_index.py [--n 10000] [--device mps|cpu]
Writes indexes/jobs10k/{jobs.parquet,jobs.faiss} (git-ignored) and results/index_build.json.
Build time = embedding the jobs + adding them to FAISS (+ writing the index), measured once.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import faiss  # noqa: E402
import torch  # noqa: E402

from jobmatch.data import PROCESSED, ROOT  # noqa: E402
from jobmatch.lexical import job_keywords  # noqa: E402
from jobmatch.neural import BiEncoder, build_index  # noqa: E402
import pandas as pd  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    out = ROOT / "indexes" / f"jobs{args.n // 1000}k"
    out.mkdir(parents=True, exist_ok=True)

    jobs = pd.read_parquet(PROCESSED / "jobs.parquet").sample(n=args.n, random_state=42).reset_index(drop=True)
    jobs["keywords"] = [job_keywords(j.title, j.requirements, j.description) for j in jobs.itertuples()]
    enc = BiEncoder(device=args.device)
    enc.encode(["warm up"])  # model load / first-call overhead is not part of the build

    t0 = time.perf_counter()
    vecs = enc.encode(jobs.embed_text.tolist())
    t_embed = time.perf_counter() - t0
    t1 = time.perf_counter()
    index = build_index(vecs)
    faiss.write_index(index, str(out / "jobs.faiss"))
    t_index = time.perf_counter() - t1

    jobs[["job_id", "title", "embed_text", "keywords"]].to_parquet(out / "jobs.parquet", index=False)
    info = {
        "jobs": args.n, "device": enc.device, "torch_threads": torch.get_num_threads(),
        "dim": int(vecs.shape[1]), "index_type": "IndexFlatIP (exact)",
        "embed_seconds": round(t_embed, 2), "faiss_add_and_write_seconds": round(t_index, 3),
        "build_seconds_total": round(t_embed + t_index, 2),
        "index_file_mb": round((out / "jobs.faiss").stat().st_size / 1e6, 2),
        "jobs_metadata_mb": round((out / "jobs.parquet").stat().st_size / 1e6, 2),
    }
    (ROOT / "results" / "index_build.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
