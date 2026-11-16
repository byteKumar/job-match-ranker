"""Latency test for POST /rank over the 10k-job index.

Usage: .venv/bin/python scripts/latency_test.py [--n 1000] [--n-rerank 300]
Starts uvicorn (1 worker, 127.0.0.1:8010, CPU), sends 20 warm-up requests, then N sequential
requests from one client, each with a different resume. Latency is client-side wall time per
request (HTTP + JSON included). p50/p99 use numpy's linear-interpolated percentiles.
Writes results/latency_<label>.json and latency_raw_<label>.csv (one row per request). Measured once; check that nothing else heavy is running first.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
URL = "http://127.0.0.1:8010"


def wait_ready(proc, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}")
        try:
            return httpx.get(URL + "/health", timeout=2).json()
        except httpx.HTTPError:
            time.sleep(0.5)
    raise RuntimeError("server did not start")


RAW = []  # per-request rows, written to results/latency_raw_<label>.csv


def run(client, texts, rerank: bool, n: int) -> dict:
    for t in texts[:20]:   # warm-up, not measured
        client.post("/rank", json={"resume_text": t, "k": 10, "rerank": rerank}).raise_for_status()
    lat, server = [], []
    for i in range(n):
        t = texts[20 + i % (len(texts) - 20)]
        t0 = time.perf_counter()
        r = client.post("/rank", json={"resume_text": t, "k": 10, "rerank": rerank})
        lat.append((time.perf_counter() - t0) * 1000)
        r.raise_for_status()
        server.append(r.json()["timings_ms"])
        RAW.append({"rerank": rerank, "i": i, "client_ms": round(lat[-1], 3), "chars": len(t),
                    "loadavg_1m": os.getloadavg()[0], **{f"server_{k}_ms": v for k, v in server[-1].items()}})
    lat = np.array(lat)
    stages = {k: round(float(np.median([s[k] for s in server])), 2) for k in server[0]}
    return {"requests": n, "p50_ms": round(float(np.percentile(lat, 50)), 2),
            "p99_ms": round(float(np.percentile(lat, 99)), 2), "mean_ms": round(float(lat.mean()), 2),
            "max_ms": round(float(lat.max()), 2), "server_stage_median_ms": stages}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="0 skips the retrieval-only mode")
    ap.add_argument("--n-rerank", type=int, default=300, help="0 skips the re-rank mode")
    ap.add_argument("--label", default="run1", help="output name: results/latency_<label>.json")
    args = ap.parse_args()
    texts = (pd.read_parquet(ROOT / "data/processed/resumes.parquet")
             .sample(frac=1, random_state=7)["full"].tolist())
    env = {**os.environ, "JOBMATCH_DEVICE": "cpu"}
    proc = subprocess.Popen([str(ROOT / ".venv/bin/uvicorn"), "jobmatch.service:app", "--app-dir", "src",
                             "--host", "127.0.0.1", "--port", "8010", "--workers", "1", "--log-level", "warning"],
                            cwd=ROOT, env=env)
    try:
        health = wait_ready(proc)
        with httpx.Client(base_url=URL, timeout=60) as client:
            out = {"health": health, "client": "1 sequential client, httpx keep-alive",
                   "server": "uvicorn 1 worker, CPU", "machine": platform.platform()}
            if args.n:
                out["retrieval_only"] = run(client, texts, False, args.n)
                print("retrieval_only", out["retrieval_only"])
            if args.n_rerank:
                out["retrieval_plus_rerank50"] = run(client, texts, True, args.n_rerank)
                print("retrieval_plus_rerank50", out["retrieval_plus_rerank50"])
    finally:
        proc.terminate()
        proc.wait(timeout=30)
    (ROOT / "results" / f"latency_{args.label}.json").write_text(json.dumps(out, indent=2) + "\n")
    pd.DataFrame(RAW).to_csv(ROOT / "results" / f"latency_raw_{args.label}.csv", index=False)


if __name__ == "__main__":
    main()
