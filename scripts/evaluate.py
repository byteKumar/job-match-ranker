"""Evaluate one ranking model on the 48 held-out test queries x eval corpus.

Usage: .venv/bin/python scripts/evaluate.py keyword|bm25 [...]
Writes results/metrics/<model>.json (mean + per-query metrics) and refreshes
results/results_table.md.
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jobmatch.data import ROOT  # noqa: E402
from jobmatch.evalset import load_evalset  # noqa: E402
from jobmatch.metrics import evaluate  # noqa: E402

OUT = ROOT / "results" / "metrics"


def run_keyword(es):
    from jobmatch.lexical import KeywordRanker
    return KeywordRanker(es.corpus).scores(es.queries.full.tolist()), {"query_text": "resume full text"}


def run_bm25(es):
    from jobmatch.lexical import BM25Ranker
    return BM25Ranker(es.corpus).scores(es.queries.full.tolist()), {"query_text": "resume full text"}


def _biencoder_scores(es, field_resume: str, field_job: str):
    import faiss
    from jobmatch.neural import BiEncoder, build_index
    enc = BiEncoder()
    t0 = time.perf_counter()
    jv = enc.encode(es.corpus[field_job].tolist())
    t_jobs = time.perf_counter() - t0
    qv = enc.encode(es.queries[field_resume].tolist())
    index = build_index(jv)
    scores, ids = index.search(qv, len(es.corpus))       # full ranking for evaluation
    full = np.zeros_like(scores)
    np.put_along_axis(full, ids, scores, axis=1)
    info = {"device": enc.device, "model": "all-MiniLM-L6-v2", "max_seq_length": enc.model.max_seq_length,
            "encode_corpus_seconds": round(t_jobs, 2), "faiss": faiss.__version__}
    return full, info


def run_biencoder(es):
    s, info = _biencoder_scores(es, "embed_text", "embed_text")
    return s, {**info, "note": "section-aware text (title, skills, summary, experience)"}


def run_biencoder_raw(es):
    s, info = _biencoder_scores(es, "full", "full")
    return s, {**info, "note": "ablation: raw full text, truncated at 256 tokens"}


RERANK_K = 50


def rerank_scores(es, reranker_path: str):
    """Bi-encoder top-50 per query, re-ordered by a cross-encoder; the rest keep bi-encoder order."""
    from jobmatch.neural import CrossEncoderReranker
    bi, info = _biencoder_scores(es, "embed_text", "embed_text")
    ce = CrossEncoderReranker(reranker_path)
    out = bi.copy()
    t0 = time.perf_counter()
    for qi, q in enumerate(es.queries.itertuples()):
        top = np.argsort(-bi[qi], kind="stable")[:RERANK_K]
        out[qi, top] = 1000.0 + ce.score(q.embed_text, es.corpus.embed_text.iloc[top].tolist())
    info.update(reranker=reranker_path, rerank_k=RERANK_K, rerank_device=ce.device,
                rerank_seconds=round(time.perf_counter() - t0, 2))
    return out, info


def run_biencoder_rerank(es):
    from jobmatch.neural import CROSS_ENCODER
    s, info = rerank_scores(es, CROSS_ENCODER)
    return s, {**info, "note": "pretrained ms-marco-MiniLM-L6-v2 cross-encoder on bi-encoder top 50"}


SEEDS = (0, 1, 2)


def run_finetuned(es):
    """Fine-tuned cross-encoder, one run per seed; reported as mean +/- sample std (ddof=1)."""
    runs, info = [], {}
    for seed in SEEDS:
        path = str(ROOT / "models" / f"ce_seed{seed}")
        scores, info = rerank_scores(es, path)
        runs.append(evaluate(scores, es.grades))
    keys = ("ndcg@10", "mrr", "recall@50")
    per_seed = [r["mean"] for r in runs]
    info.update(
        per_seed={str(s): m for s, m in zip(SEEDS, per_seed)},
        std={k: float(np.std([m[k] for m in per_seed], ddof=1)) for k in keys},
        note=f"fine-tuned on 300 train pairs; mean ± std over seeds {list(SEEDS)}",
        reranker="models/ce_seed{0,1,2}",
    )
    # Per-query metrics averaged over the seeds (used for error analysis).
    pq = [{k: float(np.mean([r["per_query"][i][k] for r in runs])) for k in keys} for i in range(len(es.queries))]
    return {"_precomputed": {"mean": {k: float(np.mean([m[k] for m in per_seed])) for k in keys},
                             "per_query": pq}}, info


def run_random(es):
    import numpy as np
    rng = np.random.default_rng(0)
    return rng.random(es.grades.shape), {"note": "chance level, seed 0"}


MODELS = {"random": run_random, "keyword": run_keyword, "bm25": run_bm25,
          "biencoder": run_biencoder, "biencoder_raw": run_biencoder_raw,
          "biencoder_rerank": run_biencoder_rerank, "finetuned": run_finetuned}


def write_table() -> None:
    order = ["random", "keyword", "bm25", "biencoder_raw", "biencoder", "biencoder_rerank", "finetuned"]
    rows = []
    for name in order + sorted(p.stem for p in OUT.glob("*.json") if p.stem not in order):
        p = OUT / f"{name}.json"
        if not p.exists():
            continue
        m = json.loads(p.read_text())
        mean, std = m["mean"], m.get("std")
        fmt = (lambda k: f"{mean[k]:.3f} ± {std[k]:.3f}") if std else (lambda k: f"{mean[k]:.3f}")
        rows.append(f"| {name} | {fmt('ndcg@10')} | {fmt('mrr')} | {fmt('recall@50')} | {m.get('note', '')} |")
    table = ("| model | nDCG@10 | MRR | Recall@50 | notes |\n|---|---|---|---|---|\n" + "\n".join(rows) + "\n")
    (ROOT / "results" / "results_table.md").write_text(
        "<!-- generated by scripts/evaluate.py; 48 test queries x eval corpus -->\n" + table)
    print(table)


def save(name: str, res: dict, es, extra: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for q, rid, cat in zip(res["per_query"], es.queries.resume_id, es.queries.category):
        q["resume_id"], q["category"] = rid, cat
    res.update(extra, n_queries=len(es.queries), n_corpus=len(es.corpus), python=platform.python_version())
    (OUT / f"{name}.json").write_text(json.dumps(res, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", choices=sorted(MODELS))
    args = ap.parse_args()
    es = load_evalset()
    for name in args.models:
        t0 = time.perf_counter()
        scores, extra = MODELS[name](es)
        res = scores["_precomputed"] if isinstance(scores, dict) else evaluate(scores, es.grades)
        extra["seconds"] = round(time.perf_counter() - t0, 2)
        save(name, res, es, extra)
        print(name, {k: round(v, 4) for k, v in res["mean"].items()}, f"{extra['seconds']}s")
    write_table()


if __name__ == "__main__":
    main()
