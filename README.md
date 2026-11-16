# job-match-ranker

[![tests](https://github.com/byteKumar/job-match-ranker/actions/workflows/tests.yml/badge.svg)](https://github.com/byteKumar/job-match-ranker/actions/workflows/tests.yml)

Semantic resume-to-job ranking with an honest evaluation. Given a resume, rank job postings by
fit. The project measures how much learned ranking beats keyword matching: keyword overlap
(PepperUni-style) and BM25 baselines vs a sentence-transformer bi-encoder with FAISS, a
cross-encoder re-ranker, and a fine-tuned re-ranker, on 48 held-out resumes x 894 graded jobs.
A FastAPI service ranks 10,000 jobs per request.

## Results

Full write-up, caveats and error analysis: [RESULTS.md](RESULTS.md).

| model | nDCG@10 | MRR | Recall@50 |
|---|---|---|---|
| random | 0.058 | 0.122 | 0.053 |
| keyword (PepperUni-style) | 0.253 | 0.366 | 0.177 |
| BM25 | 0.277 | 0.369 | 0.200 |
| **bi-encoder (all-MiniLM-L6-v2 + FAISS)** | **0.427** | **0.543** | **0.280** |
| bi-encoder + cross-encoder re-rank | 0.319 | 0.365 | 0.280 |
| bi-encoder + fine-tuned re-rank (3 seeds) | 0.316 ± 0.002 | 0.378 ± 0.004 | 0.280 ± 0.000 |

- Embedding retrieval raises nDCG@10 from 0.253 (keyword) to 0.427.
- Re-ranking with a cross-encoder hurt, and fine-tuning it on 300 weakly labeled pairs did not
  help; RESULTS.md explains why (checked on training pairs only).
- Labels are weak (category-based). A blind 100-pair check by an AI (not a person) agreed 50%
  exactly (Cohen's kappa 0.247). 9 of the 10 worst queries are resumes whose dataset category
  does not match their content.
- Service at 10k jobs (CPU, 1 client): p50/p99 17.7/23.1 ms on a quiet machine; 31.3/776.3 ms in
  a first run that overlapped another benchmark. Index build 45.4 s (MPS), index 15.4 MB.

Machine: Apple M3 (16 GB), macOS 26.6, Python 3.12.10, PyTorch 2.15.0, sentence-transformers
6.3.0, faiss-cpu 1.16.0, FastAPI 0.163.1.

## How it works

1. **Data** (`src/jobmatch/data.py`, `clean.py`): parse resume HTML into sections; clean and
   de-duplicate postings; build "section-aware" text (title, skills, summary, experience first)
   because the embedding model reads only 256 tokens.
2. **Labels** (`categories.py`, `labels.py`): map job titles to the resume dataset's 24 categories
   with ordered rules; grade pairs 2 (same) / 1 (related) / 0. 48 test resumes x 894 eval jobs are
   all graded (no unjudged jobs); 300 train pairs use other resumes and other jobs.
3. **Baselines** (`lexical.py`): keyword share 0-100 and BM25 (`rank_bm25`).
4. **Retrieval** (`neural.py`): all-MiniLM-L6-v2 embeddings, normalized, FAISS `IndexFlatIP`.
5. **Re-ranking** (`neural.py`, `scripts/train_crossencoder.py`): ms-marco-MiniLM-L6-v2 on the top
   50, then fine-tuned in a plain PyTorch loop (fixed hyperparameters, 3 seeds).
6. **Metrics** (`metrics.py`): nDCG@10 (graded), MRR and Recall@50 (grade 2 = relevant).
7. **Service** (`service.py`): `POST /rank` returns the top k jobs with matched skills.

## Run it

```bash
python3.12 -m venv .venv
.venv/bin/pip install --no-cache-dir -r requirements.txt
scripts/fix_libomp.sh                     # macOS: faiss-cpu and torch each bundle libomp

# data (not committed): ~56 MB + ~135 MB
mkdir -p data/raw/resumes data/raw/jobs
curl -L -o data/raw/resumes/Resume.csv https://huggingface.co/datasets/opensporks/resumes/resolve/main/Resume/Resume.csv
curl -L -o data/raw/jobs/job_postings.csv https://huggingface.co/datasets/xanderios/linkedin-job-postings/resolve/main/job_postings.csv

.venv/bin/python scripts/prepare_data.py                   # clean data, build the labeled sets
.venv/bin/python scripts/evaluate.py random keyword bm25 biencoder_raw biencoder biencoder_rerank
for s in 0 1 2; do .venv/bin/python scripts/train_crossencoder.py --seed $s; done
.venv/bin/python scripts/evaluate.py finetuned
.venv/bin/python scripts/error_analysis.py
.venv/bin/python scripts/build_index.py                    # 10k-job FAISS index
.venv/bin/python scripts/latency_test.py --label myrun     # starts uvicorn on :8010
.venv/bin/python -m pytest -q
```

Models (~91 MB each) download into `.hf-cache/` inside the project. Serve and query:

```bash
.venv/bin/uvicorn jobmatch.service:app --app-dir src --port 8010
curl -s localhost:8010/rank -H 'content-type: application/json' \
  -d '{"resume_text": "Staff Accountant\nSkills\nGL, reconciliations, Excel", "k": 5}'
```

`fix_libomp.sh` points faiss at torch's OpenMP runtime so only one copy loads. Without it, any
process that runs torch on the CPU and imports faiss dies with `OMP: Error #15` (or hangs).
`KMP_DUPLICATE_LIB_OK=TRUE` is not used: OpenMP documents it as unsafe. On Linux, faiss and
torch load together without the fix (CI checks this).

CI (`.github/workflows/tests.yml`, ubuntu-latest, CPU-only torch) downloads the data and the
bi-encoder, rebuilds the labeled sets, checks they match the committed `dataset_stats.json`,
and runs all 36 tests. `JOBMATCH_STRICT_TESTS=1` makes any skipped test fail. Locally, the data
and model tests skip until `prepare_data.py` has run and the model is downloaded. CI does not
re-run the evaluations or benchmarks.

## Data and licenses (data, weights and indexes are not committed)

- Resumes: [Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) by
  Snehaan Bhawal, CC0 (Public Domain), via the Hugging Face mirror `opensporks/resumes`. Resumes
  are anonymized by the dataset, but a few still contain names or profile URLs, so no resume
  text is committed; committed results contain ids, categories and short titles only.
- Jobs: [LinkedIn Job Postings (2023-2024)](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)
  by Arsh Koneru, CC BY-SA 4.0, via the Hugging Face mirror `xanderios/linkedin-job-postings`
  (2023 snapshot, `job_postings.csv`).
- Models: `sentence-transformers/all-MiniLM-L6-v2` and `cross-encoder/ms-marco-MiniLM-L6-v2`,
  both Apache-2.0.

Interview notes per milestone, with one exercise each: [docs/INTERVIEW_NOTES.md](docs/INTERVIEW_NOTES.md).
