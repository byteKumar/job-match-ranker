# Results

Every number here was measured once on this machine, on 2026-11-16, unless the text says
otherwise. Raw outputs are in `results/` (metrics per model and per query, training logs,
latency per request). Nothing was tuned on the test queries.

## Results template (spec section 6)

```
MACHINE: chip=Apple M3 (4 performance + 4 efficiency cores) RAM=16 GB, Python=3.12.10, PyTorch=2.15.0
DATA: jobs=29,855 (894 in eval corpus, 10,000 in service index) resumes=2,481 labeled pairs=500 (train/test 300/200) + dense test grid 42,912 ; hand-check agreement=50% exact (AI-labeled, not human; Cohen's kappa 0.247; 63% relevant-vs-not)
model                 | nDCG@10       | MRR           | Recall@50
keyword (PepperUni)   | 0.253         | 0.366         | 0.177
BM25                  | 0.277         | 0.369         | 0.200
bi-encoder            | 0.427         | 0.543         | 0.280
bi-encoder + rerank   | 0.319         | 0.365         | 0.280
fine-tuned (± std)    | 0.316 ± 0.002 | 0.378 ± 0.004 | 0.280 ± 0.000
SERVICE at 10k jobs: p50/p99=17.7/23.1 ms quiet run, 31.3/776.3 ms first run (shared machine) ; retrieval only, CPU, 1 sequential client ; index build=45.4 s (embedding on MPS; FAISS add+write 0.009 s; index 15.4 MB)
ANYTHING THAT FAILED OR SURPRISED YOU: re-ranking with a cross-encoder made ranking worse (nDCG@10 0.427 -> 0.319) and fine-tuning on 300 weak pairs did not fix it (0.316); weak labels are noisy (kappa 0.247) and 9 of the 10 worst queries are resumes whose dataset category does not match their content; faiss-cpu + torch crash together on macOS (duplicate libomp); first latency run was disturbed by another benchmark
REPO pushed? yes, https://github.com/byteKumar/job-match-ranker
```

## Ranking quality

48 held-out test resumes, each ranking the full 894-job eval corpus. nDCG@10 uses graded gains
(2^g - 1); MRR and Recall@50 count grade 2 as relevant. Devices: bi-encoder and cross-encoders
ran on MPS for evaluation.

| model | nDCG@10 | MRR | Recall@50 | notes |
|---|---|---|---|---|
| random | 0.058 | 0.122 | 0.053 | chance level, seed 0 |
| keyword (PepperUni-style) | 0.253 | 0.366 | 0.177 | share of the job's top-25 keywords found in the resume |
| BM25 | 0.277 | 0.369 | 0.200 | Okapi BM25, resume as query, de-duplicated terms |
| bi-encoder, raw text (ablation) | 0.419 | 0.539 | 0.313 | all-MiniLM-L6-v2, full text truncated at 256 tokens |
| **bi-encoder** | **0.427** | **0.543** | **0.280** | section-aware text; best model |
| bi-encoder + pretrained re-rank | 0.319 | 0.365 | 0.280 | ms-marco-MiniLM-L6-v2 on top 50 |
| bi-encoder + fine-tuned re-rank | 0.316 ± 0.002 | 0.378 ± 0.004 | 0.280 ± 0.000 | 3 seeds, sample std |

What the table says:

- Embedding retrieval beats keyword matching by +0.174 nDCG@10 (0.253 -> 0.427, 1.69x) and
  BM25 by +0.150. That is the main claim this project supports.
- Section-aware text vs raw text is a wash: +0.008 nDCG@10 and +0.004 MRR, but -0.033 Recall@50.
  The design was fixed before evaluation and kept; the raw variant is reported as an ablation.
- Both re-rankers are worse than the bi-encoder alone. Recall@50 is identical by construction,
  because re-ranking only reorders the top 50.
- Fine-tuning moved the cross-encoder by -0.002 nDCG@10 and +0.014 MRR: no real change. Seed
  spread is small (std 0.002), so this is not noise from one unlucky run.

### Why the re-rankers lose (checked on TRAIN pairs only)

Spearman correlation between each model's score and the weak grade, on the 300 training pairs
(`results/training/insample_diagnostic.json`):

| model | Spearman with grade | mean score for grade 0 / 1 / 2 |
|---|---|---|
| bi-encoder cosine | 0.414 | 0.280 / 0.360 / 0.408 |
| pretrained cross-encoder | 0.229 | -6.90 / -6.07 / -5.61 |
| fine-tuned cross-encoder (seed 0) | 0.427 | -0.48 / 0.03 / 0.52 |

The pretrained cross-encoder was trained on web search (short query -> passage) and tracks
resume/job fit poorly. Fine-tuning works in the sense that it learns the grades, but even on its
own training data it only reaches the bi-encoder's level, so it has nothing to add on the 50
candidates the bi-encoder already picked. Training: 3 epochs, lr 2e-5, batch 16, MPS;
train time 272.5 s / 44.1 s / 61.8 s for seeds 0 / 1 / 2 (seed 0 shared the machine with other
jobs); final train loss 0.654 / 0.643 / 0.656.

## Label quality (the biggest caveat)

The 100-pair check was labeled by **Claude (an AI subagent)**, blind to the weak labels and the
title rules. It is **not** a human hand-check. Sample stratified by weak grade (34/33/33).

| | value |
|---|---|
| exact agreement | 50.0% |
| relevant (grade >= 1) vs not | 63.0% |
| Cohen's kappa | 0.247 |
| agreement when weak grade = 0 / 1 / 2 | 88.2% / 24.2% / 36.4% |

Confusion (rows = weak grade 0/1/2, columns = checker 0/1/2): `[[30, 2, 2], [24, 8, 1], [9, 12, 12]]`.

Grade 0 is trustworthy; grades 1 and 2 are noisy. Two sources: the resume dataset's own
categories are noisy (a "BANKING" resume titled Marketing Manager), and some title rules are
too broad ("Instructor" -> TEACHER for a fitness instructor). The rules were not changed after
the check, because the check sample includes test pairs.

What this means for the table: absolute scores are understated (see error analysis), and the
ranking of systems could change with clean labels. The bi-encoder's lead over keyword/BM25 is
large enough to be unlikely to flip; the small gaps (section-aware vs raw, the two re-rankers)
are within what label noise could move.

## Error analysis: the 10 worst test queries

By nDCG@10 of the fine-tuned system (mean over seeds); all 10 score 0.000, and the bi-encoder
alone also scores 0.000 on 9 of them (0.073 on the last). Facts per query, including the top-10
job titles: `results/error_analysis_facts.md`. Explanations below come from reading those
lists and the resume text; they are judgments from job titles, not a re-labeling.

| # | dataset category | resume (short title) | what the system returned | explanation |
|---|---|---|---|---|
| 1 | INFORMATION-TECHNOLOGY | Administrative assistant / HR manager | HR coordinator and HR manager jobs | Label error: the resume is admin/HR work; only the category says IT. Results match the content. |
| 2 | ADVOCATE | Senior customer service advocate | Customer service rep jobs | Category name clash: ADVOCATE grade-2 jobs are attorneys; this "advocate" is customer service. Results match the content. |
| 3 | ADVOCATE | Health coach | Fitness, health and food service jobs | Label error (health coach, not legal advocate). Fitness results are sensible; food-service ones are not. |
| 4 | HEALTHCARE | Recruiter | Recruitment consultant, HR jobs | Label error, plus a title-rule gap: "Recruitment Consultant" maps to CONSULTANT, not HR (rule has "recruiter", not "recruitment"). |
| 5 | FITNESS | Vice president (manufacturing operations) | VP of business development, supply chain consulting | Label error: an operations executive filed under FITNESS; only 14 fitness jobs exist. |
| 6 | AGRICULTURE | Project assistant | Project manager jobs (IT, construction) | Generic resume; nothing agricultural in the parts the model reads. Results match the role, not the sector. |
| 7 | AGRICULTURE | Senior field application specialist (biotech diagnostics) | Bank teller and banker jobs | **Model error.** The skills section is an alphabetical keyword list starting with "banking"; section-aware text puts skills first, so the embedding latches onto it. |
| 8 | APPAREL | Special investigative unit (insurance fraud) | Claims, insurance defense, legal jobs | Label error: an insurance investigator filed under APPAREL. Results match the content. |
| 9 | PUBLIC-RELATIONS | Billing services associate | Accounts payable and staff accountant jobs | Label error: billing/AP work filed under PUBLIC-RELATIONS. Results match the content. |
| 10 | BANKING | Project manager, technical implementation (AS/400) | IT project manager and consulting jobs | Label error-ish: an IT project manager who likely worked at a bank. Results match the job function. |

Takeaways: 9 of 10 failures are label problems, not ranking problems; one is a real model failure
caused by section-aware ordering putting a keyword-soup skills list first. Fixes worth trying
(on a dev split): relabel resume categories from content, add "recruitment" to the HR rule, and
cap or de-prioritize alphabetical skill lists in the embedding text.

## Service (FastAPI, 10,000 jobs)

Index: 10,000 jobs sampled from the cleaned set (seed 42), all-MiniLM-L6-v2 embeddings, FAISS
`IndexFlatIP` (exact). Build: 45.43 s total = 45.42 s embedding on MPS + 0.009 s FAISS add and
write; index file 15.36 MB, job metadata 22.67 MB.

Latency: uvicorn with 1 worker on CPU (4 torch threads), one sequential httpx client, 20 warm-up
requests, each request a different resume, client-side wall time.

| run | mode | requests | p50 | p99 | mean | max |
|---|---|---|---|---|---|---|
| 1 (10:12-10:26, overlapped another project's benchmark) | retrieval only | 1,000 | 31.3 ms | 776.3 ms | 80.7 ms | 2,605.6 ms |
| 1 | retrieval + re-rank top 50 | 300 | 1,947.4 ms | 7,071.1 ms | 2,376.2 ms | 8,587.2 ms |
| 2 (quiet machine, per-request timings saved) | retrieval only | 1,000 | 17.7 ms | 23.1 ms | 18.0 ms | 58.1 ms |
| 2 (load rose to 7.2 during the run) | retrieval + re-rank top 50 | 300 | 1,613.1 ms | 2,726.0 ms | 1,679.0 ms | 6,151.6 ms |

Run 2 was a diagnostic re-run after run 1's p99 looked wrong; both are reported. In run 2,
embedding is ~95% of server time (median 16.0 ms), FAISS search is 0.27 ms, and HTTP + JSON adds
~1 ms. Latency correlates with resume length (Spearman 0.57): the tokenizer reads the full text
before truncating. Run 1 did not save per-request timings, so its tail cannot be broken down;
another project's notes show a benchmark in that window.

## Not done or not verified

- The label check was done by an AI, not a person; a human check of the same 100 pairs
  (`scripts/handcheck.py`) would make the agreement number a real "hand-check".
- Every metric is from one evaluation pass (the fine-tuned row is 3 training seeds). No
  confidence intervals over queries; with 48 queries, small gaps are not significant.
- Latency was measured with one sequential client only (no concurrency/throughput test).
- Fine-tuned checkpoints for seeds 1 and 2 were deleted after evaluation to save disk; re-create
  them with `scripts/train_crossencoder.py --seed 1` / `--seed 2`.
- No MLflow; experiment tracking is this table plus JSON files in `results/`.
- GitHub Actions is disabled on the account (billing lock), so no CI ran; tests were run locally
  (36 passed).
