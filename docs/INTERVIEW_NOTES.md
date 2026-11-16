# Interview notes, milestone by milestone

What each part does, why it is built this way, the trade-offs, and one thing to try yourself.
Numbers come from `results/` (see RESULTS.md for the full table).

## M1: data and evaluation set

**What:** clean 2,481 resumes (CC0) and 29,855 LinkedIn postings (CC BY-SA 4.0), map job titles
to the resume dataset's 24 categories with ordered keyword rules, and grade every pair:
2 same category, 1 related category, 0 otherwise.

**Why a dense judged corpus:** 48 held-out resumes are scored against all 894 eval-corpus jobs
(42,912 graded pairs). If only ~10 jobs per resume were labeled, every unlabeled job a model
ranks high would count as irrelevant, which punishes models for finding good jobs nobody labeled.

**Leakage rules:** train and test never share a resume; training pairs never use eval-corpus
jobs. A test (`tests/test_labels.py`) enforces both.

**Label quality, measured:** a blind 100-pair check (labeled by an AI subagent, not a person)
agreed with the weak labels 50% exactly, kappa 0.247. Grade 0 is reliable (88%); grades 1 and 2
are noisy. Two causes: the dataset's own resume categories are noisy (a "BANKING" resume titled
Marketing Manager), and broad title rules ("Instructor" -> TEACHER for a fitness instructor).
The rules were not edited after the check: the check sample includes test pairs, so tuning the
rules on it would inflate the agreement we report.

**Bugs found:** (1) a regex left `style="...">` at the start of every resume section; (2) pay
rates in titles ("$27.61 hr") matched the HR rule; (3) 170 resumes (6.9%) use UUID section ids
the regex did not match, so their embedding text was a 60-character fragment. All three were
caught by looking at the data, not by a failing metric.

**Try it:** move the `ENGINEERING` rule to the top of `_RULES` in `src/jobmatch/categories.py`,
predict which tests in `tests/test_categories.py` fail, run `pytest`, explain, put it back.

## M2: baselines and metrics

**Metrics (fixed before any model ran):** nDCG@10 with gains 2^g - 1; MRR and Recall@50 count
only grade 2 as relevant. Ties are broken by corpus order, never by labels. A random ranking is
included so every number has a chance-level reference.

**Keyword (PepperUni-style):** share of a job's top-25 keywords (title words first, then the most
frequent requirement words) found anywhere in the resume, 0-100. **BM25:** Okapi BM25 over job
title + description with the whole resume as the query; query terms are de-duplicated because
`rank_bm25` would otherwise add a repeated word's score once per occurrence.

**Trade-off:** lexical methods need no model and are explainable (matched words), but miss
synonyms ("bookkeeping" vs "accounts payable") and over-reward long resumes that mention many words.

**Try it:** in `src/jobmatch/lexical.py`, remove the `dict.fromkeys` de-duplication in
`BM25Ranker.scores`, re-run `scripts/evaluate.py bm25`, and explain the change in nDCG@10.

## M3: embedding retrieval

**What:** all-MiniLM-L6-v2 (22M parameters, 384-dim, 256-token limit) embeds the section-aware
text; vectors are L2-normalized so FAISS inner product equals cosine similarity; `IndexFlatIP`
is exact search. At 10k x 384 floats the index is ~15 MB, so an approximate index (IVF, HNSW)
would add recall loss and tuning for no real speed need.

**Section-aware text:** the model truncates at 256 tokens; the median resume is ~5,900
characters. Putting title, skills, summary and experience first was decided before evaluation.
The raw-text ablation was measured, not used to pick the design: section-aware is slightly better
on nDCG@10 and MRR and worse on Recall@50, so the claim is "no clear win", not "big win".

**Bug found:** faiss-cpu and torch each bundle `libomp.dylib`; loading both aborts the process
(`OMP: Error #15`) or deadlocks when torch runs on CPU. It did not show on MPS. Fixed by pointing
faiss at torch's copy (`scripts/fix_libomp.sh`), verified against a numpy brute-force search.

**Try it:** change `IndexFlatIP` to `faiss.IndexHNSWFlat(dim, 32)` (with
`METRIC_INNER_PRODUCT`) in `src/jobmatch/neural.py`, re-run the bi-encoder eval, and compare the
metrics and the search time. Explain why the numbers barely move at this corpus size.

## M4: cross-encoder re-ranking and fine-tuning

**What:** the bi-encoder's top 50 per query are re-scored by a cross-encoder
(ms-marco-MiniLM-L6-v2), which reads resume and job together (512 tokens, truncating the longer
side first). Then the same model is fine-tuned on the 300 TRAIN pairs: targets grade/2,
BCE-with-logits loss, 3 epochs, lr 2e-5, batch 16, 10% warmup, seeds 0/1/2, MPS. All
hyperparameters were fixed before evaluation; no dev set, no early stopping.

**Result, reported as is:** both re-rankers are worse than the bi-encoder alone
(nDCG@10 0.319 pretrained, 0.316 ± 0.002 fine-tuned, vs 0.427). Recall@50 cannot change
because re-ranking only reorders the top 50.

**Why (diagnosed on TRAIN pairs only):** the pretrained cross-encoder learned "does this web
passage answer this short query", and barely tracks resume/job fit (Spearman 0.229 with the
grade). Fine-tuning lifts it to 0.427, which only matches the bi-encoder (0.414) even on the
data it trained on: 300 noisy pairs teach it about what the bi-encoder already knows. Inside the
top 50, where all candidates already look similar, its order is worse than the bi-encoder's.

**What I would try next (with a dev split carved from TRAIN resumes, never the test set):**
more and cleaner training pairs (the weak labels are ~50% exact-agreement), pairwise or listwise
loss on candidates the bi-encoder actually retrieves (hard negatives), and blending the
cross-encoder score with the bi-encoder score instead of replacing it.

**Interview point:** a "more powerful" model is not automatically better. The cross-encoder is
bigger per pair, slower, and domain-mismatched; measuring it is what showed that.

**Try it:** in `scripts/train_crossencoder.py`, change the targets from grade/2 to binary
(grade == 2) and predict whether nDCG@10 goes up or down before you run it. Do NOT keep the change
because the test number went up: that would be tuning on the test set. Explain why.

## M5: ranking service and latency

**What:** FastAPI `POST /rank` takes plain resume text, re-orders it into title/skills/summary/
experience (plain-text heading detection, since the API does not get the dataset's HTML), embeds
it with the bi-encoder on CPU, searches a 10k-job FAISS flat index, and returns the top k with
`matched_skills` (the job's keywords found in the resume). `rerank: true` re-scores the top 50
with the fine-tuned cross-encoder. One uvicorn worker; models load once at startup (lifespan).

**Where the time goes (quiet run):** embedding ~16 ms of ~17 ms server time; FAISS search over
10k x 384 is ~0.3 ms; HTTP + JSON ~1 ms. Latency grows with resume length because the tokenizer
reads the full text before truncating to 256 tokens. Re-ranking adds ~2 s (50 cross-encoder
passes of up to 512 tokens on CPU), and it also ranked worse in the offline evaluation, so it is
off by default.

**Measuring honestly:** the first run overlapped another project's benchmark on this shared Mac
(p99 776 ms). A diagnostic second run on a quiet machine gave p99 23 ms. Both are reported; the
second run also saves per-request timings so a tail can be explained instead of guessed.

**Trade-offs:** exact flat search is simple and needs no tuning at 10k; at ~1M+ jobs you would
switch to IVF/HNSW and measure recall loss. A single worker keeps memory low (~670 MB RSS with
both models) but serializes requests; throughput scaling would mean more workers or batching.

**Try it:** add a `max_chars` cut (say 4,000 characters) before tokenization in
`plain_resume_embed_text`, re-run `scripts/latency_test.py --n 1000 --n-rerank 0 --label try`,
and explain the change in p99. What could it cost in ranking quality, and how would you check?

## M6: error analysis and results

**What:** the 10 worst test queries for the fine-tuned system (all nDCG@10 = 0). Reading their
top-10 job titles: 9 of 10 are resumes whose dataset category does not describe the resume (a
billing associate filed under PUBLIC-RELATIONS, an insurance fraud investigator under APPAREL),
and the system returned jobs that match the resume's content. One is a real model failure: a
biotech specialist whose skills section is an alphabetical keyword list starting with "banking",
so the embedding (which reads skills first) returned bank jobs.

**How to say it in an interview:** "The metric said 0, the results looked right, so I checked the
labels: most of our worst cases were label errors. That tells me the absolute scores understate
quality and that better labels are the highest-value next step, ahead of a bigger model."

**Known limitation (seen while testing the API):** `matched_skills` misses short skills such as
"GL", "QA" or "C#" because the tokenizer drops tokens under 3 characters, and there is no
stemming ("reconciliations" vs "reconciliation").

**Try it:** make the tokenizer in `src/jobmatch/lexical.py` keep 2-character tokens when they
are all caps in the original text (GL, QA, HR), re-run the keyword baseline, and check whether
nDCG@10 moves. Then explain why this matters more for the API's highlighting than for ranking.
