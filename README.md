# job-match-ranker

Semantic resume-to-job matching and ranking. Given a resume, rank job postings by fit
(and the reverse). The point of the project is the evaluation: measuring how much
learned ranking (sentence-transformer retrieval, cross-encoder re-ranking, fine-tuning)
beats keyword matching and BM25 on a labeled resume/job evaluation set.

Status: work in progress, built milestone by milestone. Results will be in `RESULTS.md`.

## Data (not committed)

- Resumes: [Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)
  by Snehaan Bhawal, CC0 (Public Domain). Downloaded from the Hugging Face mirror
  `opensporks/resumes`.
- Jobs: [LinkedIn Job Postings (2023-2024)](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)
  by Arsh Koneru, CC BY-SA 4.0. Downloaded from the Hugging Face mirror
  `xanderios/linkedin-job-postings` (2023 snapshot, `job_postings.csv`).

Datasets, model weights and indexes stay out of git.
