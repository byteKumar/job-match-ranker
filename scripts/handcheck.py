"""Blind hand-check of the weak labels.

Label:   .venv/bin/python scripts/handcheck.py          (resumable; press q to stop)
Report:  .venv/bin/python scripts/handcheck.py --report

You see a resume and a job and give a grade without seeing the weak label. Your answers
go to results/handcheck_labels.csv (pair ids and your grade only, no resume text). The
report compares them with the weak labels: raw agreement, per-grade agreement, Cohen's
kappa and a confusion matrix, written to results/handcheck_agreement.json.
"""

import argparse
import csv
import json
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from jobmatch.data import PROCESSED, ROOT  # noqa: E402

LABELS = ROOT / "results" / "handcheck_labels.csv"
REPORT = ROOT / "results" / "handcheck_agreement.json"
# Who produced results/handcheck_labels.csv. The committed labels were made by an AI, not a person.
LABELED_BY = ("Claude (AI subagent), blind to weak labels and to the title rules; "
              "not a human hand-check")

RUBRIC = """\
Grade the fit of this PERSON for this JOB, judging the field / occupation, not seniority:
  2 = same occupation: this person plausibly applies for and does this job
  1 = related field: transferable background, a stretch but not absurd
  0 = unrelated: different line of work
"""


def wrap(label: str, text: str, limit: int) -> str:
    text = text if len(text) <= limit else text[:limit] + " ..."
    return f"{label}: " + textwrap.fill(text, 100, subsequent_indent="    ") if text else ""


def show(i: int, n: int, r: pd.Series, j: pd.Series) -> None:
    print("\n" + "=" * 100)
    print(f"Pair {i}/{n}")
    print("-" * 40 + " RESUME " + "-" * 52)
    for lab, txt, lim in (("Title", r.title, 120), ("Summary", r.summary, 350),
                          ("Skills", r.skills, 450), ("Experience", r.experience, 700)):
        if txt:
            print(wrap(lab, txt, lim))
    print("-" * 40 + " JOB " + "-" * 55)
    print(wrap("Title", j.title, 150))
    print(wrap("Description", j.description, 900))
    if j.requirements:
        print(wrap("Requirements", j.requirements, 500))
    print("-" * 100)


def label() -> None:
    sample = pd.read_parquet(PROCESSED / "handcheck_sample.parquet")
    resumes = pd.read_parquet(PROCESSED / "resumes.parquet").set_index("resume_id")
    jobs = pd.read_parquet(PROCESSED / "jobs.parquet").set_index("job_id")
    done = set()
    if LABELS.exists():
        done = {int(row["pair_id"]) for row in csv.DictReader(LABELS.open())}
    LABELS.parent.mkdir(exist_ok=True)
    new_file = not LABELS.exists()
    print(RUBRIC)
    print(f"{len(done)} of {len(sample)} already labeled.")
    with LABELS.open("a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["pair_id", "resume_id", "job_id", "human_grade"])
        for row in sample.itertuples():
            if row.pair_id in done:
                continue
            show(row.pair_id, len(sample), resumes.loc[row.resume_id], jobs.loc[row.job_id])
            while True:
                ans = input("Grade 0/1/2 (r = show rubric, q = quit): ").strip().lower()
                if ans in {"0", "1", "2"}:
                    w.writerow([row.pair_id, row.resume_id, row.job_id, ans])
                    f.flush()
                    break
                if ans == "r":
                    print(RUBRIC)
                if ans == "q":
                    print("Saved. Run again to continue.")
                    return
    print("All pairs labeled. Run with --report.")


def cohen_kappa(a: list[int], b: list[int], k: int = 3) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in range(k))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def report() -> None:
    sample = pd.read_parquet(PROCESSED / "handcheck_sample.parquet")
    human = pd.read_csv(LABELS)
    m = sample.merge(human[["pair_id", "human_grade"]], on="pair_id")
    weak, hum = m["grade"].astype(int).tolist(), m["human_grade"].astype(int).tolist()
    conf = pd.crosstab(m["grade"], m["human_grade"]).reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0)
    # "Relevant vs not" view: grade >= 1 counts as relevant.
    bin_agree = sum((w >= 1) == (h >= 1) for w, h in zip(weak, hum)) / len(m)
    out = {
        "pairs_checked": len(m),
        "exact_agreement_pct": round(100 * sum(w == h for w, h in zip(weak, hum)) / len(m), 1),
        "binary_agreement_pct_relevant_vs_not": round(100 * bin_agree, 1),
        "cohen_kappa": round(cohen_kappa(weak, hum), 3),
        "agreement_by_weak_grade_pct": {
            int(g): round(100 * (sub["grade"] == sub["human_grade"]).mean(), 1)
            for g, sub in m.groupby("grade")
        },
        "confusion_rows_weak_cols_human": conf.values.tolist(),
        "by_split": {s: round(100 * (sub["grade"] == sub["human_grade"]).mean(), 1)
                     for s, sub in m.groupby("split")},
        "labeled_by": LABELED_BY,
        "note": "Sample is stratified by weak grade (34/33/33), drawn at random within each grade.",
    }
    REPORT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    report() if args.report else label()
