"""Facts for the 10 worst test queries of the fine-tuned system (by nDCG@10 averaged over seeds).

Usage: .venv/bin/python scripts/error_analysis.py
For each query: its category, a shortened resume title, how many grade-2 / grade-1 jobs exist
in the corpus, and the seed-0 system's top 10 (job title, mapped category, weak grade).
The explanations in RESULTS.md are written by reading these lists (they are not generated).
Writes results/error_analysis_facts.md. Resume titles are cut at the first ';' to drop
name-like suffixes that a few dataset resumes carry.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from evaluate import rerank_scores  # noqa: E402
from jobmatch.data import ROOT  # noqa: E402
from jobmatch.evalset import load_evalset  # noqa: E402
from jobmatch.metrics import rank  # noqa: E402


def short_title(t: str) -> str:
    return t.split(";")[0].strip()[:60]


def main() -> None:
    es = load_evalset()
    ft = json.loads((ROOT / "results/metrics/finetuned.json").read_text())
    bi = json.loads((ROOT / "results/metrics/biencoder.json").read_text())
    nd = np.array([q["ndcg@10"] for q in ft["per_query"]])
    worst = np.argsort(nd, kind="stable")[:10]
    scores, _ = rerank_scores(es, str(ROOT / "models" / "ce_seed0"))

    lines = ["# Error analysis facts: 10 worst test queries (fine-tuned re-ranker)\n",
             "nDCG@10 is the mean over seeds 0/1/2; the top-10 list is from seed 0. "
             "`g` = weak grade of that job for this query.\n"]
    for qi in worst:
        q = es.queries.iloc[qi]
        g = es.grades[qi]
        order = rank(scores[qi])[:10]
        lines.append(f"\n## {q.category}: \"{short_title(q.title)}\"\n")
        lines.append(f"- nDCG@10 fine-tuned {nd[qi]:.3f} (bi-encoder alone {bi['per_query'][qi]['ndcg@10']:.3f}); "
                     f"corpus has {int((g == 2).sum())} grade-2 and {int((g == 1).sum())} grade-1 jobs")
        # Resume text is printed for reading but kept out of the committed file.
        print(f"[not saved] {q.category} skills: {q.skills[:200]!r} | summary: {q.summary[:200]!r}")
        lines.append("\n| rank | job title | job category | g |\n|---|---|---|---|")
        for r, j in enumerate(order, 1):
            job = es.corpus.iloc[j]
            lines.append(f"| {r} | {job.title[:60]} | {job.category} | {int(g[j])} |")
    (ROOT / "results" / "error_analysis_facts.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
