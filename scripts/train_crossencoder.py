"""Fine-tune the ms-marco cross-encoder on the 300 TRAIN pairs (weak grades), one seed per run.

Usage: .venv/bin/python scripts/train_crossencoder.py --seed 0
Saves models/ce_seed<seed>/ (git-ignored) and results/training/train_seed<seed>.json.

Hyperparameters were fixed before any fine-tuned model was evaluated and are not tuned:
there is no dev set and no early stopping; the model after the last epoch is the result.
The test queries and eval-corpus jobs never appear in the training pairs (see tests/test_labels.py).
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from jobmatch.data import PROCESSED, ROOT  # noqa: E402
from jobmatch.neural import CROSS_ENCODER, best_device  # noqa: E402  (also sets HF_HOME)

EPOCHS = 3
BATCH_SIZE = 16
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_FRAC = 0.1
MAX_LENGTH = 512


def load_train_pairs() -> tuple[list[tuple[str, str]], list[float]]:
    core = pd.read_parquet(PROCESSED / "core_pairs.parquet")
    train = core[core.split == "train"]
    resumes = pd.read_parquet(PROCESSED / "resumes.parquet").set_index("resume_id")
    jobs = pd.read_parquet(PROCESSED / "jobs.parquet").set_index("job_id")
    pairs = [(resumes.at[r, "embed_text"], jobs.at[j, "embed_text"]) for r, j in zip(train.resume_id, train.job_id)]
    return pairs, (train.grade.astype(float) / 2.0).tolist()   # 0 / 0.5 / 1 targets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = best_device()

    tok = AutoTokenizer.from_pretrained(CROSS_ENCODER)
    model = AutoModelForSequenceClassification.from_pretrained(CROSS_ENCODER).to(device)
    pairs, labels = load_train_pairs()
    steps_per_epoch = (len(pairs) + BATCH_SIZE - 1) // BATCH_SIZE
    total = steps_per_epoch * EPOCHS
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP_FRAC * total), total)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(args.seed)

    log, t0 = [], time.perf_counter()
    model.train()
    for epoch in range(EPOCHS):
        order = torch.randperm(len(pairs), generator=gen).tolist()
        losses = []
        for b in range(0, len(order), BATCH_SIZE):
            idx = order[b:b + BATCH_SIZE]
            enc = tok([pairs[i][0] for i in idx], [pairs[i][1] for i in idx], truncation="longest_first",
                      max_length=MAX_LENGTH, padding=True, return_tensors="pt").to(device)
            y = torch.tensor([labels[i] for i in idx], dtype=torch.float32, device=device)
            logits = model(**enc).logits.squeeze(-1)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            losses.append(loss.item())
        log.append({"epoch": epoch + 1, "mean_train_loss": round(float(np.mean(losses)), 4)})
        print(f"seed {args.seed} epoch {epoch + 1}: loss {log[-1]['mean_train_loss']}")

    out = ROOT / "models" / f"ce_seed{args.seed}"
    model.save_pretrained(out)
    tok.save_pretrained(out)
    info = {"seed": args.seed, "device": device, "train_pairs": len(pairs), "epochs": EPOCHS,
            "batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY, "warmup_frac": WARMUP_FRAC,
            "max_length": MAX_LENGTH, "base_model": CROSS_ENCODER, "torch": torch.__version__,
            "train_seconds": round(time.perf_counter() - t0, 1), "log": log}
    (ROOT / "results" / "training" / f"train_seed{args.seed}.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({k: info[k] for k in ("seed", "device", "train_seconds")}))


if __name__ == "__main__":
    main()
