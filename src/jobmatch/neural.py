"""Bi-encoder retrieval (sentence-transformers + FAISS) and cross-encoder re-ranking."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# Keep downloaded models inside the project (git-ignored) so cleanup is one folder.
os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))

BI_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L6-v2"


def best_device() -> str:
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


class BiEncoder:
    def __init__(self, device: str | None = None, model_name: str = BI_ENCODER):
        from sentence_transformers import SentenceTransformer
        self.device = device or best_device()
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        # normalize -> inner product == cosine similarity
        return self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False).astype(np.float32)


def build_index(job_vectors: np.ndarray):
    """Exact inner-product index. At 10k x 384 a flat index is ~15 MB and needs no training."""
    import faiss
    index = faiss.IndexFlatIP(job_vectors.shape[1])
    index.add(job_vectors)
    return index


class CrossEncoderReranker:
    def __init__(self, model_name_or_path: str = CROSS_ENCODER, device: str | None = None,
                 max_length: int = 512):
        from sentence_transformers import CrossEncoder
        self.device = device or best_device()
        self.model = CrossEncoder(model_name_or_path, device=self.device, max_length=max_length)

    def score(self, resume_text: str, job_texts: list[str], batch_size: int = 32) -> np.ndarray:
        pairs = [(resume_text, j) for j in job_texts]
        return np.asarray(self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False),
                          dtype=np.float32)
