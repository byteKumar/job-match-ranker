import numpy as np
import pytest

from jobmatch.neural import ROOT, build_index


def test_flat_ip_index_returns_exact_nearest_neighbour():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(200, 384)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    index = build_index(v)
    scores, ids = index.search(v[:5], 3)
    assert ids[:, 0].tolist() == [0, 1, 2, 3, 4]          # each vector is its own best match
    np.testing.assert_allclose(scores[:, 0], 1.0, rtol=1e-5)


@pytest.mark.skipif(not (ROOT / ".hf-cache").exists(), reason="model not downloaded")
def test_biencoder_prefers_same_field():
    from jobmatch.neural import BiEncoder
    enc = BiEncoder(device="cpu")
    r, a, b = enc.encode(["Staff accountant, general ledger, reconciliations",
                          "Senior Accountant - month-end close and GL",
                          "Line cook, grill and prep station"])
    assert r @ a > r @ b
