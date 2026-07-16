import numpy as np
from src.evaluate import compute_auroc

LABELS = [f"L{i}" for i in range(14)]

def test_perfect_separation_auroc_1():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=(100, 14))
    prob = y * 0.9 + (1 - y) * 0.1  # perfectly separable
    res = compute_auroc(y, prob, LABELS)
    assert abs(res["macro"] - 1.0) < 1e-6
    assert abs(res["L0"] - 1.0) < 1e-6
