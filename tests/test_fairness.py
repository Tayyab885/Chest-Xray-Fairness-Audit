import numpy as np
from src.fairness import subgroup_auroc

LABELS = ["Cardiomegaly"]

def test_planted_gap_recovered():
    n = 200
    yA = np.array([[i % 2] for i in range(n)])
    pA = yA * 0.9 + (1 - yA) * 0.1
    yB = np.array([[i % 2] for i in range(n)])
    pB = np.full((n, 1), 0.5)
    y = np.vstack([yA, yB]); p = np.vstack([pA, pB])
    metas = [{"sex": "M"}]*n + [{"sex": "F"}]*n
    res = subgroup_auroc(y, p, metas, LABELS, "sex")
    assert res["Cardiomegaly"]["M"] > 0.99
    assert abs(res["Cardiomegaly"]["F"] - 0.5) < 0.1
    assert res["Cardiomegaly"]["gap"] > 0.4
