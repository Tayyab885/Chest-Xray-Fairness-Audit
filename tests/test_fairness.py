import numpy as np
from src.fairness import subgroup_auroc, equalized_odds, youden_threshold

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

def test_equalized_odds_planted_tpr_gap():
    import numpy as np
    from src.fairness import equalized_odds
    # group M: threshold 0.5 -> perfect (all correct); group F: model always predicts 0 -> TPR 0
    n = 100
    yM = np.array([[i % 2] for i in range(n)])
    pM = yM * 0.9 + (1 - yM) * 0.1
    yF = np.array([[i % 2] for i in range(n)])
    pF = np.full((n, 1), 0.1)            # always below 0.5 -> predicts negative
    y = np.vstack([yM, yF]); p = np.vstack([pM, pF])
    metas = [{"sex": "M"}]*n + [{"sex": "F"}]*n
    res = equalized_odds(y, p, metas, label_idx=0, group_key="sex", threshold=0.5)
    assert res["M"]["tpr"] > 0.99
    assert res["F"]["tpr"] < 0.01        # F never flags a true positive
    assert res["tpr_gap"] > 0.9

def test_subgroup_auroc_column_alignment():
    import numpy as np
    from src.fairness import subgroup_auroc
    full = ["A", "B", "C", "D"]
    n = 100
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=(n, 4))
    p = rng.random((n, 4))
    # make column index 1 ("B") perfectly predictive
    p[:, 1] = y[:, 1] * 0.9 + (1 - y[:, 1]) * 0.1
    metas = [{"sex": "M" if i % 2 else "F"} for i in range(n)]
    res = subgroup_auroc(y, p, metas, full, "sex")
    # B is perfectly predictive within each subgroup -> both group AUROCs ~1.0
    assert res["B"]["M"] > 0.99 and res["B"]["F"] > 0.99

def test_youden_threshold_separable():
    import numpy as np
    from src.fairness import youden_threshold
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    thr = youden_threshold(y, p)
    # a threshold between 0.2 and 0.8 perfectly separates -> should sit in that range
    assert 0.2 < thr <= 0.8
