import numpy as np, pandas as pd
from src.mitigate import build_group_sampler

def test_sampler_balances_minority():
    df = pd.DataFrame({"sex": ["M"]*90 + ["F"]*10})
    sampler = build_group_sampler(df, "sex")
    w = np.array(sampler.weights)
    assert abs(w[95] / w[0] - 9.0) < 1e-6   # F weight / M weight = 9
    assert len(w) == 100
