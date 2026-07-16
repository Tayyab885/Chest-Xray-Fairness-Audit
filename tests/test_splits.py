import pandas as pd
from src.data import make_splits

def _fake_df():
    rows = []
    for pid in range(10):
        for k in range(3):
            rows.append({
                "Image Index": f"{pid:04d}_{k}.png",
                "Patient ID": pid,
                "age_bin": "40-60", "sex": "M",
            })
    return pd.DataFrame(rows)

def test_no_patient_overlap():
    df = _fake_df()
    trainval_imgs = set(df[df["Patient ID"] < 8]["Image Index"])
    test_imgs = set(df[df["Patient ID"] >= 8]["Image Index"])
    cfg = {"val_frac": 0.25, "seed": 42}
    splits = make_splits(df, cfg, trainval_imgs, test_imgs)
    p = {k: set(v["Patient ID"]) for k, v in splits.items()}
    assert p["train"].isdisjoint(p["val"])
    assert p["train"].isdisjoint(p["test"])
    assert p["val"].isdisjoint(p["test"])
    assert p["test"] == {8, 9}
