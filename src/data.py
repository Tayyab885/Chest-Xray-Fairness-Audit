import numpy as np
import pandas as pd

def parse_labels(finding_str: str, labels: list) -> list:
    present = set(finding_str.split("|"))
    return [1 if lab in present else 0 for lab in labels]

def age_to_bin(age: int, bins: list, bin_labels: list) -> str:
    for i in range(len(bin_labels)):
        if bins[i] <= age < bins[i + 1]:
            return bin_labels[i]
    return bin_labels[-1]

def load_metadata(cfg: dict) -> pd.DataFrame:
    import os
    df = pd.read_csv(os.path.join(cfg["data_dir"], cfg["csv_name"]))
    df = df[(df["Patient Age"] > 0) & (df["Patient Age"] <= 100)].copy()
    df["age_bin"] = df["Patient Age"].apply(
        lambda a: age_to_bin(a, cfg["age_bins"], cfg["age_bin_labels"]))
    df["sex"] = df["Patient Gender"].str.upper().str[0]
    for lab in cfg["labels"]:
        df[lab] = df["Finding Labels"].apply(
            lambda s, l=lab: 1 if l in s.split("|") else 0)
    return df

def make_splits(df: pd.DataFrame, cfg: dict, trainval_imgs: set = None,
                test_imgs: set = None) -> dict:
    """Split df into train/val/test BY Patient ID (leakage-free), deterministic given cfg["seed"]."""
    if trainval_imgs is None or test_imgs is None:
        import os
        with open(os.path.join(cfg["data_dir"], cfg["train_val_list"])) as f:
            trainval_imgs = set(x.strip() for x in f if x.strip())
        with open(os.path.join(cfg["data_dir"], cfg["test_list"])) as f:
            test_imgs = set(x.strip() for x in f if x.strip())
    tv = df[df["Image Index"].isin(trainval_imgs)].copy()
    test = df[df["Image Index"].isin(test_imgs)].copy()
    patients = np.array(sorted(tv["Patient ID"].unique()))
    rng = np.random.default_rng(cfg["seed"])
    rng.shuffle(patients)
    n_val = int(len(patients) * cfg["val_frac"])
    val_pids = set(patients[:n_val])
    val = tv[tv["Patient ID"].isin(val_pids)].copy()
    train = tv[~tv["Patient ID"].isin(val_pids)].copy()
    return {"train": train, "val": val, "test": test}
