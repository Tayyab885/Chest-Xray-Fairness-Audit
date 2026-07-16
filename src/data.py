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
