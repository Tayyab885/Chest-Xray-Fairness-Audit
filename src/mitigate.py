import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from src.config import load_config
from src.train import run_training


def build_group_sampler(train_df, group_key):
    counts = train_df[group_key].value_counts().to_dict()
    weights = train_df[group_key].map(lambda g: 1.0 / counts[g]).to_numpy()
    return WeightedRandomSampler(weights=torch.as_tensor(weights, dtype=torch.double),
                                 num_samples=len(weights), replacement=True)


def run_mitigation(cfg, group_key="sex"):
    from src.data import load_metadata, make_splits
    df = load_metadata(cfg)
    if cfg["subsample_frac"] < 1.0:
        df = df.sample(frac=cfg["subsample_frac"], random_state=cfg["seed"])
    train_df = make_splits(df, cfg)["train"]
    sampler = build_group_sampler(train_df, group_key)
    return run_training(cfg, sampler=sampler, tag="mitigated")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    run_mitigation(load_config())
