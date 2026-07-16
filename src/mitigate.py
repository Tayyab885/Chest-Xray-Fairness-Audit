import torch
from torch.utils.data import WeightedRandomSampler
from src.config import load_config
from src.train import run_training


def build_group_sampler(train_df, group_key) -> WeightedRandomSampler:
	"""Build a weighted sampler balancing groups within training data."""
	counts = train_df[group_key].value_counts().to_dict()
	weights = train_df[group_key].map(lambda g: 1.0 / counts[g]).to_numpy()
	return WeightedRandomSampler(weights=torch.as_tensor(weights, dtype=torch.double),
	                             num_samples=len(weights), replacement=True)


def run_mitigation(cfg: dict, group_key: str = "sex") -> str:
	"""Retrain with group-balanced resampling on `group_key`, tagged 'mitigated'."""
	return run_training(cfg, balance_group=group_key, tag="mitigated")


if __name__ == "__main__":
	import logging
	logging.basicConfig(level=logging.INFO)
	run_mitigation(load_config())
