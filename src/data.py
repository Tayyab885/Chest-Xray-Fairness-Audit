import os
import logging
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

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


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(cfg, train: bool):
    size = cfg["image_size"]
    ops = [transforms.Resize((size, size))]
    if train:
        ops += [transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(7)]
    ops += [transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)]
    return transforms.Compose(ops)


class ChestXrayDataset(torch.utils.data.Dataset):
    def __init__(self, df, cfg, transform, image_root=None):
        self.cfg = cfg
        self.transform = transform
        self.labels = cfg["labels"]
        self.root = image_root or cfg["data_dir"]
        rows, missing = [], 0
        for _, r in df.iterrows():
            if os.path.exists(os.path.join(self.root, r["Image Index"])):
                rows.append(r)
            else:
                missing += 1
        if df.shape[0] and missing / df.shape[0] > 0.01:
            raise RuntimeError(f"{missing}/{df.shape[0]} images missing (>1%)")
        if missing:
            logging.warning("Skipped %d missing images", missing)
        self.df = pd.DataFrame(rows).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(os.path.join(self.root, r["Image Index"])).convert("L")
        x = self.transform(img)
        y = torch.tensor([float(r[l]) for l in self.labels])
        return x, y, {"sex": r["sex"], "age_bin": r["age_bin"]}
