import numpy as np
import pandas as pd
from PIL import Image
from src.data import ChestXrayDataset, build_transforms

LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
          "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule",
          "Pleural_Thickening", "Pneumonia", "Pneumothorax"]


def _make(tmp_path, n=4):
    rows = []
    for i in range(n):
        Image.fromarray((np.random.rand(64, 64) * 255).astype("uint8")).save(tmp_path / f"{i}.png")
        rows.append({"Image Index": f"{i}.png", "sex": "M", "age_bin": "40-60",
                     **{l: (1 if i % 2 else 0) for l in LABELS}})
    return pd.DataFrame(rows)


def test_dataset_item_shapes(tmp_path):
    df = _make(tmp_path)
    cfg = {"labels": LABELS, "image_size": 224, "data_dir": str(tmp_path)}
    ds = ChestXrayDataset(df, cfg, build_transforms(cfg, train=False), image_root=str(tmp_path))
    img, label, meta = ds[0]
    assert tuple(img.shape) == (3, 224, 224)
    assert tuple(label.shape) == (14,)
    assert meta["sex"] == "M"
