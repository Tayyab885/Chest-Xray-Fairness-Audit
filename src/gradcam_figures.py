"""Generate side-by-side Grad-CAM overlays per subgroup for each deep-dive label.

For every deep-dive label and every subgroup value (sex M/F, or age bin), pick the
test-set example the model is *most confident* is positive (a confident true positive,
so the saliency map is meaningful), compute a Grad-CAM heatmap on features.norm5, and
render the 224x224 input next to its heatmap overlay. One figure per (label, group_key).
"""
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import load_config
from src.data import load_metadata, make_splits, ChestXrayDataset, build_transforms
from src.model import build_model
from src.gradcam import gradcam

_MEAN = np.array([0.485, 0.456, 0.406])
_STD = np.array([0.229, 0.224, 0.225])


def _denorm(x):
    """[3,224,224] normalized tensor -> [224,224] grayscale in [0,1] for display."""
    img = x.cpu().numpy().transpose(1, 2, 0) * _STD + _MEAN
    return np.clip(img.mean(axis=2), 0, 1)


def _best_positive(model, df, cfg, label, device, scan):
    """Row + input tensor for the highest-prob positive of `label` in `df` (None if none)."""
    li = cfg["labels"].index(label)
    pos = df[df[label] == 1]
    if pos.empty:
        return None
    pos = pos.head(scan)
    ds = ChestXrayDataset(pos, cfg, build_transforms(cfg, False))
    if len(ds) == 0:
        return None
    loader = DataLoader(ds, batch_size=cfg["batch_size"])
    probs, tensors = [], []
    model.eval()
    with torch.no_grad():
        for x, _, _ in loader:
            p = torch.sigmoid(model(x.to(device)))[:, li]
            probs.append(p.cpu())
            tensors.append(x)
    probs = torch.cat(probs)
    tensors = torch.cat(tensors)
    best = int(probs.argmax())
    return {"tensor": tensors[best:best + 1], "prob": float(probs[best]),
            "meta": ds.df.iloc[best]}


def render(cfg, checkpoint, out_dir, group_key, group_values, scan):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_metadata(cfg)
    test_df = make_splits(df, cfg)["test"]
    model = build_model(len(cfg["labels"]), pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model = model.to(device)
    os.makedirs(out_dir, exist_ok=True)

    written = []
    for label in cfg["deep_dive_labels"]:
        li = cfg["labels"].index(label)
        cols = len(group_values)
        fig, axes = plt.subplots(2, cols, figsize=(3 * cols, 6.2), squeeze=False)
        found = False
        for c, gv in enumerate(group_values):
            sub = test_df[test_df[group_key] == gv]
            pick = _best_positive(model, sub, cfg, label, device, scan)
            top, bot = axes[0][c], axes[1][c]
            top.set_title(f"{group_key}={gv}", fontsize=11)
            for ax in (top, bot):
                ax.set_xticks([]); ax.set_yticks([])
            if pick is None:
                top.text(0.5, 0.5, "no positive\nexample", ha="center", va="center")
                bot.axis("off")
                continue
            found = True
            base = _denorm(pick["tensor"][0])
            cam = gradcam(model, pick["tensor"].clone(), li, device)
            top.imshow(base, cmap="gray")
            bot.imshow(base, cmap="gray")
            bot.imshow(cam, cmap="jet", alpha=0.45)
            bot.set_xlabel(f"p={pick['prob']:.2f}", fontsize=9)
        if not found:
            plt.close(fig)
            continue
        axes[0][0].set_ylabel("input", fontsize=10)
        axes[1][0].set_ylabel("Grad-CAM", fontsize=10)
        fig.suptitle(f"{label}: Grad-CAM by {group_key}", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        path = os.path.join(out_dir, f"gradcam_{label.lower()}_{group_key}.png")
        fig.savefig(path, dpi=110)
        plt.close(fig)
        written.append(path)
        print("wrote", path)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="results/baseline/gradcam")
    ap.add_argument("--scan", type=int, default=60,
                    help="max positive candidates per subgroup to score for the best one")
    args = ap.parse_args()
    cfg = load_config(args.config)
    render(cfg, args.checkpoint, args.out, "sex", ["M", "F"], args.scan)
    render(cfg, args.checkpoint, args.out, "age_bin", cfg["age_bin_labels"], args.scan)


if __name__ == "__main__":
    main()
