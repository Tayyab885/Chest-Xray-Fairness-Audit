import os, argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.config import load_config
from src.data import load_metadata, make_splits, ChestXrayDataset, build_transforms
from src.model import build_model
from src.evaluate import predict, compute_auroc
from src.fairness import subgroup_auroc, equalized_odds, youden_threshold

def _load(cfg, ckpt, device):
    m = build_model(len(cfg["labels"]), pretrained=False)
    m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    return m.to(device)

def audit(cfg: dict, checkpoint: str) -> str:
    """Evaluate `checkpoint` on the test split, write metric + fairness tables to cfg["results_dir"]."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_metadata(cfg)
    test_df = make_splits(df, cfg)["test"]
    ds = ChestXrayDataset(test_df, cfg, build_transforms(cfg, False))
    loader = DataLoader(ds, batch_size=cfg["batch_size"], num_workers=cfg.get("num_workers", 2))
    model = _load(cfg, checkpoint, device)
    yt, yp, metas = predict(model, loader, device)
    os.makedirs(cfg["results_dir"], exist_ok=True)
    auroc = compute_auroc(yt, yp, cfg["labels"])
    pd.Series(auroc).to_csv(os.path.join(cfg["results_dir"], "auroc.csv"))
    for key, fname in [("sex", "fairness_sex.csv"), ("age_bin", "fairness_age.csv")]:
        sg_full = subgroup_auroc(yt, yp, metas, cfg["labels"], key)
        sg = {lab: sg_full[lab] for lab in cfg["deep_dive_labels"]}
        pd.DataFrame(sg).T.to_csv(os.path.join(cfg["results_dir"], fname))
    rows = []
    for lab in cfg["deep_dive_labels"]:
        li = cfg["labels"].index(lab)
        thr = youden_threshold(yt[:, li], yp[:, li])
        eo = equalized_odds(yt, yp, metas, li, "sex", thr)
        rows.append({"label": lab,
                     **{f"{g}_{k}": v for g, d in eo.items()
                        if isinstance(d, dict) for k, v in d.items()},
                     "tpr_gap": eo["tpr_gap"], "fpr_gap": eo["fpr_gap"]})
    pd.DataFrame(rows).to_csv(os.path.join(cfg["results_dir"], "eqodds.csv"), index=False)
    print("Audit complete. Results in", cfg["results_dir"])
    return cfg["results_dir"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    audit(load_config(args.config), args.checkpoint)

if __name__ == "__main__":
    main()
