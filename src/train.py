import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.config import load_config
from src.data import load_metadata, make_splits, ChestXrayDataset, build_transforms
from src.model import build_model
from src.evaluate import predict, compute_auroc


def train_one_epoch(model, loader, optimizer, scaler, device):
    """Train for one epoch, return mean loss."""
    model.train()
    crit = nn.BCEWithLogitsLoss()
    total, n = 0.0, 0
    for x, y, _ in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            loss = crit(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * len(y)
        n += len(y)
    return total / max(n, 1)


def run_training(cfg, balance_group=None, tag="baseline"):
    """Full training pipeline with early stopping on val macro AUROC. Returns best checkpoint path."""
    if not torch.cuda.is_available():
        raise SystemExit("No GPU detected. Full training requires CUDA. Aborting.")
    device = torch.device("cuda")

    # Load and optionally subsample metadata
    df = load_metadata(cfg)
    if cfg["subsample_frac"] < 1.0:
        df = df.sample(frac=cfg["subsample_frac"], random_state=cfg["seed"])

    # Build splits
    splits = make_splits(df, cfg)

    # Build datasets and loaders
    train_ds = ChestXrayDataset(splits["train"], cfg, build_transforms(cfg, True))
    val_ds = ChestXrayDataset(splits["val"], cfg, build_transforms(cfg, False))
    sampler = None
    if balance_group is not None:
        from src.mitigate import build_group_sampler
        sampler = build_group_sampler(train_ds.df, balance_group)  # post-filter frame
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=2
    )
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], num_workers=2)

    # Build model, optimizer, AMP scaler
    model = build_model(len(cfg["labels"]), pretrained=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scaler = torch.cuda.amp.GradScaler()

    # Setup checkpoint directory
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    best, best_path, patience = -1.0, os.path.join(cfg["checkpoint_dir"], f"{tag}.pt"), 0

    # Training loop with early stopping
    for epoch in range(cfg["epochs"]):
        loss = train_one_epoch(model, train_loader, opt, scaler, device)
        yt, yp, _ = predict(model, val_loader, device)
        macro = compute_auroc(yt, yp, cfg["labels"])["macro"]
        logging.info("epoch %d loss %.4f val_macro_auroc %.4f", epoch, loss, macro)

        if macro > best:
            best, patience = macro, 0
            torch.save(model.state_dict(), best_path)
        else:
            patience += 1
            if patience >= cfg["early_stop_patience"]:
                break

    return best_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_training(load_config())
