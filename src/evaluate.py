import numpy as np
import torch
from sklearn.metrics import roc_auc_score

def compute_auroc(y_true, y_prob, labels):
    out = {}
    scores = []
    for i, lab in enumerate(labels):
        yt = y_true[:, i]
        if len(np.unique(yt)) < 2:
            out[lab] = float("nan")
            continue
        s = roc_auc_score(yt, y_prob[:, i])
        out[lab] = float(s)
        scores.append(s)
    out["macro"] = float(np.mean(scores)) if scores else float("nan")
    return out

def predict(model, loader, device):
    model.eval()
    ys, ps, metas = [], [], []
    with torch.no_grad():
        for x, y, meta in loader:
            x = x.to(device)
            p = torch.sigmoid(model(x)).cpu().numpy()
            ps.append(p); ys.append(y.numpy())
            for j in range(len(y)):
                metas.append({"sex": meta["sex"][j], "age_bin": meta["age_bin"][j]})
    return np.concatenate(ys), np.concatenate(ps), metas
