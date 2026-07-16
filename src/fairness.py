import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

def _mask(metas, group_key, group_val):
	return np.array([m[group_key] == group_val for m in metas])

def subgroup_auroc(y_true: np.ndarray, y_prob: np.ndarray, metas: list, labels: list, group_key: str) -> dict:
	"""Compute AUROC per label and per group; include group gap."""
	groups = sorted({m[group_key] for m in metas})
	out = {}
	for li, lab in enumerate(labels):
		out[lab] = {}
		vals = []
		for g in groups:
			mk = _mask(metas, group_key, g)
			yt, yp = y_true[mk, li], y_prob[mk, li]
			if len(np.unique(yt)) < 2:
				out[lab][g] = float("nan")
				continue
			s = float(roc_auc_score(yt, yp))
			out[lab][g] = s
			vals.append(s)
		out[lab]["gap"] = (max(vals) - min(vals)) if len(vals) > 1 else float("nan")
	return out

def youden_threshold(y_true_col: np.ndarray, y_prob_col: np.ndarray) -> float:
	"""Find threshold maximizing TPR - FPR (Youden's J); 0.5 if only one class present."""
	if len(np.unique(y_true_col)) < 2:
		return 0.5
	fpr, tpr, thr = roc_curve(y_true_col, y_prob_col)
	return float(thr[np.argmax(tpr - fpr)])

def equalized_odds(y_true: np.ndarray, y_prob: np.ndarray, metas: list, label_idx: int, group_key: str, threshold: float) -> dict:
	"""Compute TPR/FPR per group and gaps for equalized odds."""
	groups = sorted({m[group_key] for m in metas})
	res, tprs, fprs = {}, [], []
	for g in groups:
		mk = _mask(metas, group_key, g)
		yt = y_true[mk, label_idx]
		pred = (y_prob[mk, label_idx] >= threshold).astype(int)
		tp = int(((pred == 1) & (yt == 1)).sum()); fn = int(((pred == 0) & (yt == 1)).sum())
		fp = int(((pred == 1) & (yt == 0)).sum()); tn = int(((pred == 0) & (yt == 0)).sum())
		tpr = tp / (tp + fn) if (tp + fn) else float("nan")
		fpr = fp / (fp + tn) if (fp + tn) else float("nan")
		res[g] = {"tpr": tpr, "fpr": fpr}
		tprs.append(tpr); fprs.append(fpr)
	res["tpr_gap"] = float(np.nanmax(tprs) - np.nanmin(tprs))
	res["fpr_gap"] = float(np.nanmax(fprs) - np.nanmin(fprs))
	return res
