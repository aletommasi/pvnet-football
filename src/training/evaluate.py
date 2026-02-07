from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from sklearn.calibration import calibration_curve

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def compute_metrics(logits, y_true):
    """
    logits: (N,2), y_true: (N,2)
    """
    probs = sigmoid(logits)
    metrics = {}

    for i, name in enumerate(["shot", "goal"]):
        y = y_true[:, i].astype(int)
        p = probs[:, i]

        if len(np.unique(y)) < 2:
            auc = None
            pr_auc = None
        else:
            auc = float(roc_auc_score(y, p))
            pr_auc = float(average_precision_score(y, p))

        metrics[name] = {
            "roc_auc": auc,
            "pr_auc": pr_auc,
            "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1-1e-6))),
            "brier": float(np.mean((p - y) ** 2)),
            "pos_rate": float(y.mean())
        }

    return metrics, probs

def calibration_data(probs, y_true, n_bins=10):
    out = {}
    for i, name in enumerate(["shot", "goal"]):
        y = y_true[:, i].astype(int)
        p = probs[:, i]
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="uniform")
        out[name] = {"mean_pred": mean_pred.tolist(), "frac_pos": frac_pos.tolist()}
    return out
