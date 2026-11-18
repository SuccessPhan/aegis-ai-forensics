"""Accuracy / Precision / Recall / F1 / AUROC / ECE / Selective Accuracy / Flagged Rate -> EvalReport."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import roc_auc_score

from src.core.config import Config
from src.core.contracts import EvalReport


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """Top-label ECE: equal-width bins over confidence max(p,1-p), |accuracy - confidence| weighted by bin mass."""
    conf = np.maximum(p, 1 - p)
    correct = ((p >= 0.5).astype(int) == y).astype(float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def compute_metrics(y: np.ndarray, p: np.ndarray, cfg: Config, scenario: str = "", method: str = "",
                    flags: Optional[np.ndarray] = None, extras: Optional[Dict[str, Any]] = None) -> EvalReport:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    auroc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    sel_acc = flag_rate = None
    if flags is not None:
        flags = np.asarray(flags, dtype=bool)
        flag_rate = float(flags.mean())
        keep = ~flags
        sel_acc = float((pred[keep] == y[keep]).mean()) if keep.any() else float("nan")
    return EvalReport(scenario=scenario, method=method, n=int(len(y)), accuracy=float((pred == y).mean()),
                      precision=float(prec), recall=float(rec), f1=float(f1), auroc=auroc,
                      ece=expected_calibration_error(y, p, cfg["evaluation"]["ece_bins"]),
                      selective_accuracy=sel_acc, flagged_rate=flag_rate, extras=dict(extras or {}))
