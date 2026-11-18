"""Fusion of per-expert outputs with gating weights w (n,3).

"prob" : P = sum_i w_i * P_i                      (P_i = calibrated probability)
"logit": P = sigmoid(sum_i w_i * l_i)             (l_i = raw logit / T_i, the calibrated logit)
Config key gating.fusion (default "prob"). Weights always come from the same gating MLP / constraints.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

FUSIONS = ("prob", "logit")


def get_fusion(cfg: dict) -> str:
    f = (cfg.get("gating") or {}).get("fusion", "prob")
    if f not in FUSIONS:
        raise ValueError(f"gating.fusion must be one of {FUSIONS}, got {f!r}")
    return f


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def to_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p) - np.log1p(-p)


def calibrated_logits(p: np.ndarray, z: Optional[np.ndarray]) -> np.ndarray:
    """z if available, else recovered from the calibrated probabilities."""
    return to_logit(p) if z is None else z


def fuse(w: np.ndarray, p: np.ndarray, z: Optional[np.ndarray], fusion: str) -> np.ndarray:
    if fusion == "prob":
        return (w * p).sum(1)
    if fusion == "logit":
        return sigmoid((w * calibrated_logits(p, z)).sum(1))
    raise ValueError(f"unknown fusion {fusion!r}")
