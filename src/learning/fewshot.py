"""Data-aware few-shot adaptation (brief 5.4).

N < conservative_below  : alpha = 0, weights unchanged, scarcity warning.
conservative_below <= N <= anchor_max : w_adapted = (1-alpha) w_gating(x) + alpha w_few_shot, alpha=min(cap, N/anchor_max).
N > anchor_max          : refit the gating on val_gating_fit + support (new model; the base is never modified).

Support labels must be GROUND TRUTH from the split manifest (s4_support pool). Detector predictions are never
used as labels. Per user decision (2026-11-18) few-shot only recalibrates weights; no prototype update.
"""
from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

from src.core.config import Config
from src.core.logger import get_logger
from src.learning.bundle_io import Bundle
from src.learning.features import FeatureSet
from src.learning.gating import GatingController

log = get_logger(__name__)

REGIME_CONSERVATIVE, REGIME_ANCHOR, REGIME_REFIT = "conservative", "anchor", "refit"


def select_support(b: Bundle, n: int, seed: int, generator: Optional[str] = None) -> Bundle:
    """Seeded support set from the s4_support bundle: ceil(n/2) AI + floor(n/2) Real, clean only.
    Mirrors src.data.split.get_support_set (sorted by sha, seeded permutation, nested across N)."""
    clean = np.flatnonzero(b.clean_mask())
    ai = sorted((i for i in clean if b.label[i] == 1 and (generator is None or b.generator[i] == generator)),
                key=lambda i: b.sha256[i])
    real = sorted((i for i in clean if b.label[i] == 0), key=lambda i: b.sha256[i])
    n_ai, n_real = (n + 1) // 2, n // 2
    if len(ai) < n_ai or len(real) < n_real:
        raise ValueError(f"support pool too small for N={n}: AI {len(ai)}/{n_ai}, Real {len(real)}/{n_real}")
    rng = np.random.default_rng(seed)
    ai_p = [ai[i] for i in rng.permutation(len(ai))]
    real_p = [real[i] for i in rng.permutation(len(real))]
    return b.subset(np.array(ai_p[:n_ai] + real_p[:n_real], dtype=int))


class AdaptedModel:
    """Anchor-blended routing: w(x) = (1-alpha) * base.weights(x) + alpha * w_few_shot. Base is not modified."""

    def __init__(self, base: GatingController, w_few_shot: Optional[np.ndarray], alpha: float):
        self.base, self.w_few_shot, self.alpha = base, w_few_shot, alpha

    def weights(self, fs: FeatureSet) -> np.ndarray:
        w = self.base.weights(fs)
        if self.alpha == 0.0 or self.w_few_shot is None:
            return w
        return (1 - self.alpha) * w + self.alpha * self.w_few_shot[None, :]

    def predict(self, fs: FeatureSet) -> np.ndarray:
        return (self.weights(fs) * fs.p).sum(1)


@dataclass
class AdaptResult:
    regime: str
    n: int
    alpha: float
    model: Any                       # exposes weights(fs) and predict(fs)
    source: str                      # GatingWeights.source: "base" | "adapted" | "refit"
    w_few_shot: Optional[np.ndarray] = None
    warning: Optional[str] = None


def fit_few_shot_weights(p: np.ndarray, y: np.ndarray, w_ref: np.ndarray, w_min: float, scale: float,
                         step: float = 0.05, reg: float = 0.1) -> np.ndarray:
    """Grid search over the bounded simplex for the global weight vector minimising
    BCE(sum_i w_i p_i, y) + reg * ||w - w_ref||^2 on the support set (L2 anchor to the gating's own weights)."""
    k = int(round(1.0 / step))
    kmin, kmax = int(round(w_min / step)), int(round((w_min + scale) / step))
    grid = np.array([c for c in itertools.product(range(kmin, kmax + 1), repeat=3) if sum(c) == k],
                    dtype=float) * step
    pf = np.clip(p @ grid.T, 1e-6, 1 - 1e-6)                 # (n, G)
    yy = y[:, None].astype(float)
    bce = -(yy * np.log(pf) + (1 - yy) * np.log(1 - pf)).mean(0)
    loss = bce + reg * ((grid - w_ref[None, :]) ** 2).sum(1)
    return grid[int(np.argmin(loss))]


def adapt(base: GatingController, fit_fs: FeatureSet, fit_y: np.ndarray, sup_fs: FeatureSet, sup_y: np.ndarray,
          cfg: Config, seed: Optional[int] = None) -> AdaptResult:
    """Return the adapted model for a support set of ground-truth-labelled samples (N = len(sup_y))."""
    fc, g = cfg["fewshot"], cfg["gating"]
    n = len(sup_y)
    if n < fc["conservative_below"]:
        msg = (f"few-shot data scarcity: N={n} < {fc['conservative_below']}; keeping gating weights unchanged "
               f"(alpha=0)")
        log.warning(msg)
        return AdaptResult(REGIME_CONSERVATIVE, n, 0.0, AdaptedModel(base, None, 0.0), "base", None, msg)
    if n <= fc["anchor_max"]:
        alpha = min(fc["alpha_cap"], n / fc["anchor_max"])
        w_ref = base.weights(sup_fs).mean(0)
        w_fs = fit_few_shot_weights(sup_fs.p, np.asarray(sup_y), w_ref, g["w_min"], g["scale"],
                                    fc.get("grid_step", 0.05), fc.get("anchor_reg", 0.1))
        log.info("few-shot anchor update: N=%d alpha=%.2f w_few_shot=%s", n, alpha, np.round(w_fs, 3))
        return AdaptResult(REGIME_ANCHOR, n, alpha, AdaptedModel(base, w_fs, alpha), "adapted", w_fs)
    log.info("few-shot full refit: N=%d > %d (val_gating_fit + support)", n, fc["anchor_max"])
    rep = int(fc.get("support_repeat", 1))
    X = fit_fs
    for _ in range(rep):
        X = X.concat(sup_fs)
    y_all = np.concatenate([np.asarray(fit_y)] + [np.asarray(sup_y)] * rep)
    new = GatingController(cfg, seed=seed).fit(X, y_all)
    return AdaptResult(REGIME_REFIT, n, 0.0, new, "refit")
