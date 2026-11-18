"""Build the 12-d gating input (column order == contracts.GATING_FEATURE_NAMES) from a Bundle.

Columns: quality(3) | entropy of calibrated p (3) | cosdist(3) | p_cal(3); standardised with statistics fitted on
val_gating_fit. Variants: calibrate=False -> T=1 (raw sigmoid); use_ood=False -> the 9 profiler columns
(quality, entropy, cosdist) are zeroed so only p_cal reaches the gating network.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.core.config import Config
from src.core.contracts import GATING_DIM, SPLIT_VAL_FIT
from src.core.logger import get_logger
from src.learning.bundle_io import Bundle
from src.learning.calibration import TemperatureCalibrator

log = get_logger(__name__)

_EPS = 1e-7


def _local_entropy(p: np.ndarray) -> np.ndarray:
    """Binary entropy in nats (same formula as the profiler's predictive_entropy)."""
    p = np.clip(p, _EPS, 1 - _EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


# TODO(remove-fallback): once src/ood/profiler.py exposes `predictive_entropy`, drop _local_entropy and the guard.
try:  # depends on the parallel src/ood work
    from src.ood.profiler import predictive_entropy as _ood_entropy
except Exception:  # ImportError or module still incomplete
    _ood_entropy = None


def predictive_entropy(p: np.ndarray) -> np.ndarray:
    """Binary predictive entropy (nats). Uses src.ood.profiler when available, else the local formula."""
    p = np.asarray(p, dtype=float)
    if _ood_entropy is not None:
        try:
            return np.asarray(_ood_entropy(p), dtype=float).reshape(p.shape)
        except Exception:
            pass
    return _local_entropy(p)


@dataclass
class FeatureSet:
    X: np.ndarray        # (n,12) standardised gating input
    p: np.ndarray        # (n,3) calibrated P(fake) per expert
    mean_ood: np.ndarray  # (n,) mean raw cosdist (flag criterion)

    def __len__(self) -> int:
        return len(self.X)

    def subset(self, sel) -> "FeatureSet":
        return FeatureSet(self.X[sel], self.p[sel], self.mean_ood[sel])

    def concat(self, o: "FeatureSet") -> "FeatureSet":
        return FeatureSet(np.concatenate([self.X, o.X]), np.concatenate([self.p, o.p]),
                          np.concatenate([self.mean_ood, o.mean_ood]))


class FeaturePipeline:
    def __init__(self, cfg: Config, calibrate: bool = True, use_ood: bool = True):
        self.cfg, self.calibrate, self.use_ood = cfg, calibrate, use_ood
        self.calibrator = TemperatureCalibrator.identity()
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def raw_matrix(self, b: Bundle) -> np.ndarray:
        p = self.calibrator.transform(b.logits)
        return np.concatenate([b.quality, predictive_entropy(p), b.cosdist, p], axis=1)

    def fit(self, b: Bundle, split: str = SPLIT_VAL_FIT) -> "FeaturePipeline":
        want = self.cfg.get("calibration", {}).get("fit_split", SPLIT_VAL_FIT)
        if split != want:
            raise ValueError(f"calibration/scaler may only be fit on {want!r}, got {split!r}")
        if self.calibrate:
            self.calibrator = TemperatureCalibrator().fit(b.logits, b.label)
        X = self.raw_matrix(b)
        self.mean = X.mean(0)
        self.std = np.maximum(X.std(0), 1e-6)
        return self

    def transform(self, b: Bundle) -> FeatureSet:
        if self.mean is None:
            raise RuntimeError("FeaturePipeline not fitted")
        X = (self.raw_matrix(b) - self.mean) / self.std
        if not self.use_ood:
            X[:, :9] = 0.0
        assert X.shape[1] == GATING_DIM
        return FeatureSet(X.astype(np.float32), self.calibrator.transform(b.logits), b.mean_ood)
