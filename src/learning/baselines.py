"""Ablation baselines operating on FeatureSet.p (calibrated expert probabilities)."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.core.contracts import EXPERT_NAMES
from src.learning.features import FeatureSet


class SingleExpert:
    def __init__(self, expert: str):
        self.k = EXPERT_NAMES.index(expert)

    def predict(self, fs: FeatureSet) -> np.ndarray:
        return fs.p[:, self.k]


class NaiveAverage:
    def predict(self, fs: FeatureSet) -> np.ndarray:
        return fs.p.mean(axis=1)


class StaticLearnedWeights:
    """Logistic-regression stacking on p_cal, fit on val_gating_fit only; weights fixed across samples."""

    def __init__(self, C: float = 1.0):
        self.lr = LogisticRegression(C=C, max_iter=500)

    def fit(self, fs: FeatureSet, y: np.ndarray) -> "StaticLearnedWeights":
        self.lr.fit(fs.p, y)
        return self

    def predict(self, fs: FeatureSet) -> np.ndarray:
        return self.lr.predict_proba(fs.p)[:, 1]


class QualityOnly:
    """Shortcut FLOOR baseline: logistic regression on the 3 image-quality columns of the bundle ONLY (no expert
    probability, no OOD distance), standardised with statistics fitted on val_gating_fit. Independent of the head
    variant. Any method must beat this by a wide margin; it is not a competing baseline."""

    def __init__(self, C: float = 1.0):
        self.lr = LogisticRegression(C=C, max_iter=500)
        self.mean = None
        self.std = None

    def fit(self, b, y=None) -> "QualityOnly":
        q = np.asarray(b.quality, dtype=float)
        self.mean, self.std = q.mean(0), np.maximum(q.std(0), 1e-6)
        self.lr.fit((q - self.mean) / self.std, np.asarray(b.label if y is None else y))
        return self

    def predict_bundle(self, b) -> np.ndarray:
        return self.lr.predict_proba((np.asarray(b.quality, dtype=float) - self.mean) / self.std)[:, 1]
