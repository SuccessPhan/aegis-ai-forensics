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
