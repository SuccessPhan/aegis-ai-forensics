"""Per-expert temperature scaling. Fit ONLY on val_gating_fit (guarded by FeaturePipeline.fit)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import minimize_scalar

from src.core.contracts import EXPERT_NAMES
from src.core.logger import get_logger

log = get_logger(__name__)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def nll(logit: np.ndarray, y: np.ndarray, T: float = 1.0) -> float:
    z = logit / T
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


class TemperatureCalibrator:
    """p_cal = sigmoid(logit / T_k) with one scalar T_k > 0 per expert (EXPERT_NAMES order)."""

    def __init__(self, temperatures: Optional[Sequence[float]] = None):
        self.T = np.ones(len(EXPERT_NAMES)) if temperatures is None else np.asarray(temperatures, dtype=float)

    @classmethod
    def identity(cls) -> "TemperatureCalibrator":
        return cls()

    def fit(self, logits: np.ndarray, y: np.ndarray, t_range=(0.05, 20.0)) -> "TemperatureCalibrator":
        y = np.asarray(y, dtype=float)
        lo, hi = np.log(t_range[0]), np.log(t_range[1])
        for k in range(logits.shape[1]):
            res = minimize_scalar(lambda lt: nll(logits[:, k], y, float(np.exp(lt))), bounds=(lo, hi),
                                  method="bounded", options={"xatol": 1e-4})
            self.T[k] = float(np.exp(res.x))
        log.info("temperatures fitted: %s", dict(zip(EXPERT_NAMES, (round(float(t), 4) for t in self.T))))
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return sigmoid(logits / self.T[None, :])

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"experts": list(EXPERT_NAMES), "T": self.T.tolist()}), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TemperatureCalibrator":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if d["experts"] != list(EXPERT_NAMES):
            raise ValueError("calibration file expert order mismatch")
        return cls(d["T"])
