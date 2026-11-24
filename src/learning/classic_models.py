"""The three classic machine-learning models compared with the image models of AEGIS.

  rbf_svm        support vector machine with a Gaussian (RBF) kernel      - kernel method
  random_forest  300 decision trees on bootstrap samples, majority vote  - bagging ensemble of trees
  xgboost        gradient-boosted decision trees (XGBoost)                - boosting ensemble of trees

They are trained on three kinds of input (src.evaluation.ml_comparison):
  * hand-crafted features computed from the pixels (src.data.handcrafted): classic machine learning end to end,
    no deep network involved -- the direct answer to "can machine learning do as well as the deep image models?";
  * the frozen CLIP / DINOv2 embeddings: same features as the AEGIS experts, different classifier;
  * the 12-d gating input (`SklearnStacker`): a machine-learning model in place of the gating network.

Protocol (identical for every model, so the comparison is fair):
  1. hyperparameters are chosen on a VALIDATION split by AUROC (small grid, `ModelSpec.grid`); the test split is never
     touched while tuning;
  2. every model returns a raw score (decision_function, or logit of predict_proba), and a 1-D Platt scaler fitted on
     the same validation split turns it into P(fake). Accuracy@0.5 / ECE are therefore comparable across models, in
     the same way the repo calibrates the expert heads with a temperature fitted on val_gating_fit.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.core.logger import get_logger

log = get_logger(__name__)

_EPS = 1e-6
PCA_ABOVE = 256        # inputs wider than this (the 768/1024-d embeddings) are reduced to PCA_DIM first
PCA_DIM = 128


# ------------------------------------------------------------------------------------------------- the models
@dataclass
class ModelSpec:
    name: str
    family: str                                   # kernel / bagging / boosting
    factory: Callable[..., Any]                   # factory(n_samples, n_features, seed, **params) -> estimator
    grid: Dict[str, List[Any]] = field(default_factory=dict)
    note: str = ""


def _reduce(n_samples: int, n_features: int, seed: int):
    """PCA-128 for wide embeddings (keeps RBF-SVM / XGBoost fast); hand-crafted features and the 12-d gating input
    are used as they are."""
    if n_features <= PCA_ABOVE:
        return "passthrough"
    return PCA(n_components=min(PCA_DIM, max(n_samples - 1, 1)), random_state=seed)


def _rbf_svm(n, d, seed, C=1.0):
    return Pipeline([("scale", StandardScaler()), ("pca", _reduce(n, d, seed)),
                     ("clf", SVC(C=C, kernel="rbf", gamma="scale", random_state=seed))])


def _random_forest(n, d, seed, min_samples_leaf=1):
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=min_samples_leaf, max_features="sqrt",
                                  n_jobs=-1, random_state=seed)


def _xgboost(n, d, seed, max_depth=6):
    from xgboost import XGBClassifier
    return Pipeline([("pca", _reduce(n, d, seed)),
                     ("clf", XGBClassifier(n_estimators=400, max_depth=max_depth, learning_rate=0.1, subsample=0.8,
                                           colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=seed,
                                           verbosity=0))])


MODEL_SPECS: Dict[str, ModelSpec] = {s.name: s for s in [
    ModelSpec("rbf_svm", "kernel", _rbf_svm, {"C": [1.0, 10.0]},
              "SVM, Gaussian (RBF) kernel; standardised input (PCA-128 for embeddings)"),
    ModelSpec("random_forest", "bagging", _random_forest, {"min_samples_leaf": [1, 5]},
              "random forest, 300 trees, sqrt(d) features per split"),
    ModelSpec("xgboost", "boosting", _xgboost, {"max_depth": [4, 6]},
              "XGBoost, 400 boosted trees, lr 0.1 (PCA-128 for embeddings)"),
]}

DEFAULT_MODELS: Tuple[str, ...] = tuple(MODEL_SPECS)


def build_model(name: str, n_samples: int, n_features: int, seed: int = 0, cfg: Optional[dict] = None, **params):
    if name not in MODEL_SPECS:
        raise KeyError(f"unknown model {name!r}; choose from {list(MODEL_SPECS)}")
    return MODEL_SPECS[name].factory(n_samples, n_features, seed, **params)


# ---------------------------------------------------------------------------------------------- scores / calibration
def raw_score(model, X) -> np.ndarray:
    """Monotone score for class 1: decision_function if available, else logit(predict_proba)."""
    if hasattr(model, "decision_function"):
        try:
            return np.asarray(model.decision_function(X), dtype=float).ravel()
        except AttributeError:      # pipelines expose the method but the final step may lack it
            pass
    p = np.clip(model.predict_proba(X)[:, 1], _EPS, 1 - _EPS)
    return np.log(p) - np.log1p(-p)


class PlattScaler:
    """P(fake) = sigmoid(a * s + b), fitted on validation scores (1-D logistic regression)."""

    def fit(self, s: np.ndarray, y: np.ndarray) -> "PlattScaler":
        s = np.asarray(s, dtype=float).reshape(-1, 1)
        self.mu, self.sd = float(s.mean()), float(max(s.std(), _EPS))
        self.lr = LogisticRegression(C=1e4, max_iter=1000).fit((s - self.mu) / self.sd, np.asarray(y).astype(int))
        return self

    def transform(self, s: np.ndarray) -> np.ndarray:
        s = (np.asarray(s, dtype=float).reshape(-1, 1) - self.mu) / self.sd
        return self.lr.predict_proba(s)[:, 1]


def _auroc(y, s) -> float:
    y = np.asarray(y).astype(int)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


# ------------------------------------------------------------------------------------------------------ tuning
@dataclass
class TunedModel:
    name: str
    model: Any
    calibrator: PlattScaler
    params: Dict[str, Any]
    val_auroc: float
    fit_seconds: float
    trials: List[Dict[str, Any]]

    def predict_proba1(self, X) -> np.ndarray:
        return self.calibrator.transform(raw_score(self.model, X))

    def predict_timed(self, X) -> Tuple[np.ndarray, float]:
        t = time.perf_counter()
        p = self.predict_proba1(X)
        return p, time.perf_counter() - t


def tune(name: str, X_train, y_train, X_val, y_val, seed: int = 0, cfg: Optional[dict] = None,
         grid: Optional[Dict[str, List[Any]]] = None) -> TunedModel:
    """Fit every grid point on the training split, keep the best validation AUROC, Platt-calibrate on validation.
    The test split is not an argument: tuning cannot see it."""
    X_train, X_val = np.asarray(X_train, dtype=np.float32), np.asarray(X_val, dtype=np.float32)
    y_train, y_val = np.asarray(y_train).astype(int), np.asarray(y_val).astype(int)
    n, d = X_train.shape
    grid = MODEL_SPECS[name].grid if grid is None else grid
    best, trials = None, []
    for params in ParameterGrid(grid or [{}]):
        t = time.perf_counter()
        m = build_model(name, n, d, seed, cfg, **params).fit(X_train, y_train)
        secs = time.perf_counter() - t
        s_val = raw_score(m, X_val)
        auc = _auroc(y_val, s_val)
        trials.append({"model": name, **{f"param_{k}": v for k, v in params.items()}, "val_auroc": auc,
                       "fit_seconds": secs})
        key = -np.inf if np.isnan(auc) else auc
        if best is None or key > best[0]:
            best = (key, m, params, s_val, secs, auc)
    _, m, params, s_val, secs, auc = best
    log.info("%-14s best %s  val AUROC %.4f  (%.1fs)", name, params or "-", auc, secs)
    return TunedModel(name, m, PlattScaler().fit(s_val, y_val), dict(params), auc, secs, trials)


# ------------------------------------------------------------------------------------- fusion-level stacker
class SklearnStacker:
    """Any of the three models on the 12-d gating input FeatureSet.X (quality | entropy | cosdist | p_cal), with the same
    fit/predict interface as src.learning.baselines.StaticLearnedWeights. Weights are NOT per-expert here: the model
    maps the whole profile to P(fake) directly (a learned meta-classifier, i.e. stacking)."""

    def __init__(self, name: str, seed: int = 0, cfg: Optional[dict] = None, **params):
        self.name, self.seed, self.cfg, self.params = name, seed, cfg, params

    def fit(self, fs, y) -> "SklearnStacker":
        X = np.asarray(fs.X, dtype=np.float32)
        self.model = build_model(self.name, len(X), X.shape[1], self.seed, self.cfg, **self.params)
        self.model.fit(X, np.asarray(y).astype(int))
        if hasattr(self.model, "predict_proba"):
            self.calibrator = None
        else:
            self.calibrator = PlattScaler().fit(raw_score(self.model, X), y)
        return self

    def predict(self, fs) -> np.ndarray:
        X = np.asarray(fs.X, dtype=np.float32)
        if self.calibrator is None:
            return self.model.predict_proba(X)[:, 1]
        return self.calibrator.transform(raw_score(self.model, X))


def fusion_grid(name: str) -> Dict[str, List[Any]]:
    """Hyperparameter grid used on the 12-d gating input (same grid as on the image features)."""
    return dict(MODEL_SPECS[name].grid)


def tune_stacker(name: str, fit_fs, fit_y, sel_fs, sel_y, seed: int = 0, cfg: Optional[dict] = None
                 ) -> Tuple[SklearnStacker, Dict[str, Any], float]:
    """Fit on val_gating_fit features, select hyperparameters on a disjoint split (val_gating_verify)."""
    best = None
    for params in ParameterGrid(fusion_grid(name) or [{}]):
        st = SklearnStacker(name, seed, cfg, **params).fit(fit_fs, fit_y)
        auc = _auroc(sel_y, st.predict(sel_fs))
        if best is None or (not np.isnan(auc) and auc > best[2]):
            best = (st, dict(params), auc)
    return best


def describe_models(names: Optional[Sequence[str]] = None) -> List[Dict[str, str]]:
    return [{"model": n, "family": MODEL_SPECS[n].family, "grid": str(MODEL_SPECS[n].grid or "-"),
             "note": MODEL_SPECS[n].note} for n in (names or DEFAULT_MODELS)]
