"""Classic machine-learning model zoo used to compare against the AEGIS linear heads and the gating network.

By default two models are compared with the reference `linear_head` (COMPARISON_MODELS: RBF-SVM and random forest);
the other zoo models stay available through ALL_MODELS / the `models` argument.

The same zoo works at two levels of the system:
  * embedding level  - replaces the `LinearHead` on frozen CLIP / DINOv2 features (src.evaluation.ml_comparison);
  * fusion level     - replaces Static LR / the gating MLP on the 12-d gating input (`SklearnStacker`).

Protocol (identical for every model, so the comparison is fair):
  1. hyperparameters are chosen on a VALIDATION split by AUROC (small grid, `ModelSpec.grid`); the test split is never
     touched while tuning;
  2. every model returns a raw score (logit of predict_proba, or decision_function), and a 1-D Platt scaler fitted on
     the same validation split turns it into P(fake). Accuracy@0.5 / ECE are therefore comparable across models, in
     the same way the repo calibrates the expert heads with a temperature fitted on val_gating_fit.

`linear_head` is the repo's own head (src.models.heads.train_linear_head: standardise + Linear + Adam) wrapped as a
sklearn estimator: it is the reference every other model is compared against.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from src.core.logger import get_logger

log = get_logger(__name__)

_EPS = 1e-6


# --------------------------------------------------------------------------------------- repo baseline as estimator
class LinearHeadClassifier(BaseEstimator, ClassifierMixin):
    """sklearn wrapper around src.models.heads.train_linear_head (the head AEGIS actually uses).
    Hyperparameters come from cfg['experts']['head'] (lr, weight_decay, epochs, batch_size) unless overridden."""

    def __init__(self, cfg: Optional[dict] = None, epochs: Optional[int] = None, seed: int = 0):
        self.cfg = cfg
        self.epochs = epochs
        self.seed = seed

    def _cfg(self) -> dict:
        base = self.cfg or {"experts": {"head": {"lr": 1e-3, "weight_decay": 1e-4, "epochs": 20, "batch_size": 256}}}
        head = dict(base["experts"]["head"])
        if self.epochs is not None:
            head["epochs"] = self.epochs
        return {**base, "seed": self.seed, "experts": {**base["experts"], "head": head}}

    def fit(self, X, y):
        import torch
        from src.models.heads import train_linear_head
        self.classes_ = np.array([0, 1])
        self.head_ = train_linear_head(torch.as_tensor(np.asarray(X), dtype=torch.float32),
                                       torch.as_tensor(np.asarray(y)), self._cfg())
        return self

    def decision_function(self, X):
        import torch
        with torch.no_grad():
            return self.head_(torch.as_tensor(np.asarray(X), dtype=torch.float32)).numpy().astype(float)


# ------------------------------------------------------------------------------------------------- model zoo
@dataclass
class ModelSpec:
    name: str
    family: str                                   # linear / kernel / instance / tree / ensemble / neural / probabilistic
    factory: Callable[..., Any]                   # factory(n_samples, n_features, seed, cfg, **params) -> estimator
    grid: Dict[str, List[Any]] = field(default_factory=dict)
    note: str = ""


def _pca(k: int, n_samples: int, n_features: int, seed: int):
    """PCA clamped to what the data allows; None when no reduction is needed (pipeline step 'passthrough')."""
    k = min(k, n_features, max(n_samples - 1, 1))
    return "passthrough" if k >= n_features else PCA(n_components=k, random_state=seed)


def _linear_head(n, d, seed, cfg, **p):
    return LinearHeadClassifier(cfg=cfg, seed=seed, **p)


def _logreg(n, d, seed, cfg, C=1.0):
    return Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(C=C, max_iter=2000))])


def _linear_svm(n, d, seed, cfg, C=0.01):
    return Pipeline([("scale", StandardScaler()), ("clf", LinearSVC(C=C, max_iter=5000, random_state=seed))])


def _rbf_svm(n, d, seed, cfg, C=1.0, pca=128):
    return Pipeline([("scale", StandardScaler()), ("pca", _pca(pca, n, d, seed)),
                     ("clf", SVC(C=C, kernel="rbf", gamma="scale", random_state=seed))])


def _knn(n, d, seed, cfg, k=15, pca=64):
    # cosine geometry: L2-normalise, then Euclidean kNN == cosine ranking
    return Pipeline([("l2", Normalizer()), ("pca", _pca(pca, n, d, seed)), ("l2b", Normalizer()),
                     ("clf", KNeighborsClassifier(n_neighbors=max(1, min(k, n - 1)), weights="distance"))])


def _tree(n, d, seed, cfg, max_depth=10):
    return DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=5, random_state=seed)


def _rf(n, d, seed, cfg, min_samples_leaf=1, n_estimators=300):
    return RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_features="sqrt",
                                  n_jobs=-1, random_state=seed)


def _hgb(n, d, seed, cfg, max_leaf_nodes=31, pca=128, learning_rate=0.1):
    return Pipeline([("pca", _pca(pca, n, d, seed)),
                     ("clf", HistGradientBoostingClassifier(learning_rate=learning_rate, max_iter=300,
                                                            max_leaf_nodes=max_leaf_nodes, early_stopping=True,
                                                            validation_fraction=0.1, random_state=seed))])


def _xgb(n, d, seed, cfg, max_depth=6, pca=128):
    from xgboost import XGBClassifier
    return Pipeline([("pca", _pca(pca, n, d, seed)),
                     ("clf", XGBClassifier(n_estimators=400, max_depth=max_depth, learning_rate=0.1, subsample=0.8,
                                           colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=seed,
                                           verbosity=0))])


def _mlp(n, d, seed, cfg, alpha=1e-3, hidden=256):
    return Pipeline([("scale", StandardScaler()),
                     ("clf", MLPClassifier(hidden_layer_sizes=(hidden,), alpha=alpha, early_stopping=True,
                                           max_iter=300, random_state=seed))])


def _lda(n, d, seed, cfg, pca=256):
    return Pipeline([("scale", StandardScaler()), ("pca", _pca(pca, n, d, seed)),
                     ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))])


def _gnb(n, d, seed, cfg, pca=64):
    return Pipeline([("scale", StandardScaler()), ("pca", _pca(pca, n, d, seed)), ("clf", GaussianNB())])


def _has(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:
        return False


MODEL_SPECS: Dict[str, ModelSpec] = {s.name: s for s in [
    ModelSpec("linear_head", "linear", _linear_head, {}, "AEGIS head (reference): standardise + Linear + Adam"),
    ModelSpec("logreg", "linear", _logreg, {"C": [0.01, 0.1, 1.0]}, "L2 logistic regression"),
    ModelSpec("linear_svm", "linear", _linear_svm, {"C": [0.001, 0.01, 0.1]}, "linear SVM (hinge loss)"),
    ModelSpec("lda", "probabilistic", _lda, {"pca": [64, 256]}, "LDA with Ledoit-Wolf shrinkage"),
    ModelSpec("gaussian_nb", "probabilistic", _gnb, {"pca": [16, 64]}, "Gaussian Naive Bayes on PCA"),
    ModelSpec("knn", "instance", _knn, {"k": [5, 15, 45]}, "kNN, cosine geometry, distance-weighted"),
    ModelSpec("rbf_svm", "kernel", _rbf_svm, {"C": [1.0, 10.0]}, "RBF-kernel SVM on PCA-128"),
    ModelSpec("decision_tree", "tree", _tree, {"max_depth": [5, 10, 20]}, "single CART tree"),
    ModelSpec("random_forest", "ensemble", _rf, {"min_samples_leaf": [1, 5]}, "random forest, 300 trees"),
    ModelSpec("hist_gb", "ensemble", _hgb, {"max_leaf_nodes": [15, 31]}, "histogram gradient boosting on PCA-128"),
    ModelSpec("xgboost", "ensemble", _xgb, {"max_depth": [4, 6]}, "XGBoost on PCA-128 (optional dependency)"),
    ModelSpec("mlp", "neural", _mlp, {"alpha": [1e-4, 1e-2]}, "MLP, 1 hidden layer (256), early stopping"),
]}

REFERENCE = "linear_head"
# Every model in the zoo that can run here (xgboost only when installed).
ALL_MODELS: Tuple[str, ...] = tuple(m for m in MODEL_SPECS if m != "xgboost" or _has("xgboost"))
# Models compared by default against the reference: one kernel method and one tree ensemble. To compare more, add
# names from ALL_MODELS here (or pass models=... / --models ...); nothing else needs to change.
COMPARISON_MODELS: Tuple[str, ...] = ("rbf_svm", "random_forest")
DEFAULT_MODELS: Tuple[str, ...] = (REFERENCE,) + COMPARISON_MODELS


def available_models() -> List[str]:
    return list(ALL_MODELS)


def build_model(name: str, n_samples: int, n_features: int, seed: int = 0, cfg: Optional[dict] = None, **params):
    if name not in MODEL_SPECS:
        raise KeyError(f"unknown model {name!r}; choose from {list(MODEL_SPECS)}")
    return MODEL_SPECS[name].factory(n_samples, n_features, seed, cfg, **params)


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
    """Any zoo model on the 12-d gating input FeatureSet.X (quality | entropy | cosdist | p_cal), with the same
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
    """Grids for the 12-d fusion input: PCA settings are meaningless there, keep the other knobs."""
    return {k: v for k, v in MODEL_SPECS[name].grid.items() if k != "pca"}


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
