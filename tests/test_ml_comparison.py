"""Classic ML model zoo + embedding / fusion-level comparison (src.learning.classic_models, src.evaluation.ml_comparison)."""
import numpy as np
import pytest
import torch
from sklearn.datasets import make_classification

from src.evaluation.ml_comparison import (RESULT_COLUMNS, compare_on_embeddings, iter_slices, load_all,
                                          run_comparison, run_fusion_comparison, summary_table, write_outputs)
from src.learning.classic_models import (DEFAULT_MODELS, MODEL_SPECS, REFERENCE, LinearHeadClassifier, PlattScaler,
                                         SklearnStacker, build_model, raw_score, tune)
from src.learning.features import FeatureSet

ROOT_DEMO = "test_demo_cache"
FAST_MODELS = ["linear_head", "logreg", "knn", "decision_tree", "gaussian_nb"]


@pytest.fixture(scope="module")
def xy():
    X, y = make_classification(n_samples=360, n_features=40, n_informative=8, random_state=0)
    return X[:200], y[:200], X[200:280], y[200:280], X[280:], y[280:]


@pytest.fixture(scope="module")
def demo(cfg):
    from tests.conftest import ROOT
    from src.models.demo_features import build_demo
    return build_demo(ROOT / "data" / ROOT_DEMO, scale=2, cfg=cfg)


@pytest.mark.parametrize("name", list(DEFAULT_MODELS))
def test_every_model_fits_and_beats_chance(name, xy, cfg):
    Xtr, ytr, Xva, yva, Xte, yte = xy
    tm = tune(name, Xtr, ytr, Xva, yva, seed=0, cfg=cfg)
    p = tm.predict_proba1(Xte)
    assert p.shape == (len(Xte),) and np.all((p >= 0) & (p <= 1))
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(yte, p) > 0.6, name
    assert len(tm.trials) == max(1, int(np.prod([len(v) for v in MODEL_SPECS[name].grid.values()] or [1])))


def test_tune_keeps_best_validation_auroc(xy, cfg):
    Xtr, ytr, Xva, yva, _, _ = xy
    tm = tune("logreg", Xtr, ytr, Xva, yva, cfg=cfg)
    assert tm.val_auroc == pytest.approx(max(t["val_auroc"] for t in tm.trials))


def test_pca_is_clamped_for_small_inputs():
    X, y = make_classification(n_samples=30, n_features=6, random_state=1)
    for name in ("rbf_svm", "lda", "gaussian_nb", "knn", "hist_gb"):
        build_model(name, len(X), X.shape[1], pca=512).fit(X, y)       # PCA wider than the data must not fail


def test_linear_head_wrapper_matches_repo_head(xy, cfg):
    from src.models.heads import train_linear_head
    Xtr, ytr, _, _, Xte, _ = xy
    w = LinearHeadClassifier(cfg=cfg, seed=cfg["seed"]).fit(Xtr, ytr)
    h = train_linear_head(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr), cfg)
    with torch.no_grad():
        ref = h(torch.tensor(Xte, dtype=torch.float32)).numpy()
    np.testing.assert_allclose(raw_score(w, Xte), ref, rtol=1e-4, atol=1e-5)


def test_platt_scaler_is_monotone():
    s = np.linspace(-3, 3, 50)
    y = (s + np.random.default_rng(0).normal(0, 1, 50) > 0).astype(int)
    p = PlattScaler().fit(s, y).transform(s)
    assert np.all(np.diff(p) >= 0) and 0 < p.min() < p.max() < 1


def test_slices_match_ablation_layout(demo):
    cfg, man = demo
    data = load_all(cfg, man, "clip", "clean")
    sl = list(iter_slices(data["test"], data["s4"], seed=0))
    scen = {s for s, *_ in sl}
    assert scen == {"S1", "S2", "S3", "S4"}
    s1 = [idx for s, lvl, g, src, idx in sl if s == "S1"][0]
    assert all(data["test"].scenario[i] is None for i in s1)
    per_gen = {g: idx for s, lvl, g, src, idx in sl if s == "S4" and g != "all"}
    reals = [set(i for i in idx if data["s4"].y[i] == 0) for idx in per_gen.values()]
    assert not set.intersection(*reals)                                   # disjoint Real parts per generator
    for g, idx in per_gen.items():
        assert {data["s4"].generator[i] for i in idx if data["s4"].y[i] == 1} == {g}


def test_train_variant_controls_augmentation(demo):
    cfg, man = demo
    clean = load_all(cfg, man, "dinov2", "clean")["train"]
    aug = load_all(cfg, man, "dinov2", "aug")["train"]
    assert all(s is None for s in clean.scenario)
    assert len(aug) > len(clean) and {s for s in aug.scenario if s} == {"train_aug"}


def test_concatenated_features(demo):
    cfg, man = demo
    a = load_all(cfg, man, "clip", "clean")["val"].X.shape[1]
    b = load_all(cfg, man, "dinov2", "clean")["val"].X.shape[1]
    assert load_all(cfg, man, "clip+dinov2", "clean")["val"].X.shape[1] == a + b


def test_run_comparison_end_to_end(demo, tmp_path):
    cfg, man = demo
    out = run_comparison(cfg, man, ["dinov2"], ["clean"], FAST_MODELS[1:], out_dir=tmp_path, B=50)
    models = {r["model"] for r in out["results"]}
    assert models == set(FAST_MODELS)                                      # reference added automatically
    assert REFERENCE not in {r["model"] for r in out["significance"]}
    assert all(set(RESULT_COLUMNS) <= set(r) for r in out["results"])
    for f in ("results.csv", "significance.csv", "tuning.csv", "timing.csv", "summary.md"):
        assert (tmp_path / f).exists()
    assert "S1 clean" in (tmp_path / "summary.md").read_text()


def test_results_deterministic(demo):
    cfg, man = demo
    data = load_all(cfg, man, "clip", "clean")
    a = compare_on_embeddings(cfg, data, ["logreg", "random_forest"], B=0)["results"]
    b = compare_on_embeddings(cfg, data, ["logreg", "random_forest"], B=0)["results"]
    assert [r["auroc"] for r in a] == [r["auroc"] for r in b]


def test_stacker_on_feature_set():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 12)).astype(np.float32)
    y = (X[:, 9] + X[:, 10] > 0).astype(int)
    fs = FeatureSet(X, np.zeros((120, 3)), np.zeros(120))
    for name in ("logreg", "random_forest", "linear_svm", "knn"):
        p = SklearnStacker(name).fit(fs, y).predict(fs)
        assert p.shape == (120,) and np.all((p >= 0) & (p <= 1))


def test_fusion_comparison_runs(demo):
    cfg, _ = demo
    rows = run_fusion_comparison(cfg, "aug", ["logreg", "decision_tree"])
    methods = {r["method"] for r in rows}
    assert {"static_lr", "aegis", "stack_logreg", "stack_decision_tree"} <= methods
    assert {r["scenario"] for r in rows} == {"S1", "S2", "S3", "S4"}


def test_summary_table_shape():
    rows = [{"head_variant": "clean", "feature_set": "clip", "model": "m", "scenario": "S1", "level": "clean",
             "generator": "all", "auroc": 0.9, "accuracy": 0.8}]
    lines = summary_table(rows)
    assert lines[0].startswith("| head") and "0.900" in lines[2]
