"""Machine learning vs the AEGIS image models: hand-crafted features (src.data.handcrafted), the three models
(src.learning.classic_models) and the comparison runner (src.evaluation.ml_comparison)."""
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score

from src.data.handcrafted import FEATURE_NAMES, N_FEATURES, build_cache, extract_handcrafted, open_cache
from src.evaluation.ml_comparison import (DEEP_METHODS, DEEP_SET, FEATURE_SETS, REFERENCE, RESULT_COLUMNS,
                                          align_predictions, compare_on_embeddings, deep_predictions, iter_slices,
                                          load_all, run_comparison, run_fusion_comparison, summary_table)
from src.learning.classic_models import (DEFAULT_MODELS, MODEL_SPECS, PlattScaler, SklearnStacker, build_model,
                                         raw_score, tune)
from src.learning.features import FeatureSet

ROOT_DEMO = "test_demo_cache"


@pytest.fixture(scope="module")
def xy():
    X, y = make_classification(n_samples=360, n_features=40, n_informative=8, random_state=0)
    return X[:200], y[:200], X[200:280], y[200:280], X[280:], y[280:]


@pytest.fixture(scope="module")
def demo(cfg):
    from tests.conftest import ROOT
    from src.models.demo_features import build_demo
    c, man = build_demo(ROOT / "data" / ROOT_DEMO, scale=2, cfg=cfg)
    build_cache(c, man)
    return c, man


# ---------------------------------------------------------------------------------------- hand-crafted features
def test_handcrafted_vector_shape_and_names():
    img = np.random.default_rng(0).integers(0, 256, (224, 224, 3), dtype=np.uint8)
    f = extract_handcrafted(img)
    assert f.shape == (N_FEATURES,) == (len(FEATURE_NAMES),) and f.dtype == np.float32
    assert np.isfinite(f).all()
    assert len(set(FEATURE_NAMES)) == N_FEATURES


def test_handcrafted_is_deterministic_and_reacts_to_blur():
    from PIL import Image, ImageFilter
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    np.testing.assert_array_equal(extract_handcrafted(img), extract_handcrafted(img.copy()))
    blurred = np.asarray(Image.fromarray(img).filter(ImageFilter.GaussianBlur(2)))
    a, b = extract_handcrafted(img), extract_handcrafted(blurred)
    lap = FEATURE_NAMES.index("grad_lap_var")
    assert b[lap] < a[lap]                                     # blur removes high-frequency energy


def test_handcrafted_constant_image_is_finite():
    f = extract_handcrafted(np.full((224, 224, 3), 128, dtype=np.uint8))
    assert np.isfinite(f).all()


def test_handcrafted_srm_kernels_match_frequency_expert():
    from src.data.handcrafted import _SRM_KERNELS
    from src.models import freq_expert
    for a, b in zip(_SRM_KERNELS, freq_expert._SRM):
        np.testing.assert_allclose(a, np.asarray(b))


def test_handcrafted_cache_covers_manifest_and_resumes(demo):
    cfg, man = demo
    cache = open_cache(cfg)
    assert len(cache) == len(man.records)
    assert cache.meta["feature_names"] == FEATURE_NAMES
    assert build_cache(cfg, man) is not None                    # second call: nothing to compute


# ------------------------------------------------------------------------------------------------- the models
def test_exactly_three_models():
    assert DEFAULT_MODELS == ("rbf_svm", "random_forest", "xgboost")
    assert set(MODEL_SPECS) == set(DEFAULT_MODELS)
    assert {MODEL_SPECS[m].family for m in DEFAULT_MODELS} == {"kernel", "bagging", "boosting"}


@pytest.mark.parametrize("name", list(DEFAULT_MODELS))
def test_every_model_fits_and_beats_chance(name, xy, cfg):
    Xtr, ytr, Xva, yva, Xte, yte = xy
    tm = tune(name, Xtr, ytr, Xva, yva, seed=0, cfg=cfg)
    p = tm.predict_proba1(Xte)
    assert p.shape == (len(Xte),) and np.all((p >= 0) & (p <= 1))
    assert roc_auc_score(yte, p) > 0.6, name
    assert len(tm.trials) == int(np.prod([len(v) for v in MODEL_SPECS[name].grid.values()]))
    assert tm.val_auroc == pytest.approx(max(t["val_auroc"] for t in tm.trials))


def test_pca_only_for_wide_inputs():
    narrow = build_model("rbf_svm", 300, 92)
    wide = build_model("rbf_svm", 300, 768)
    assert narrow.named_steps["pca"] == "passthrough"
    assert wide.named_steps["pca"].n_components == 128
    X, y = make_classification(n_samples=60, n_features=300, random_state=1)
    build_model("xgboost", len(X), X.shape[1]).fit(X, y)       # PCA clamped to the sample count


def test_unknown_model_rejected():
    with pytest.raises(KeyError):
        build_model("knn", 10, 5)


def test_platt_scaler_is_monotone():
    s = np.linspace(-3, 3, 50)
    y = (s + np.random.default_rng(0).normal(0, 1, 50) > 0).astype(int)
    p = PlattScaler().fit(s, y).transform(s)
    assert np.all(np.diff(p) >= 0) and 0 < p.min() < p.max() < 1


def test_raw_score_uses_decision_function_or_logit(xy):
    Xtr, ytr, *_ = xy
    svm = build_model("rbf_svm", len(Xtr), Xtr.shape[1]).fit(Xtr, ytr)
    rf = build_model("random_forest", len(Xtr), Xtr.shape[1]).fit(Xtr, ytr)
    np.testing.assert_allclose(raw_score(svm, Xtr), svm.decision_function(Xtr))
    assert np.isfinite(raw_score(rf, Xtr)).all()


# ------------------------------------------------------------------------------------------------- the runner
def test_slices_match_ablation_layout(demo):
    cfg, man = demo
    data = load_all(cfg, man, "handcrafted", "clean")
    sl = list(iter_slices(data["test"], data["s4"], seed=0))
    assert {s for s, *_ in sl} == {"S1", "S2", "S3", "S4"}
    s1 = [idx for s, lvl, g, src, idx in sl if s == "S1"][0]
    assert all(data["test"].scenario[i] is None for i in s1)
    per_gen = {g: idx for s, lvl, g, src, idx in sl if s == "S4" and g != "all"}
    reals = [set(i for i in idx if data["s4"].y[i] == 0) for idx in per_gen.values()]
    assert not set.intersection(*reals)
    for g, idx in per_gen.items():
        assert {data["s4"].generator[i] for i in idx if data["s4"].y[i] == 1} == {g}


def test_feature_sets_and_variants(demo):
    cfg, man = demo
    hc = load_all(cfg, man, "handcrafted", "clean")
    assert hc["train"].X.shape[1] == N_FEATURES
    assert all(s is None for s in hc["train"].scenario)
    aug = load_all(cfg, man, "handcrafted", "aug")["train"]
    assert len(aug) > len(hc["train"]) and {s for s in aug.scenario if s} == {"train_aug"}
    a = load_all(cfg, man, "clip", "clean")["val"].X.shape[1]
    b = load_all(cfg, man, "dinov2", "clean")["val"].X.shape[1]
    assert load_all(cfg, man, "clip+dinov2", "clean")["val"].X.shape[1] == a + b
    assert FEATURE_SETS[0] == "handcrafted"
    with pytest.raises(ValueError):
        load_all(cfg, man, "resnet", "clean")


def test_deep_predictions_cover_test_and_s4(demo):
    cfg, man = demo
    by_sha = deep_predictions(cfg, "clean")
    assert set(by_sha) == {"test", "s4"}
    assert set(by_sha["test"]) == set(DEEP_METHODS)
    data = load_all(cfg, man, "handcrafted", "clean")
    al = align_predictions(by_sha, data)
    for key in ("test", "s4"):
        for m, p in al[key].items():
            assert p.shape == (len(data[key]),) and np.all((p >= 0) & (p <= 1)), m


def test_run_comparison_end_to_end(demo, tmp_path):
    cfg, man = demo
    out = run_comparison(cfg, man, ["handcrafted", "dinov2"], ["clean"], out_dir=tmp_path, B=50)
    res = out["results"]
    assert all(set(RESULT_COLUMNS) <= set(r) for r in res)
    deep = {r["model"] for r in res if r["group"] == "deep"}
    ml = {(r["feature_set"], r["model"]) for r in res if r["group"] == "ml"}
    assert deep == set(DEEP_METHODS)
    assert {r["feature_set"] for r in res if r["group"] == "deep"} == {DEEP_SET}
    assert ml == {(fs, m) for fs in ("handcrafted", "dinov2") for m in DEFAULT_MODELS}
    s1_deep = [r for r in res if r["group"] == "deep" and r["scenario"] == "S1"]
    assert len(s1_deep) == len(DEEP_METHODS)                  # image models written once, not per feature set
    sig = out["significance"]
    assert sig and {r["model"] for r in sig} == set(DEFAULT_MODELS) and REFERENCE not in {r["model"] for r in sig}
    for f in ("results.csv", "significance.csv", "tuning.csv", "timing.csv", "summary.md"):
        assert (tmp_path / f).exists()
    md = (tmp_path / "summary.md").read_text()
    assert "image_models" in md and "handcrafted" in md and "S1 clean" in md


def test_results_deterministic(demo):
    cfg, man = demo
    data = load_all(cfg, man, "handcrafted", "clean")
    a = compare_on_embeddings(cfg, data, ["random_forest", "xgboost"], B=0)["results"]
    b = compare_on_embeddings(cfg, data, ["random_forest", "xgboost"], B=0)["results"]
    assert [r["auroc"] for r in a] == [r["auroc"] for r in b]


def test_stacker_on_feature_set():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 12)).astype(np.float32)
    y = (X[:, 9] + X[:, 10] > 0).astype(int)
    fs = FeatureSet(X, np.zeros((120, 3)), np.zeros(120))
    for name in DEFAULT_MODELS:
        p = SklearnStacker(name).fit(fs, y).predict(fs)
        assert p.shape == (120,) and np.all((p >= 0) & (p <= 1))


def test_fusion_comparison_runs(demo):
    cfg, _ = demo
    rows = run_fusion_comparison(cfg, "aug", ["random_forest"])
    methods = {r["method"] for r in rows}
    assert {"static_lr", "aegis", "stack_random_forest"} <= methods
    assert {r["scenario"] for r in rows} == {"S1", "S2", "S3", "S4"}


def test_summary_table_puts_image_models_first():
    rows = [{"head_variant": "clean", "feature_set": "handcrafted", "model": "xgboost", "scenario": "S1",
             "level": "clean", "generator": "all", "auroc": 0.8, "accuracy": 0.7},
            {"head_variant": "clean", "feature_set": DEEP_SET, "model": "aegis", "scenario": "S1",
             "level": "clean", "generator": "all", "auroc": 0.9, "accuracy": 0.8}]
    lines = summary_table(rows)
    assert lines[0].startswith("| head") and "aegis" in lines[2] and "0.900" in lines[2]
