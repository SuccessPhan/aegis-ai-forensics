import numpy as np
import pytest

from src.core.config import load_config
from src.core.contracts import (EXPERT_NAMES, FLAG_CONFLICT, FLAG_OOD, FLAG_UNCERTAIN, GATING_DIM, GatingInput,
                                QualityFeatures)
from src.evaluation.metrics import compute_metrics
from src.learning.baselines import NaiveAverage, SingleExpert, StaticLearnedWeights
from src.learning.bundle_io import BundleError, bundle_from_dict, load_bundle, save_bundle, validate_bundle
from src.learning.calibration import TemperatureCalibrator, nll
from src.learning.features import FeaturePipeline, predictive_entropy
from src.learning.fewshot import adapt, select_support
from src.learning.flags import compute_flags, fit_ood_threshold
from src.learning.gating import GatingController
from src.learning.synthetic import make_synthetic_bundle


@pytest.fixture(scope="module")
def cfg():
    c = load_config()
    c["gating_train"].update({"epochs": 100, "lr": 0.02, "batch_size": 512})   # test-speed override
    return c


@pytest.fixture(scope="module")
def fit_b():
    return make_synthetic_bundle(500, seed=1)


@pytest.fixture(scope="module")
def trained(cfg, fit_b):
    pipe = FeaturePipeline(cfg).fit(fit_b)
    fs = pipe.transform(fit_b)
    return pipe, fs, GatingController(cfg, seed=0).fit(fs, fit_b.label)


def test_bundle_roundtrip_and_validation(tmp_path):
    b = make_synthetic_bundle(20, seed=3)
    save_bundle(b, tmp_path / "val_gating_fit.pt")
    b2 = load_bundle(tmp_path / "val_gating_fit.pt")
    assert len(b2) == 20 and b2.logits.shape == (20, 3) and b2.degradation == b.degradation
    import torch
    d = {"sha256": b.sha256, "label": torch.as_tensor(b.label), "generator": b.generator,
         "degradation": b.degradation, "scenario": b.scenario, "logits": torch.zeros(20, 2),
         "cosdist": torch.zeros(20, 3), "quality": torch.zeros(20, 3)}
    with pytest.raises(BundleError, match="logits shape"):
        validate_bundle(d)
    d.pop("quality")
    with pytest.raises(BundleError, match="missing key"):
        bundle_from_dict(d)


def test_temperature_calibration_and_io(tmp_path):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    logits = (2 * y - 1) * 1.0 + rng.normal(0, 1.5, 2000)
    over = np.stack([logits * 4, logits, logits / 4], 1)        # over-confident / ok / under-confident
    cal = TemperatureCalibrator().fit(over, y)
    assert cal.T[0] > 2.5 and 0.7 < cal.T[1] < 1.4 and cal.T[2] < 0.5
    assert nll(over[:, 0], y, cal.T[0]) < nll(over[:, 0], y, 1.0)
    cal.save(tmp_path / "T.json")
    assert np.allclose(TemperatureCalibrator.load(tmp_path / "T.json").T, cal.T)


def test_fit_split_guard_and_feature_order(cfg, fit_b):
    with pytest.raises(ValueError):
        FeaturePipeline(cfg).fit(fit_b, split="test_benchmark")
    pipe = FeaturePipeline(cfg).fit(fit_b)
    fs = pipe.transform(fit_b)
    assert fs.X.shape == (500, GATING_DIM)
    assert np.allclose(fs.X.mean(0), 0, atol=1e-4)
    raw = pipe.raw_matrix(fit_b)[0]
    gi = GatingInput.from_vector(raw)
    assert np.allclose(gi.quality.to_array(), fit_b.quality[0], atol=1e-5)
    assert np.allclose(gi.cosine_distance, fit_b.cosdist[0], atol=1e-5)
    assert np.allclose(gi.p_cal, fs.p[0], atol=1e-5)
    assert np.allclose(gi.entropy, predictive_entropy(fs.p[0]), atol=1e-5)
    noood = FeaturePipeline(cfg, use_ood=False).fit(fit_b).transform(fit_b)
    assert np.all(noood.X[:, :9] == 0)


def test_gating_weight_bounds(cfg, trained):
    _, fs, g = trained
    w = g.weights(fs)
    assert np.allclose(w.sum(1), 1.0, atol=1e-5)
    assert w.min() >= 0.05 - 1e-6 and w.max() <= 0.90 + 1e-6
    assert np.allclose(g.predict(fs), (w * fs.p).sum(1))


def test_aegis_beats_naive_when_expert_breaks_on_bad_quality(cfg, trained):
    pipe, fs_fit, g = trained
    test = make_synthetic_bundle(600, seed=99)
    fs = pipe.transform(test)
    aegis = compute_metrics(test.label, g.predict(fs), cfg)
    naive = compute_metrics(test.label, NaiveAverage().predict(fs), cfg)
    static = compute_metrics(test.label, StaticLearnedWeights().fit(fs_fit, np.asarray(make_synthetic_bundle(500, seed=1).label)).predict(fs), cfg)
    assert aegis.accuracy > naive.accuracy + 0.02 and aegis.auroc > naive.auroc
    assert aegis.accuracy > static.accuracy          # per-sample routing beats the fixed learned weights
    bad = np.array([d is not None for d in test.degradation])
    w = g.weights(fs)
    assert w[bad, 2].mean() < w[~bad, 2].mean() - 0.03   # freq down-weighted on bad quality
    assert SingleExpert("freq").predict(fs).shape == (600,)


def test_fewshot_regimes(cfg, trained):
    pipe, fs_fit, g = trained
    fit_y = make_synthetic_bundle(500, seed=1).label
    pool = make_synthetic_bundle(300, seed=5, unseen=True, clean_only=True)
    base_w = g.weights(fs_fit).copy()

    def run(n):
        sup = select_support(pool, n, seed=0)
        assert sup.label.sum() == (n + 1) // 2 and (sup.label == 0).sum() == n // 2
        return adapt(g, fs_fit, fit_y, pipe.transform(sup), sup.label, cfg, seed=0)

    r5 = run(5)
    assert r5.regime == "conservative" and r5.alpha == 0.0 and "scarcity" in r5.warning
    assert np.allclose(r5.model.weights(fs_fit), base_w)
    r20 = run(20)
    assert r20.regime == "anchor" and r20.alpha == pytest.approx(0.4) and r20.source == "adapted"
    w20 = r20.model.weights(fs_fit)
    assert np.allclose(w20.sum(1), 1) and w20.min() >= 0.05 - 1e-6 and w20.max() <= 0.9 + 1e-6
    assert np.allclose(g.weights(fs_fit), base_w)                      # base untouched
    assert run(50).alpha == pytest.approx(0.6)
    r100 = run(100)
    assert r100.regime == "refit" and r100.model is not g
    assert np.allclose(g.weights(fs_fit), base_w)
    # support is nested across N and disjoint from the pool's other members
    assert set(select_support(pool, 5, 0).sha256) <= set(select_support(pool, 20, 0).sha256)


def test_flags(cfg):
    p = np.array([[0.9, 0.1, 0.5],     # conflict (std 0.33)
                  [0.9, 0.6, 0.7],     # none
                  [0.55, 0.5, 0.45],   # uncertain
                  [0.9, 0.85, 0.95]])  # ood only
    pf = np.array([0.5, 0.75, 0.5, 0.9])
    ood = np.array([0.1, 0.1, 0.1, 5.0])
    flags, why = compute_flags(p, pf, ood, 1.0, cfg)
    assert flags.tolist() == [True, False, True, True]
    assert FLAG_CONFLICT in why[0] and FLAG_UNCERTAIN in why[0] and why[1] == []
    assert why[2] == [FLAG_UNCERTAIN] and why[3] == [FLAG_OOD]
    assert fit_ood_threshold(np.arange(101.0), cfg) == pytest.approx(95.0)
