import copy

import numpy as np
import pytest

from src.core.config import load_config
from src.learning.baselines import NaiveAverage, NaiveAverageLogit, StaticLearnedWeights
from src.learning.features import FeaturePipeline
from src.learning.fewshot import AdaptedModel, adapt, fit_few_shot_weights, select_support
from src.learning.fusion import fuse, get_fusion, sigmoid, to_logit
from src.learning.gating import GatingController
from src.learning.synthetic import make_synthetic_bundle


def _cfg(fusion=None):
    c = load_config()
    c["gating_train"].update({"epochs": 30, "lr": 0.02, "batch_size": 512})
    if fusion is None:
        c["gating"].pop("fusion", None)
    else:
        c["gating"]["fusion"] = fusion
    return c


@pytest.fixture(scope="module")
def fit_b():
    return make_synthetic_bundle(400, seed=1)


def test_fusion_default_and_validation():
    assert get_fusion(_cfg(None)) == "prob"
    assert get_fusion(_cfg("logit")) == "logit"
    with pytest.raises(ValueError):
        get_fusion(_cfg("bogus"))


def test_prob_fusion_regression_identical_to_old_behaviour(fit_b):
    """Missing key and explicit 'prob' give the same trained gating, and predict == sum w_i P_i (old formula)."""
    outs = []
    for f in (None, "prob"):
        c = _cfg(f)
        fs = FeaturePipeline(c).fit(fit_b).transform(fit_b)
        g = GatingController(c, seed=0).fit(fs, fit_b.label)
        assert g.fusion == "prob"
        outs.append((g.weights(fs), g.predict(fs), fs))
    assert np.array_equal(outs[0][0], outs[1][0])
    assert np.allclose(outs[0][1], (outs[0][0] * outs[0][2].p).sum(1))
    # the static / naive baselines are unchanged under prob
    fs = outs[0][2]
    assert np.allclose(NaiveAverage().predict(fs), fs.p.mean(1))
    st = StaticLearnedWeights().fit(fs, fit_b.label)
    assert np.allclose(st.predict(fs), st.lr.predict_proba(fs.p)[:, 1])


def test_logit_fusion_formula_and_weight_constraints(fit_b):
    c = _cfg("logit")
    pipe = FeaturePipeline(c).fit(fit_b)
    fs = pipe.transform(fit_b)
    assert np.allclose(fs.z, fit_b.logits / pipe.calibrator.T[None, :])
    assert np.allclose(sigmoid(fs.z), fs.p)
    g = GatingController(c, seed=0).fit(fs, fit_b.label)
    w = g.weights(fs)
    assert np.allclose(w.sum(1), 1.0, atol=1e-5) and w.min() >= 0.05 - 1e-6 and w.max() <= 0.90 + 1e-6
    assert np.allclose(g.predict(fs), sigmoid((w * fs.z).sum(1)))
    assert not np.allclose(g.predict(fs), (w * fs.p).sum(1))
    # logit fusion learns something sensible
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(fit_b.label, g.predict(fs)) > 0.9
    # z subsets / concat stay aligned
    assert np.array_equal(fs.subset(np.arange(5)).z, fs.z[:5]) and fs.concat(fs).z.shape == (800, 3)


def test_fuse_logit_without_z_falls_back_to_p():
    p = np.array([[0.9, 0.2, 0.6]])
    w = np.array([[0.5, 0.25, 0.25]])
    assert np.allclose(fuse(w, p, None, "logit"), sigmoid((w * to_logit(p)).sum(1)), atol=1e-9)


def test_fewshot_logit_three_regimes(fit_b):
    c = _cfg("logit")
    pipe = FeaturePipeline(c).fit(fit_b)
    fit_fs = pipe.transform(fit_b)
    base = GatingController(c, seed=0).fit(fit_fs, fit_b.label)
    pool = make_synthetic_bundle(200, seed=3, unseen=True, clean_only=True, tag="sup")
    ev_fs = pipe.transform(make_synthetic_bundle(80, seed=4, unseen=True, clean_only=True, tag="ev"))
    for n, regime in ((5, "conservative"), (20, "anchor"), (100, "refit")):
        sup = select_support(pool, n, 0)
        res = adapt(base, fit_fs, fit_b.label, pipe.transform(sup), sup.label, c, seed=0)
        assert res.regime == regime and res.model.fusion == "logit"
        w = res.model.weights(ev_fs)
        assert np.allclose(res.model.predict(ev_fs), sigmoid((w * ev_fs.z).sum(1)))
        assert w.min() >= 0.05 - 1e-6 and w.max() <= 0.90 + 1e-6 and np.allclose(w.sum(1), 1, atol=1e-5)
        if regime == "anchor":
            expect = (1 - res.alpha) * base.weights(ev_fs) + res.alpha * res.w_few_shot[None, :]
            assert np.allclose(w, expect)
        if regime == "conservative":
            assert np.array_equal(w, base.weights(ev_fs))


def test_grid_search_uses_requested_fusion():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 60)
    z = np.stack([(2 * y - 1) * 3.0 + rng.normal(0, 1, 60), rng.normal(0, 3, 60), rng.normal(0, 3, 60)], 1)
    p = sigmoid(z)
    w_ref = np.array([1 / 3] * 3)
    w_logit = fit_few_shot_weights(p, y, w_ref, 0.05, 0.85, fusion="logit", z=z)
    w_prob = fit_few_shot_weights(p, y, w_ref, 0.05, 0.85)
    assert abs(w_logit.sum() - 1) < 1e-9 and w_logit.min() >= 0.05 - 1e-9 and w_logit.max() <= 0.90 + 1e-9
    assert w_logit[0] > 0.5 and w_prob[0] > 0.5


def test_static_and_naive_logit_baselines(fit_b):
    c = _cfg("logit")
    fs = FeaturePipeline(c).fit(fit_b).transform(fit_b)
    st = StaticLearnedWeights(fusion="logit").fit(fs, fit_b.label)
    assert np.allclose(st.predict(fs), st.lr.predict_proba(fs.z)[:, 1])
    assert np.allclose(NaiveAverageLogit().predict(fs), sigmoid(fs.z.mean(1)))
    assert np.allclose(NaiveAverage().predict(fs), fs.p.mean(1))      # naive_avg stays the probability mean
