import numpy as np
import pytest

from src.core.config import load_config
from src.core.contracts import EXPERT_NAMES
from src.evaluation.ablation import _System
from src.learning.export import AegisSystem, fit_system
from src.learning.flags import compute_flags
from src.learning.synthetic import make_synthetic_bundle


@pytest.fixture(scope="module")
def setup():
    cfg = load_config()
    fit = make_synthetic_bundle(300, seed=1)
    test = make_synthetic_bundle(120, seed=2, unseen=True, tag="t")
    return cfg, fit, test, fit_system(cfg, "aug", fit=fit)


def test_matches_ablation_system(setup):
    cfg, fit, test, sysm = setup
    ref = _System(cfg, fit, True, True, cfg.get("seed", 0))
    fs = ref.pipe.transform(test)
    p_ref, w_ref = ref.gating.predict(fs), ref.gating.weights(fs)
    flags_ref, reasons_ref = compute_flags(fs.p, p_ref, fs.mean_ood, ref.thr, cfg)
    r = sysm.predict_batch(test.logits, test.cosdist, test.quality)
    assert np.array_equal(r["p_ai"], p_ref)
    assert np.array_equal(r["weights"], w_ref)
    assert np.array_equal(r["warning_flag"], flags_ref)
    assert r["flag_reasons"] == reasons_ref
    assert r["ood_threshold"] == ref.thr
    assert sysm.metadata["n_fit"] == 300 and sysm.metadata["head_variant"] == "aug"


def test_save_load_identical(setup, tmp_path):
    _, _, test, sysm = setup
    path = tmp_path / "s.pt"
    sysm.save(path)
    loaded = AegisSystem.load(path)
    a = sysm.predict_batch(test.logits, test.cosdist, test.quality)
    b = loaded.predict_batch(test.logits, test.cosdist, test.quality)
    for k in ("p_ai", "weights", "p_cal", "ood_score", "warning_flag"):
        assert np.array_equal(a[k], b[k]), k
    assert a["flag_reasons"] == b["flag_reasons"] and a["ood_threshold"] == b["ood_threshold"]
    assert loaded.metadata["config_hash"] == sysm.metadata["config_hash"]
    assert loaded.metadata["loaded_config_hash_ok"] is True


def test_predict_one_matches_batch(setup):
    _, _, test, sysm = setup
    r = sysm.predict_batch(test.logits, test.cosdist, test.quality)
    for i in range(0, len(test), 7):
        d = sysm.predict_one(test.logits[i], test.cosdist[i], test.quality[i])
        assert d["p_ai"] == pytest.approx(r["p_ai"][i], abs=1e-6)
        assert d["label"] == ("AI" if r["p_ai"][i] >= 0.5 else "Real")
        assert d["confidence"] == pytest.approx(max(r["p_ai"][i], 1 - r["p_ai"][i]))
        assert d["warning_flag"] == bool(r["warning_flag"][i]) and d["flag_reasons"] == r["flag_reasons"][i]
        assert list(d["weights"]) == list(EXPERT_NAMES) and list(d["p_cal"]) == list(EXPERT_NAMES)
        assert sum(d["weights"].values()) == pytest.approx(1.0)
        assert all(0.05 - 1e-9 <= v <= 0.90 + 1e-9 for v in d["weights"].values())
        assert d["ood_score"] == pytest.approx(r["ood_score"][i])
        assert set(d) == {"p_ai", "label", "confidence", "warning_flag", "flag_reasons", "weights", "p_cal",
                          "ood_score", "ood_threshold"}
