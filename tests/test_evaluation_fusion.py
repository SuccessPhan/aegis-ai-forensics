import csv
import json

import numpy as np
import pytest

from src.core.config import load_config
from src.evaluation import select_fusion as sf
from src.evaluation.ablation import run_ablation
from src.evaluation.predictions import PredictionSink, load_meta, load_predictions
from src.learning.synthetic import make_synthetic_bundle


def _cfg(fusion=None):
    c = load_config()
    c["gating_train"].update({"epochs": 10, "lr": 0.01, "batch_size": 512})
    if fusion:
        c["gating"]["fusion"] = fusion
    return c


def _verify_bundle(n=600, seed=7):
    """Synthetic verify bundle with clean / jpeg / blur degradations (cluster ids via a parent map)."""
    b = make_synthetic_bundle(n, seed=seed, tag="v")
    b.degradation = [None if i % 3 == 0 else ("jpeg_q57" if i % 3 == 1 else "blur_s1.5") for i in range(n)]
    return b


def _fake_fit_predict(better):
    """prob = weak noisy scores; logit = either clearly better ('logit') or clearly worse ('prob')."""
    def f(cfg, fusion, fit, verify):
        rng = np.random.default_rng(1 if fusion == "prob" else 2)
        y = verify.label
        good = np.clip(0.5 + (2 * y - 1) * 0.3 + rng.normal(0, 0.12, len(y)), 0.01, 0.99)
        bad = np.clip(0.5 + (2 * y - 1) * 0.05 + rng.normal(0, 0.25, len(y)), 0.01, 0.99)
        p = good if fusion == better else bad
        return {"aegis": p, "static_lr": p}
    return f


def test_selects_logit_when_clearly_better(monkeypatch):
    monkeypatch.setattr(sf, "_fit_predict", _fake_fit_predict("logit"))
    cfg = _cfg()
    cfg["evaluation"]["champion_challenger"]["max_ece_increase"] = 1.0
    v = _verify_bundle()
    r = sf.select_for_variant(cfg, v, v, B=200)
    assert r["selected"] == "logit" and r["auroc_criterion"]
    assert {x["slice"] for x in r["rows"]} == {"clean", "jpeg", "blur", "overall"}


def test_selects_prob_when_logit_worse_or_equal(monkeypatch):
    cfg = _cfg()
    v = _verify_bundle()
    monkeypatch.setattr(sf, "_fit_predict", _fake_fit_predict("prob"))
    assert sf.select_for_variant(cfg, v, v, B=200)["selected"] == "prob"
    monkeypatch.setattr(sf, "_fit_predict", lambda c, f, a, b: {"aegis": b.label * 0.4 + 0.3, "static_lr": b.label * 0.4 + 0.3})
    r = sf.select_for_variant(cfg, v, v, B=200)            # identical predictions: delta 0, CI contains 0
    assert r["selected"] == "prob"


def test_ece_guard_blocks_logit(monkeypatch):
    """AUROC clearly better but ECE worse than the allowed increase -> prob."""
    monkeypatch.setattr(sf, "_fit_predict", _fake_fit_predict("logit"))
    cfg = _cfg()
    v = _verify_bundle()
    cfg["evaluation"]["champion_challenger"]["max_ece_increase"] = -1.0     # impossible to satisfy
    r = sf.select_for_variant(cfg, v, v, B=200)
    assert r["auroc_criterion"] and not r["ece_criterion"] and r["selected"] == "prob"


def test_cluster_bootstrap_and_outputs(monkeypatch, tmp_path):
    monkeypatch.setattr(sf, "_fit_predict", _fake_fit_predict("logit"))
    cfg = _cfg()
    cfg["evaluation"]["champion_challenger"]["max_ece_increase"] = 1.0
    v = _verify_bundle()
    mani = tmp_path / "m.json"
    mani.write_text(json.dumps({"records": [{"sha256": s, "parent_sha256": None} for s in v.sha256]}))
    monkeypatch.setattr(sf, "load_allowed", lambda cfg_, split, hv=None: v)
    res = sf.run(cfg, ["clean"], tmp_path / "out", manifest=mani, B=100, loader=sf.load_allowed)
    assert res["selected"] == "logit" and res["variants"][0]["clustered"]
    assert json.loads((tmp_path / "out" / "fusion_selection.json").read_text())["selected"] == "logit"
    assert "Selected fusion" in (tmp_path / "out" / "fusion_selection.md").read_text(encoding="utf-8")


def test_guard_refuses_test_benchmark(tmp_path):
    cfg = _cfg()
    with pytest.raises(sf.ForbiddenSplitError):
        sf.load_allowed(cfg, "test_benchmark", "clean")
    for s in ("s4_eval", "s4_support", "train_experts"):
        with pytest.raises(sf.ForbiddenSplitError):
            sf.load_allowed(cfg, s)
    tb = make_synthetic_bundle(40, seed=2, scenarios=True)            # carries S2 tags like test_benchmark
    ok = make_synthetic_bundle(40, seed=3)
    with pytest.raises(sf.ForbiddenSplitError):
        sf.select_for_variant(cfg, ok, tb, B=50)
    with pytest.raises(sf.ForbiddenSplitError):
        sf.select_for_variant(cfg, tb, ok, B=50)
    ok.meta["split"] = "test_benchmark"
    with pytest.raises(sf.ForbiddenSplitError):
        sf.assert_not_test_bundle(ok)


def test_select_fusion_real_fit_smoke(tmp_path):
    """End-to-end on synthetic bundles: both fusions fitted, report written, rule returns a valid choice."""
    cfg = _cfg()
    fit, ver = make_synthetic_bundle(300, seed=1), _verify_bundle(300)
    r = sf.select_for_variant(cfg, fit, ver, B=100)
    assert r["selected"] in ("prob", "logit") and r["rows"]


def test_ablation_records_fusion_and_logit_baseline(tmp_path):
    fit = make_synthetic_bundle(300, seed=1)
    test = make_synthetic_bundle(200, seed=2, scenarios=True)
    sup = make_synthetic_bundle(150, seed=3, unseen=True, clean_only=True, tag="sup")
    ev = make_synthetic_bundle(120, seed=4, unseen=True, clean_only=True, tag="ev")
    for f in ("prob", "logit"):
        rows = run_ablation(_cfg(f), fit, test, sup, ev, out_dir=tmp_path / f, n_values=[5, 20, 100])
        assert {r["fusion"] for r in rows} == {f}
        has_nl = any(r["method"] == "naive_avg_logit" for r in rows)
        assert has_nl == (f == "logit")
        with open(tmp_path / f / "ablation.csv", encoding="utf-8") as fh:
            assert {r["fusion"] for r in csv.DictReader(fh)} == {f}
        assert any(r["method"] == "static_learned_support" for r in rows)
        assert {"conservative", "anchor", "refit"} <= {r.get("regime") for r in rows}


def test_prediction_sink_metadata_roundtrip(tmp_path):
    s = PredictionSink({"fusion": "logit"})
    s.add("full::S1::clean::-::-::-", "aegis", ["a", "b"], [0, 1], np.array([0.2, 0.8]))
    path = s.save(tmp_path / "predictions_clean.npz")
    assert load_meta(path) == {"fusion": "logit"}
    assert list(load_predictions(path)) == ["full::S1::clean::-::-::-"]
