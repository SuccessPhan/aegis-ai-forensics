import csv
import json

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.core.config import load_config
from src.core.contracts import EXPERT_NAMES
from src.evaluation import ablation as ab
from src.evaluation import significance as sig
from src.evaluation import weights_analysis as wa
from src.evaluation.predictions import PredictionSink, load_predictions
from src.learning.synthetic import make_synthetic_bundle


@pytest.fixture(scope="module")
def cfg():
    c = load_config()
    c["gating_train"].update({"epochs": 6, "lr": 0.01, "batch_size": 512})
    c["fewshot"]["n_values"] = [5, 20]
    return c


def _bundles(seed=1):
    return (make_synthetic_bundle(200, seed=seed), make_synthetic_bundle(150, seed=seed + 1, scenarios=True),
            make_synthetic_bundle(200, seed=seed + 2, unseen=True, clean_only=True, tag="sup"),
            make_synthetic_bundle(100, seed=seed + 3, unseen=True, clean_only=True, tag="ev"))


@pytest.fixture(scope="module")
def runs(cfg):
    out = []
    for _ in range(2):
        fit, test, sup, ev = _bundles()
        sink = PredictionSink()
        rows = ab.run_ablation(cfg, fit, test, sup, ev, s4_seeds=[0, 1], head_variant="clean", sink=sink)
        out.append((rows, sink))
    return out


def test_predictions_deterministic_and_match_metrics(runs, tmp_path):
    (r1, s1), (r2, s2) = runs
    key = lambda r: (r["scenario"], r["level"], r["generator"], r["N"], r["method"], r["variant"])   # noqa: E731
    m1 = {key(r): (r["accuracy"], r["auroc"], r["ece"]) for r in r1}
    m2 = {key(r): (r["accuracy"], r["auroc"], r["ece"]) for r in r2}
    assert m1.keys() == m2.keys()
    for k in m1:
        assert np.allclose(m1[k], m2[k], equal_nan=True, rtol=0, atol=0), k
    assert s1.groups.keys() == s2.groups.keys()
    for g in s1.groups:
        for f, v in s1.groups[g].items():
            assert np.array_equal(v, s2.groups[g][f]), (g, f)
    path = s1.save(tmp_path / "predictions_clean.npz")
    back = load_predictions(path)
    for g in s1.groups:
        for f, v in s1.groups[g].items():
            assert np.array_equal(v, back[g][f])
    # stored predictions reproduce the table
    g = back["full::S1::clean::-::-::-"]
    row = next(r for r in r1 if r["scenario"] == "S1" and r["method"] == "aegis" and r["variant"] == "full")
    assert roc_auc_score(g["y"], g["p_aegis"]) == pytest.approx(row["auroc"])
    assert g["w_aegis"].shape == (len(g["y"]), 3) and np.allclose(g["w_aegis"].sum(1), 1)


def test_weighted_auc_matches_sklearn_with_ties():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p = np.round(rng.random(300) * (0.3 * y + 1), 1)                    # many ties
    C = np.ones((1, 300))
    assert sig._auc_weighted(C, y, p)[0] == pytest.approx(roc_auc_score(y, p))
    cnt = rng.multinomial(300, np.ones(300) / 300)[None, :].astype(float)
    idx = np.repeat(np.arange(300), cnt[0].astype(int))
    assert sig._auc_weighted(cnt, y, p)[0] == pytest.approx(roc_auc_score(y[idx], p[idx]))


def test_bootstrap_paired_detects_difference_and_null():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 600)
    good = np.clip(0.5 + 0.35 * (2 * y - 1) + rng.normal(0, 0.25, 600), 0, 1)
    bad = np.clip(0.5 + 0.10 * (2 * y - 1) + rng.normal(0, 0.25, 600), 0, 1)
    boot = sig.bootstrap_metrics(y, {"a": good, "b": bad, "c": good.copy()}, B=300, seed=3)
    s = sig.summarize_delta(boot["a"]["auroc"] - boot["b"]["auroc"], 0.1)
    assert s["significant"] and s["ci_lo"] > 0 and s["p_value"] < 0.05
    s0 = sig.summarize_delta(boot["a"]["auroc"] - boot["c"]["auroc"], 0.0)       # identical predictions
    assert s0["ci_lo"] == 0 == s0["ci_hi"] and s0["p_value"] == 1.0 and not s0["significant"]
    again = sig.bootstrap_metrics(y, {"a": good}, B=300, seed=3)
    assert np.array_equal(again["a"]["auroc"], boot["a"]["auroc"])               # fixed seed -> reproducible


def test_significance_cli_end_to_end(runs, tmp_path):
    (_, sink), _ = runs
    sink.save(tmp_path / "pred" / "predictions_clean.npz")
    assert sig.main(["--pred", str(tmp_path / "pred"), "--out", str(tmp_path / "out"), "--B", "60"]) == 0
    with open(tmp_path / "out" / "significance.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert {r["baseline"] for r in rows} >= {"clip", "dinov2", "freq", "naive_avg", "static_lr", "quality_only",
                                             "static_learned_support", "aegis_noadapt"}
    s4 = [r for r in rows if r["scenario"] == "S4" and r["N"] == "20" and r["generator"] == "all"]
    assert s4 and all(r["n_seeds"] == "2" for r in s4)
    # S4 point delta = mean over seeds of per-seed deltas, equal to the ablation-table difference
    md = (tmp_path / "out" / "significance.md").read_text(encoding="utf-8")
    assert "strongest baseline" in md and "| S2 |" in md


def test_weights_analysis(cfg, runs, tmp_path):
    fit, _, _, _ = _bundles()
    ver = make_synthetic_bundle(120, seed=9, scenarios=True)
    params = {s: {"jpeg_quality": None, "blur_sigma": None, "resize_factor": None} for s in ver.sha256}
    rng = np.random.default_rng(0)
    for i, s in enumerate(ver.sha256):
        if i % 3 == 0:
            params[s]["jpeg_quality"] = int(rng.integers(30, 96))
        elif i % 3 == 1:
            params[s]["blur_sigma"] = float(rng.uniform(0.5, 3.0))
    (_, sink), _ = runs
    rows = wa.analyse(cfg, "clean", fit, ver, params, {k: v for k, v in sink.groups.items()})
    wa.write_outputs(rows, tmp_path)
    for f in ("weights_by_jpeg.png", "weights_by_blur.png", "weights_summary.csv", "weights_summary.md"):
        assert (tmp_path / f).stat().st_size > 0
    jp = [r for r in rows if r["axis"] == "jpeg" and r["split"] == "val_gating_verify"]
    assert [r["bin"] for r in jp][-2:] == ["90-95", "clean"] and sum(r["n"] for r in jp[:-1]) == 40
    assert all(abs(sum(r[f"w_{e}_mean"] for e in EXPERT_NAMES) - 1) < 1e-6 for r in rows if r["n"])
    assert {r["bin"] for r in rows if r["split"] == "test_benchmark"} >= {"clean", "jpeg_q40"}
    assert wa._bin_index(95, wa.JPEG_EDGES) == 6 and wa._bin_index(3.0, wa.BLUR_EDGES) == 4
