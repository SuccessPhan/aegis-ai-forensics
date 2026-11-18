import csv

import numpy as np
import pytest

from src.core.config import load_config
from src.evaluation.ablation import run_ablation
from src.evaluation.champion import ChampionChallenger, ModelRegistry, SplitGuardError
from src.evaluation.metrics import compute_metrics, expected_calibration_error
from src.learning.synthetic import make_synthetic_bundle


@pytest.fixture(scope="module")
def cfg():
    c = load_config()
    c["gating_train"].update({"epochs": 12, "lr": 0.01, "batch_size": 512})
    return c


def test_metrics_known_values(cfg):
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.4, 0.2, 0.6])
    r = compute_metrics(y, p, cfg, "S1", "m", flags=np.array([False, True, False, True]))
    assert r.accuracy == 0.5 and r.precision == 0.5 and r.recall == 0.5 and r.f1 == 0.5
    assert r.auroc == 0.75 and r.flagged_rate == 0.5 and r.selective_accuracy == 1.0
    assert expected_calibration_error(np.array([1, 0]), np.array([1.0, 0.0])) == 0.0
    assert np.isnan(compute_metrics(np.ones(3), np.array([.2, .6, .9]), cfg).auroc)


def test_champion_challenger_guard_promote_rollback(tmp_path, cfg):
    y = np.array([0, 0, 1, 1] * 10)
    good, mid = np.where(y == 1, 0.9, 0.1), np.where(y == 1, 0.7, 0.4)
    mid[::5] = 0.5 + 0.1 * (1 - 2 * y[::5])             # make 'mid' imperfect in ranking
    rng_bad = np.random.default_rng(0).random(len(y))
    drift = {}
    cc = ChampionChallenger(ModelRegistry(tmp_path / "reg"), cfg, lambda m, d: drift.get("p", m["p"]))
    with pytest.raises(SplitGuardError):
        cc.submit({"p": good}, None, y, "test_benchmark")
    with pytest.raises(SplitGuardError):
        cc.monitor(None, y, "val_gating_fit")
    assert cc.submit({"p": mid}, None, y, "val_gating_verify").reason == "no champion yet"
    d = cc.submit({"p": rng_bad}, None, y, "val_gating_verify")
    assert not d.promoted and cc.reg.champion_id == "v0001"      # worse challenger rejected
    d = cc.submit({"p": good}, None, y, "val_gating_verify")
    assert d.promoted and cc.reg.champion_id == "v0003"
    assert cc.monitor(None, y, "val_gating_verify").reason == "healthy"
    # the champion degrades on fresh verify data -> automatic rollback to v0001
    drift["p"] = rng_bad
    dec = cc.monitor(None, y, "val_gating_verify")
    assert dec.promoted and dec.reason.startswith("regression") and cc.reg.champion_id == "v0001"
    assert [v["status"] for v in cc.reg.history] == ["champion", "rejected", "rolled_back"]
    assert ModelRegistry(tmp_path / "reg").champion_id == "v0001"   # persisted


def test_ablation_end_to_end(tmp_path, cfg):
    fit = make_synthetic_bundle(300, seed=1)
    test = make_synthetic_bundle(200, seed=2, scenarios=True)
    sup = make_synthetic_bundle(150, seed=3, unseen=True, clean_only=True, tag="sup")
    ev = make_synthetic_bundle(120, seed=4, unseen=True, clean_only=True, tag="ev")
    rows = run_ablation(cfg, fit, test, sup, ev, out_dir=tmp_path, n_values=[5, 20, 100])
    assert {r["scenario"] for r in rows} == {"S1", "S2", "S4"}
    full = [r for r in rows if r["variant"] == "full" and r["scenario"] == "S4" and r["method"] == "aegis"]
    assert sorted(r["N"] for r in full) == [5, 20, 100]
    assert {r["regime"] for r in full} == {"conservative", "anchor", "refit"}
    assert {"nocal", "noood", "nocal_noood"} <= {r["variant"] for r in rows}
    with open(tmp_path / "ablation.csv", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == len(rows)
    assert "AUROC" in (tmp_path / "ablation.md").read_text(encoding="utf-8")
