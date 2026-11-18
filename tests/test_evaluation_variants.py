import csv
import warnings

import numpy as np
import pytest

from src.core.config import load_config
from src.evaluation import ablation as ab
from src.learning.bundle_io import save_bundle
from src.learning.fewshot import select_support
from src.learning.synthetic import make_synthetic_bundle


@pytest.fixture(scope="module")
def cfg():
    c = load_config()
    c["gating_train"].update({"epochs": 8, "lr": 0.01, "batch_size": 512})
    c["fewshot"]["n_values"] = [5, 20]
    return c


def _two_gen(b):
    """Relabel every second AI sample as 'glide' (the synthetic unseen generator is 'midjourney')."""
    k = 0
    for i, lb in enumerate(b.label):
        if lb == 1:
            k += 1
            if k % 2 == 0:
                b.generator[i] = "glide"
    return b


def _bundles(seed):
    return (make_synthetic_bundle(200, seed=seed), make_synthetic_bundle(150, seed=seed + 1, scenarios=True),
            _two_gen(make_synthetic_bundle(260, seed=seed + 2, unseen=True, clean_only=True, tag="sup")),
            _two_gen(make_synthetic_bundle(100, seed=seed + 3, unseen=True, clean_only=True, tag="ev")))


@pytest.fixture(scope="module")
def two_variant_rows(cfg):
    rows = []
    for hv, seed in (("clean", 1), ("aug", 11)):
        fit, test, sup, ev = _bundles(seed)
        rows += ab.run_ablation(cfg, fit, test, sup, ev, head_variant=hv)
    return rows


def test_both_head_variants_in_tables(tmp_path, two_variant_rows):
    ab.write_tables(two_variant_rows, tmp_path)
    assert {r["head_variant"] for r in two_variant_rows} == {"clean", "aug"}
    with open(tmp_path / "ablation.csv", encoding="utf-8") as f:
        assert {r["head_variant"] for r in csv.DictReader(f)} == {"clean", "aug"}
    for hv in ("clean", "aug"):
        assert f"head variant = {hv}" in (tmp_path / f"ablation_{hv}.md").read_text(encoding="utf-8")
    assert "best_baseline" in (tmp_path / "ablation_summary.md").read_text(encoding="utf-8")


def test_s4_split_per_generator_with_correct_support(cfg, two_variant_rows):
    s4 = [r for r in two_variant_rows if r["scenario"] == "S4" and r["method"] == "aegis" and r["variant"] == "full"
          and r["head_variant"] == "clean"]
    assert {r["generator"] for r in s4} == {"midjourney", "glide", "all"}
    _, _, sup, _ = _bundles(1)
    for g in ("midjourney", "glide"):
        for r in (x for x in s4 if x["generator"] == g):
            shas = r["support_sha256"][cfg["seed"]]
            idx = {s: i for i, s in enumerate(sup.sha256)}
            sel = [idx[s] for s in shas]
            part = ab._gen_bundle(sup, g, ab.partition_real(sup, ["glide", "midjourney"], cfg["seed"],
                                                            clean_only=True)[g])
            assert shas == [str(x) for x in select_support(part, r["N"], cfg["seed"], generator=g).sha256]
            assert sum(sup.label[i] == 1 for i in sel) == (r["N"] + 1) // 2
            assert {sup.generator[i] for i in sel if sup.label[i] == 1} == {g}
            assert sum(sup.label[i] == 0 for i in sel) == r["N"] // 2


def _row(hv, method, auroc, n=None, sc="S1", variant="full"):
    return {"head_variant": hv, "scenario": sc, "level": "clean", "generator": None, "N": n, "method": method,
            "variant": variant, "n": 10, "accuracy": auroc - 0.05, "auroc": auroc, "precision": 0, "recall": 0,
            "f1": 0, "ece": 0, "selective_accuracy": None, "flagged_rate": 0}


def test_summary_picks_strongest_baseline_across_variants():
    rows = [_row("clean", "aegis", 0.90), _row("clean", "dinov2", 0.85), _row("clean", "clip", 0.70),
            _row("aug", "aegis", 0.93), _row("aug", "dinov2", 0.95), _row("aug", "naive_avg", 0.80),
            _row("aug", "aegis", 0.99, variant="nocal")]
    summ = {r["head_variant"]: r for r in ab.summarize_best_baseline(rows)}
    assert summ["clean"]["best_baseline"] == "dinov2_aug"          # beats clean head's own best baseline
    assert summ["clean"]["delta_auroc"] == pytest.approx(-0.05)
    assert summ["aug"]["best_baseline"] == "dinov2_aug" and summ["aug"]["delta_auroc"] == pytest.approx(-0.02)


def test_cli_missing_bundles_and_success(tmp_path, cfg, monkeypatch, caplog):
    c = dict(cfg, paths=dict(cfg["paths"], cache_dir=str(tmp_path / "cache")))
    with pytest.raises(ab.MissingBundlesError) as ei:
        ab.run_all(c, ["clean", "aug"], tmp_path / "out")
    assert len(ei.value.missing) == 8 and "clean" in ei.value.missing[0] and "s4_eval.pt" in str(ei.value)
    import src.core.config as cc
    monkeypatch.setattr(cc, "load_config", lambda path=None: c)
    assert ab.main(["--head-variants", "aug", "--out", str(tmp_path / "out")]) == 2
    for hv, seed in (("clean", 1), ("aug", 11)):                   # now provide the bundles (auto-detect variants)
        fit, test, sup, ev = _bundles(seed)
        for name, b in (("val_gating_fit", fit), ("test_benchmark", test), ("s4_support", sup), ("s4_eval", ev)):
            save_bundle(b, tmp_path / "cache" / "bundles" / hv / f"{name}.pt")
    c["fewshot"]["n_values"] = [5]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)               # no 'Mean of empty slice'
        assert ab.main(["--out", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "ablation_summary.csv").exists() and (tmp_path / "out" / "ablation_aug.md").exists()


def test_mean_reports_all_nan_no_warning():
    from types import SimpleNamespace as NS
    reps = [NS(**{k: float("nan") for k in ab._NUM}, extras={"support_sha256": {i: []}}) for i in range(2)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = ab._mean_reports(reps)
    assert r.auroc != r.auroc and r.extras["n_seeds"] == 2


def test_quality_only_floor_everywhere_and_not_best_baseline(two_variant_rows):
    rows = two_variant_rows
    for hv in ("clean", "aug"):
        for key in {(r["scenario"], r["level"], r["generator"]) for r in rows if r["head_variant"] == hv}:
            assert any(r["method"] == "quality_only" and r["variant"] == "full" and r["head_variant"] == hv
                       and (r["scenario"], r["level"], r["generator"]) == key for r in rows), key
    assert {r["scenario"] for r in rows if r["method"] == "quality_only"} == {"S1", "S2", "S4"}
    summ = ab.summarize_best_baseline(rows)
    assert summ and all(not r["best_baseline"].startswith("quality_only") for r in summ)
    assert all(r["floor_auroc"] is not None for r in summ)
    assert all(r["aegis_minus_floor"] == pytest.approx(r["aegis_auroc"] - r["floor_auroc"]) for r in summ)


def test_summary_excludes_floor_even_if_strongest():
    rows = [_row("clean", "aegis", 0.90), _row("clean", "dinov2", 0.85), _row("clean", "quality_only", 0.99)]
    s = ab.summarize_best_baseline(rows)[0]
    assert s["best_baseline"].startswith("dinov2") and s["floor_auroc"] == 0.99
    assert s["aegis_minus_floor"] == pytest.approx(-0.09)


def test_partition_real_disjoint_sized_deterministic():
    _, _, _, ev = _bundles(1)
    gens = ["glide", "midjourney"]
    part = ab.partition_real(ev, gens, 0)
    n_ai = {g: sum(1 for lb, x in zip(ev.label, ev.generator) if lb == 1 and x == g) for g in gens}
    n_real = int((ev.label == 0).sum())
    assert set(part["glide"]).isdisjoint(part["midjourney"])
    assert all(ev.label[i] == 0 for v in part.values() for i in v)
    for g in gens:
        assert len(part[g]) == (n_ai[g] if sum(n_ai.values()) <= n_real else n_real // 2)
    again = ab.partition_real(ev, gens, 0)
    assert all((again[g] == part[g]).all() for g in gens)
    other = ab.partition_real(ev, gens, 1)
    assert any(not np.array_equal(other[g], part[g]) for g in gens)
    small = ab.partition_real(ev.subset(np.arange(len(ev))[:30]), gens, 0)   # too few Real -> equal share
    assert set(small["glide"]).isdisjoint(small["midjourney"])
    assert len(small["glide"]) == len(small["midjourney"])


def test_s4_rows_use_disjoint_real_parts(cfg, two_variant_rows):
    _, _, _, ev = _bundles(1)
    part = ab.partition_real(ev, ["glide", "midjourney"], cfg["seed"])
    rows = {r["generator"]: r for r in two_variant_rows if r["scenario"] == "S4" and r["method"] == "quality_only"
            and r["variant"] == "full" and r["head_variant"] == "clean"}
    for g in ("glide", "midjourney"):
        n_ai = sum(1 for lb, x in zip(ev.label, ev.generator) if lb == 1 and x == g)
        assert rows[g]["n"] == n_ai + len(part[g])
    assert rows["all"]["n"] == len(ev)
