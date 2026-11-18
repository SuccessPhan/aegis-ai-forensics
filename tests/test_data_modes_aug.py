"""Normalisation modes, per-generator shortcut audit and train_aug variants (work package #8)."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from src.core.config import deep_merge
from src.core.contracts import SCENARIO_TRAIN_AUG, SPLIT_TRAIN, SplitManifest
from src.data.degradations import sample_train_aug_spec
from src.data.ingest import IngestError, main, run_ingest
from src.data.normalize import normalize_image, resize_short_side
from src.data.shortcut_audit import audit_full, audit_per_generator
from src.data.split import check_no_leakage


def _cfg_mode(cfg, mode):
    c = copy.deepcopy(cfg)
    c["data"]["normalize"]["mode"] = mode
    return c


@pytest.mark.parametrize("size", [(300, 500), (700, 260), (256, 256), (240, 400)])
def test_resize_crop_short_side_and_output(tmp_path, cfg, size):
    w, h = size
    rng = np.random.default_rng(0)
    src = tmp_path / "a.png"
    Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)).save(src)
    with Image.open(src) as im:
        r = resize_short_side(im.convert("RGB"), 256)
    assert min(r.size) == 256
    assert max(r.size) == round(max(w, h) * 256 / min(w, h))
    c = _cfg_mode(cfg, "resize_crop")
    c["data"]["normalize"]["jpeg_harmonize"] = False
    prov = normalize_image(src, tmp_path / "o.png", is_real=False, cfg=c)
    with Image.open(tmp_path / "o.png") as o:
        assert o.size == (224, 224)
    assert prov["normalize_mode"] == "resize_crop"
    assert min(prov["pre_crop_width"], prov["pre_crop_height"]) == 256
    # crop mode: no resize, pre-crop size == original
    prov2 = normalize_image(src, tmp_path / "o2.png", is_real=False, cfg=_cfg_mode(c, "crop"))
    assert (prov2["pre_crop_width"], prov2["pre_crop_height"]) == (w, h)


def test_unknown_mode_rejected(tmp_path, cfg):
    src = tmp_path / "a.png"
    Image.fromarray(np.zeros((300, 300, 3), np.uint8)).save(src)
    with pytest.raises(ValueError):
        normalize_image(src, tmp_path / "o.png", is_real=True, cfg=_cfg_mode(cfg, "bogus"))


def test_toy_resize_crop_pipeline_and_audit_per_generator(toy_manifest_resize_crop, cfg):
    man = toy_manifest_resize_crop
    sizes = {Image.open(man.abs_path(r)).size for r in man.records if r.is_clean}
    assert sizes == {(224, 224)}
    assert all(r.provenance["normalize_mode"] == "resize_crop" for r in man.records if r.is_clean)
    per = audit_per_generator(man, cfg)
    assert {"adm", "sd_v1_4"} <= set(per) <= {"adm", "sd_v1_4", "midjourney", "glide"}
    assert all(per[g].seen is (g in ("adm", "sd_v1_4")) for g in per)
    full = audit_full(man, cfg)
    assert set(full["per_generator"]) == set(per)
    assert full["passed"] == (full["pooled_passed"] and not full["failed_generators"])
    assert full["status"] == "reported"           # shortcut_audit.enforce is false by default
    assert full["passed"] == (full["pooled_passed"] and not full["failed_generators"])


def test_audit_full_flags_generator_shortcut(toy_manifest, cfg, monkeypatch):
    """Overall pass requires every per-generator audit to pass, even if the pooled one does."""
    import src.data.shortcut_audit as sa
    real_audit = sa._audit_records

    def fake(man, recs, c):
        res = real_audit(man, recs, c)
        gens = {r.generator for r in recs if r.label == 1}
        res.passed = gens != {"adm"}
        return res

    monkeypatch.setattr(sa, "_audit_records", fake)
    cfg = copy.deepcopy(cfg)
    cfg["shortcut_audit"]["enforce"] = True
    full = sa.audit_full(toy_manifest, cfg)
    assert full["pooled_passed"] is True and full["failed_generators"] == ["adm"]
    assert full["passed"] is False and full["status"] == "fail"


def _raw_tree(root: Path, n=3):
    rng = np.random.default_rng(1)
    for gen in ("adm", "sd_v1_4", "midjourney", "glide"):
        for kind in ("ai", "nature"):
            d = root / gen / kind
            d.mkdir(parents=True)
            for i in range(n):
                arr = rng.integers(0, 255, (260 + 10 * i, 300, 3), dtype=np.uint8)
                Image.fromarray(arr).save(d / f"{gen}_{kind}_{i}.png")


def test_cli_modes_separate_dirs_and_recorded(tmp_path, cfg):
    tree = tmp_path / "raw"
    _raw_tree(tree)
    from tests.conftest import no_variants
    split_cfg = no_variants(deep_merge(cfg, {"split": cfg["toy"]["split_overrides"]}))
    cp = tmp_path / "c.yaml"
    cp.write_text(yaml.safe_dump(split_cfg))
    outs = {}
    for mode in ("crop", "resize_crop"):
        out = tmp_path / f"n_{mode}"
        assert main(["--root", str(tree), "--out", str(out), "--mode", mode, "--config", str(cp)]) == 0
        outs[mode] = out
        man = SplitManifest.load(out / "manifest.json")
        rep = json.loads((out / "shortcut_audit.json").read_text())
        assert man.meta["normalize_mode"] == mode and rep["mode"] == mode
        assert rep["shortcut_audit"]["mode"] == mode
        if rep["shortcut_audit"]["status"] != "skipped":
            assert "per_generator" in rep["shortcut_audit"]
    assert (outs["crop"] / "manifest.json").exists() and (outs["resize_crop"] / "manifest.json").exists()
    assert not check_no_leakage(SplitManifest.load(outs["crop"] / "manifest.json"), cfg)
    # re-running a different mode into an existing dir is refused instead of overwriting
    with pytest.raises(IngestError):
        run_ingest(tree, outs["crop"], _cfg_mode(split_cfg, "resize_crop"))


def test_train_aug_only_in_train_and_rate(toy_manifest, cfg):
    aug = [r for r in toy_manifest.records if r.degradation and r.degradation.scenario == SCENARIO_TRAIN_AUG]
    assert aug and all(r.split == SPLIT_TRAIN and not r.is_clean for r in aug)
    by_sha = {r.sha256: r for r in toy_manifest.records}
    for v in aug:
        p = by_sha[v.parent_sha256]
        assert p.split == SPLIT_TRAIN and p.source_group_id == v.source_group_id and p.is_clean
    assert not check_no_leakage(toy_manifest, cfg)
    # clean_only drops them; include_scenarios brings them back
    clean = toy_manifest.by_split(SPLIT_TRAIN, clean_only=True)
    assert clean and all(r.is_clean for r in clean)
    both = toy_manifest.by_split(SPLIT_TRAIN, clean_only=True, include_scenarios=(SCENARIO_TRAIN_AUG,))
    assert len(both) == len(clean) + len(aug)
    # every train source got a draw recorded; variants == sources whose draw was not "none"
    assert all(len(r.provenance["train_aug_draws"]) == 1 for r in clean)
    assert len(aug) == sum(r.provenance["train_aug_draws"][0] != "none"
                           and not r.provenance["train_aug_draws"][0].endswith("(no-op)") for r in clean)
    assert len({r.sha256 for r in toy_manifest.records}) == len(toy_manifest.records)
    # no other split ever holds train_aug
    for s in ("val_gating_fit", "val_gating_verify", "test_benchmark", "s4_support", "s4_eval"):
        assert not [r for r in toy_manifest.by_split(s)
                    if r.degradation and r.degradation.scenario == SCENARIO_TRAIN_AUG]


def test_train_aug_leak_detected(toy_manifest, cfg):
    bad = SplitManifest(copy.deepcopy(toy_manifest.records), toy_manifest.seed, toy_manifest.root)
    v = next(r for r in bad.records if r.degradation and r.degradation.scenario == SCENARIO_TRAIN_AUG)
    v.split = "test_benchmark"
    assert any("train_aug" in p for p in check_no_leakage(bad, cfg))


def test_train_aug_draw_statistics_and_determinism(cfg):
    rng = np.random.default_rng(0)
    specs = [sample_train_aug_spec(rng, cfg) for _ in range(4000)]
    blur = np.mean([s is not None and s.blur_sigma is not None for s in specs])
    jpg = np.mean([s is not None and s.jpeg_quality is not None for s in specs])
    none = np.mean([s is None for s in specs])
    assert abs(blur - 0.5) < 0.05 and abs(jpg - 0.5) < 0.05 and abs(none - 0.25) < 0.05
    qs = [s.jpeg_quality for s in specs if s and s.jpeg_quality]
    sg = [s.blur_sigma for s in specs if s and s.blur_sigma]
    assert 30 <= min(qs) and max(qs) <= 100 and 0 < min(sg) and max(sg) <= 3.0
    assert all(not s.is_clean() for s in specs if s)
    a = [sample_train_aug_spec(np.random.default_rng(5), cfg) for _ in range(3)]
    b = [sample_train_aug_spec(np.random.default_rng(5), cfg) for _ in range(3)]
    assert a == b
