"""Head clean/aug variants, freq input variants + selection, bundle head_variant output."""
import torch

from src.core.contracts import SCENARIO_TRAIN_AUG, SPLIT_TRAIN, SPLIT_VAL_FIT
from src.models import build_expert
from src.models.bundle import main as bundle_main
from src.models.cache import FeatureCache
from src.models.freq_expert import FreqExpert
from src.models.heads import (head_path, load_head_into, select_freq_input, train_expert_from_manifest,
                              train_records)
from src.ood.profiler import compute_centroid, load_centroids
from tests.test_models_experts import MockBackbone, _small_cfg


def _fill_cache(c, man, dim=8, names=("clip", "dinov2")):
    g = torch.Generator().manual_seed(0)
    for n in names:
        cache = FeatureCache.open(c, n)
        for r in man.records:
            cache.put(r.sha256, torch.randn(dim, generator=g))
        cache.save()


def _patch_build(monkeypatch):
    import src.models as M
    orig = M.build_expert
    monkeypatch.setattr(M, "build_expert", lambda name, cfg, backbone=None, feature_dim=None: orig(
        name, cfg, backbone=MockBackbone(8) if name != "freq" else None, feature_dim=8 if name != "freq" else None))


def test_models_train_records_sets(toy_manifest):
    clean = train_records(toy_manifest, "clean")
    aug = train_records(toy_manifest, "aug")
    assert clean == toy_manifest.by_split(SPLIT_TRAIN, clean_only=True)
    assert len(aug) > len(clean)
    extra = [r for r in aug if not r.is_clean]
    assert extra and all(r.split == SPLIT_TRAIN and r.degradation.scenario == SCENARIO_TRAIN_AUG for r in extra)
    assert all(r.split == SPLIT_TRAIN for r in aug)  # no val/test variants
    assert {r.sha256 for r in clean} <= {r.sha256 for r in aug}


def test_models_variant_file_names_and_centroid_clean_only(cfg, toy_manifest, tmp_path, monkeypatch):
    c = _small_cfg(cfg, tmp_path)
    _fill_cache(c, toy_manifest)
    _patch_build(monkeypatch)
    pc = train_expert_from_manifest(toy_manifest, "clip", c, variant="clean")
    cent_clean = load_centroids(c)["clip"].clone()
    pa = train_expert_from_manifest(toy_manifest, "clip", c, variant="aug")
    assert pc != pa and pc.exists() and pa.exists()
    assert pc.name == "clip_clean_v1.pt" and pa.name == "clip_aug_v1.pt"
    # centroid = mean of CLEAN train_experts features, identical for both variants
    cache = FeatureCache.open(c, "clip")
    clean = train_records(toy_manifest, "clean")
    want = compute_centroid(cache.features_for([r.sha256 for r in clean]))
    assert torch.allclose(load_centroids(c)["clip"], want, atol=1e-5)
    assert torch.allclose(cent_clean, want, atol=1e-5)
    # the aug head really saw other data
    a = torch.load(pa, weights_only=True)["state"]["mean"]
    b = torch.load(pc, weights_only=True)["state"]["mean"]
    assert not torch.allclose(a, b)
    assert head_path(c, "freq", "aug", "srm").name == "freq_srm_aug_v1.pt"
    assert head_path(c, "freq", "aug", "log_magnitude_fft") != head_path(c, "freq", "aug", "srm")


def test_models_select_freq_input_picks_better(cfg, toy_manifest, tmp_path):
    c = _small_cfg(cfg, tmp_path)
    recs = [r for r in toy_manifest.by_split(SPLIT_VAL_FIT) if r.is_clean]

    class Stub:
        def __init__(self, scores):
            self.it = iter(scores)

        def preprocess(self, imgs):
            return imgs

        def features_and_logits(self, x):
            return None, torch.tensor([next(self.it)])

    good = [float(r.label) + 0.01 * i for i, r in enumerate(recs)]
    bad = [float(i % 2) - float(r.label) * 0.5 for i, r in enumerate(recs)]
    out = select_freq_input(toy_manifest, c, "aug", experts={"log_magnitude_fft": Stub(bad), "srm": Stub(good)})
    assert out["selected"] == "srm" and out["auroc"]["srm"] > out["auroc"]["log_magnitude_fft"]
    import json
    blob = json.loads((tmp_path / "cache" / "freq_selection.json").read_text())
    assert blob["selected"]["aug"] == "srm" and "srm" in blob["auroc"]["aug"]


def test_models_freq_variants_and_bundle_subdir(cfg, toy_manifest, tmp_path, monkeypatch):
    c = _small_cfg(cfg, tmp_path)
    c["experts"]["freq"]["train"]["epochs"] = 1
    _fill_cache(c, toy_manifest)
    _patch_build(monkeypatch)
    # restrict train to a few records to stay fast
    import src.models.heads as H
    full = H.train_records
    monkeypatch.setattr(H, "train_records", lambda m, v="clean": (lambda rs: [
        r for r in rs if r.is_clean][:4] + [r for r in rs if not r.is_clean][:2] if v == "aug"
        else [r for r in rs if r.is_clean][:4])(full(m, v)))
    for name in ("clip", "dinov2"):
        train_expert_from_manifest(toy_manifest, name, c, variant="aug")
    p_srm = train_expert_from_manifest(toy_manifest, "freq", c, variant="aug", freq_input="srm")
    p_fft = train_expert_from_manifest(toy_manifest, "freq", c, variant="aug", freq_input="log_magnitude_fft")
    assert p_srm.name == "freq_srm_aug_v1.pt" and p_fft.name == "freq_log_magnitude_fft_aug_v1.pt"
    e = FreqExpert(c, "srm")
    load_head_into(c, e, "aug")
    assert e.input_mode == "srm"

    # bundle CLI -> bundles/aug/<split>.pt with meta
    man_path = tmp_path / "m.json"
    toy_manifest.save(man_path)
    import src.models.bundle as B
    monkeypatch.setattr(B, "load_config", lambda: c)
    small = [r for r in toy_manifest.records if r.split == SPLIT_VAL_FIT][:4]
    monkeypatch.setattr(B.SplitManifest, "load", staticmethod(
        lambda p: type(toy_manifest)(records=small, seed=0, root=toy_manifest.root, meta={})))
    assert bundle_main(["--manifest", str(man_path), "--split", SPLIT_VAL_FIT, "--head-variant", "aug",
                        "--freq-input", "srm", "--device", "cpu"]) == 0
    out = tmp_path / "cache" / "bundles" / "aug" / f"{SPLIT_VAL_FIT}.pt"
    b = torch.load(out, weights_only=False)
    assert b["meta"]["head_variant"] == "aug" and b["meta"]["freq_input"] == "srm"
    assert b["logits"].shape == (4, 3) and set(b) >= {"sha256", "label", "cosdist", "quality", "meta"}
