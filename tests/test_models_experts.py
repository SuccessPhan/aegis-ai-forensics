"""Expert pool tests: mock backbones only (no network, no weights)."""
import numpy as np
import pytest
import torch
from torch import nn

from src.core.config import deep_merge
from src.core.contracts import EXPERT_NAMES, QUALITY_FEATURE_NAMES, SPLIT_TRAIN, SPLIT_VAL_FIT
from src.models import build_expert
from src.models.bundle import build_bundle
from src.models.cache import FeatureCache
from src.models.extract import extract_records
from src.models.freq_expert import FreqExpert, ImagePathDataset, srm_residual, train_freq
from src.models.heads import LinearHead, load_head_into, save_head, train_linear_head
from src.ood.profiler import compute_centroid, cosine_distance


class MockBackbone(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(48, dim))

    def forward(self, pixel_values):
        return self.net(pixel_values)


def _small_cfg(cfg, tmp_path):
    return deep_merge(cfg, {"paths": {"cache_dir": str(tmp_path / "cache")},
                            "experts": {"freq": {"channels": [4, 8, 8, 8], "train": {"epochs": 2, "batch_size": 8}},
                                        "head": {"epochs": 3}}})


def test_freq_param_count_in_range(cfg):
    n = FreqExpert(cfg).n_params()
    assert 1.0e6 <= n <= 1.5e6, n


@pytest.mark.parametrize("name", ["clip", "dinov2"])
def test_frozen_backbone_expert(cfg, toy_manifest, name):
    ex = build_expert(name, cfg, backbone=MockBackbone(16), feature_dim=16)
    assert not ex.backbone.training and all(not p.requires_grad for p in ex.backbone.parameters())
    ex.train()
    assert not ex.backbone.training  # backbone stays in eval
    recs = toy_manifest.by_split(SPLIT_TRAIN, clean_only=True)[:4]
    from src.models.base import load_image_array
    x = ex.preprocess([load_image_array(str(toy_manifest.abs_path(r))) for r in recs])
    assert x.shape == (4, 3, 224, 224)
    f = ex.extract_features(x)
    assert f.shape == (4, 16) and not f.requires_grad
    with pytest.raises(RuntimeError):
        ex.predict_logits(f)  # no head yet
    ex.head = LinearHead(16)
    assert ex.predict_logits(f).shape == (4,)
    assert ((ex.predict_proba(f) >= 0) & (ex.predict_proba(f) <= 1)).all()


def test_clip_rejects_non_224(cfg):
    ex = build_expert("clip", cfg, backbone=MockBackbone(8), feature_dim=8)
    with pytest.raises(ValueError):
        ex.preprocess([np.zeros((64, 64, 3), np.uint8)])


def test_freq_preprocess_keeps_size_and_srm(cfg):
    ex = FreqExpert(deep_merge(cfg, {"experts": {"freq": {"input": "srm", "channels": [4, 4, 4, 4]}}}))
    img = np.random.default_rng(0).integers(0, 255, (40, 56, 3), dtype=np.uint8)
    x = ex.preprocess([img])
    assert x.shape == (1, 3, 40, 56)
    assert srm_residual(x).shape == (1, 3, 40, 56)
    assert ex.extract_features(x).shape == (1, 4)


def test_extract_resume_and_head_training(cfg, toy_manifest, tmp_path):
    c = _small_cfg(cfg, tmp_path)
    ex = build_expert("dinov2", c, backbone=MockBackbone(16), feature_dim=16)
    cache = FeatureCache.open(c, "dinov2")
    recs = toy_manifest.by_split(SPLIT_TRAIN, clean_only=True)[:12]
    assert extract_records(ex, toy_manifest, recs, cache, batch_size=5) == 12
    cache2 = FeatureCache.open(c, "dinov2")  # reload from disk
    assert len(cache2) == 12 and cache2.data[recs[0].sha256].dtype == torch.float16
    assert extract_records(ex, toy_manifest, recs, cache2, batch_size=5) == 0  # resume: nothing to do
    assert "v1" in str(cache.path)

    feats = cache2.features_for([r.sha256 for r in recs])
    y = torch.tensor([r.label for r in recs])
    head = train_linear_head(feats, y, c)
    assert head(feats).shape == (12,)
    p = save_head(c, "dinov2", head.state_dict(), 16)
    ex2 = build_expert("dinov2", c, backbone=MockBackbone(16), feature_dim=16)
    load_head_into(c, ex2)
    assert p.exists() and torch.allclose(ex2.predict_logits(feats), head(feats), atol=1e-5)


def test_freq_train_and_bundle(cfg, toy_manifest, tmp_path):
    c = _small_cfg(cfg, tmp_path)
    recs = toy_manifest.by_split(SPLIT_VAL_FIT)[:10]
    train_recs = toy_manifest.by_split(SPLIT_TRAIN, clean_only=True)[:16]
    freq = FreqExpert(c)
    ds = ImagePathDataset([str(toy_manifest.abs_path(r)) for r in train_recs], [r.label for r in train_recs])
    hist = train_freq(freq, ds, c)
    assert len(hist) == 2 and all(np.isfinite(hist))

    experts, caches, cents = {"freq": freq}, {}, {}
    for name in ("clip", "dinov2"):
        e = build_expert(name, c, backbone=MockBackbone(16), feature_dim=16)
        cache = FeatureCache.open(c, name)
        extract_records(e, toy_manifest, recs + train_recs, cache, batch_size=8)
        tf = cache.features_for([r.sha256 for r in train_recs])
        e.head = train_linear_head(tf, torch.tensor([r.label for r in train_recs]), c)
        experts[name], caches[name], cents[name] = e, cache, compute_centroid(tf)
    cents["freq"] = compute_centroid(freq.extract_features(ds[0][0][None]))
    b = build_bundle(toy_manifest, recs, SPLIT_VAL_FIT, c, experts, caches, cents, batch_size=4)

    n = len(recs)
    assert b["logits"].shape == (n, 3) and b["cosdist"].shape == (n, 3) and b["quality"].shape == (n, 3)
    assert b["label"].dtype == torch.long and len(b["sha256"]) == n
    assert len(b["degradation"]) == len(b["scenario"]) == len(b["generator"]) == n
    assert torch.isfinite(b["logits"]).all() and (b["cosdist"] >= -1e-5).all()
    assert set(b["meta"]) >= {"cache_version", "model_ids", "split", "created"}
    assert len(QUALITY_FEATURE_NAMES) == 3 and len(EXPERT_NAMES) == 3
    for r, d in zip(recs, b["degradation"]):
        assert d == (None if r.is_clean else r.degradation.name)
