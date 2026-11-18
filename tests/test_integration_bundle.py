"""Cross-owner contract test: a bundle written by src.models.bundle must load in src.learning.bundle_io."""
import torch

from src.core.contracts import SPLIT_TRAIN, SPLIT_VAL_FIT
from src.learning.bundle_io import bundle_from_dict
from src.learning.features import predictive_entropy as learning_entropy
from src.models import build_expert
from src.models.base import load_image_array
from src.models.bundle import build_bundle
from src.models.cache import FeatureCache
from src.models.extract import extract_records
from src.models.freq_expert import FreqExpert
from src.models.heads import train_linear_head
from src.ood.profiler import compute_centroid, predictive_entropy as ood_entropy
from tests.test_models_experts import MockBackbone, _small_cfg


def test_models_bundle_loads_in_learning(cfg, toy_manifest, tmp_path):
    c = _small_cfg(cfg, tmp_path)
    recs = toy_manifest.by_split(SPLIT_VAL_FIT)[:6]
    train_recs = toy_manifest.by_split(SPLIT_TRAIN, clean_only=True)[:8]

    experts, caches, cents = {}, {}, {}
    for name in ("clip", "dinov2"):
        e = build_expert(name, c, backbone=MockBackbone(8), feature_dim=8)
        cache = FeatureCache.open(c, name)
        extract_records(e, toy_manifest, recs + train_recs, cache, batch_size=8)
        tf = cache.features_for([r.sha256 for r in train_recs])
        e.head = train_linear_head(tf, torch.tensor([r.label for r in train_recs]), c)
        experts[name], caches[name], cents[name] = e, cache, compute_centroid(tf)
    freq = FreqExpert(c)
    x = freq.preprocess([load_image_array(str(toy_manifest.abs_path(train_recs[0])))])
    experts["freq"], cents["freq"] = freq, compute_centroid(freq.extract_features(x))

    raw = build_bundle(toy_manifest, recs, SPLIT_VAL_FIT, c, experts, caches, cents, batch_size=4)
    b = bundle_from_dict(raw)  # raises BundleError on any format mismatch
    assert len(b.sha256) == len(recs)

    # learning must use the same entropy as src.ood (its local fallback must agree)
    p = torch.tensor([0.1, 0.5, 0.9])
    assert torch.allclose(torch.as_tensor(learning_entropy(p)).float(),
                          torch.as_tensor(ood_entropy(p)).float(), atol=1e-6)
