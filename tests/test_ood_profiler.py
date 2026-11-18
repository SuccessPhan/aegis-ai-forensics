import numpy as np
import pytest
import torch

from src.core.config import deep_merge
from src.core.contracts import OODProfile
from src.ood.profiler import (build_ood_profile, compute_centroid, cosine_distance, load_centroids,
                              mean_ood_score, predictive_entropy, update_centroid_file)


def test_entropy_values_and_types():
    e = predictive_entropy(np.array([0.5, 0.0, 1.0, 0.9]))
    assert isinstance(e, np.ndarray)
    assert e[0] == pytest.approx(np.log(2)) and e[1] == pytest.approx(0, abs=1e-9) and e[2] == pytest.approx(0, abs=1e-9)
    assert e[3] < e[0]
    t = predictive_entropy(torch.tensor([0.5, 0.1]))
    assert isinstance(t, torch.Tensor) and t[0].item() == pytest.approx(np.log(2), rel=1e-5)


def test_cosine_distance_and_mean_score():
    c = torch.tensor([1.0, 0.0])
    d = cosine_distance(torch.tensor([[2.0, 0.0], [0.0, 3.0], [-1.0, 0.0]]), c)
    assert d.tolist() == pytest.approx([0.0, 1.0, 2.0], abs=1e-6)
    assert float(cosine_distance(np.array([1.0, 0.0], np.float32), np.array([1.0, 0.0]))) == pytest.approx(0, abs=1e-6)
    assert mean_ood_score(np.array([[0.1, 0.2, 0.3]]))[0] == pytest.approx(0.2)


def test_centroid_roundtrip(cfg, tmp_path):
    c = deep_merge(cfg, {"paths": {"cache_dir": str(tmp_path)}})
    f = torch.randn(10, 4)
    update_centroid_file(c, "clip", compute_centroid(f))
    update_centroid_file(c, "freq", compute_centroid(f * 2))
    got = load_centroids(c)
    assert set(got) == {"clip", "freq"} and torch.allclose(got["clip"], f.mean(0))


def test_build_ood_profile_uses_quality_module(cfg):
    img = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    p = build_ood_profile(img, (0.1, 0.2, 0.3), (0.3, 0.2, 0.1), cfg)
    assert isinstance(p, OODProfile) and p.mean_ood_score == pytest.approx(0.2)
    assert len(p.quality.to_array()) == 3
    with pytest.raises(ValueError):
        build_ood_profile(img, (0.1, 0.2), (0.3, 0.2, 0.1), cfg)
