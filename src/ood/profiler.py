"""Data & OOD profiler: predictive entropy, cosine distance to train_experts centroids, OODProfile builder.

Image-degradation features are NOT re-implemented here; they come from src/data/quality.py.
Centroids (clean train_experts, one per expert) are stored in cfg.paths.cache_dir/centroids_<version>.pt.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch

from src.core.config import resolve_path
from src.core.contracts import EXPERT_NAMES, OODProfile, QualityFeatures
from src.data.quality import extract_quality_features

ArrayLike = Union[np.ndarray, torch.Tensor]
_EPS = 1e-12


def predictive_entropy(p: ArrayLike):
    """Binary entropy in nats of P(fake). Same type as input (tensor -> tensor, else ndarray); elementwise."""
    if isinstance(p, torch.Tensor):
        q = p.float().clamp(_EPS, 1 - _EPS)
        return -(q * q.log() + (1 - q) * (1 - q).log())
    q = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1 - _EPS)
    return -(q * np.log(q) + (1 - q) * np.log(1 - q))


def cosine_distance(features: ArrayLike, centroid: ArrayLike):
    """1 - cos(feature, centroid). features (n,d) or (d,), centroid (d,). Returns same kind as `features`."""
    f = torch.as_tensor(features, dtype=torch.float32)
    c = torch.as_tensor(centroid, dtype=torch.float32)
    out = 1.0 - torch.nn.functional.cosine_similarity(f, c.expand_as(f), dim=-1, eps=1e-8)
    return out.numpy() if isinstance(features, np.ndarray) else out


def mean_ood_score(cosdist: ArrayLike):
    """Mean cosine distance across the 3 experts (last axis)."""
    if isinstance(cosdist, torch.Tensor):
        return cosdist.float().mean(dim=-1)
    return np.asarray(cosdist, dtype=np.float64).mean(axis=-1)


def compute_centroid(features: torch.Tensor) -> torch.Tensor:
    """Mean feature vector (float32) of clean train_experts features."""
    return features.float().mean(dim=0)


def centroid_path(cfg: dict) -> Path:
    name = (cfg.get("ood") or {}).get("centroid_file")
    if not name:
        name = f"centroids_{cfg['experts']['extract']['cache_version']}.pt"
    return resolve_path(cfg, "cache_dir") / name


def save_centroids(cfg: dict, centroids: Dict[str, torch.Tensor]) -> Path:
    p = centroid_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.float().cpu() for k, v in centroids.items()}, p)
    return p


def load_centroids(cfg: dict) -> Dict[str, torch.Tensor]:
    return torch.load(centroid_path(cfg), map_location="cpu", weights_only=True)


def update_centroid_file(cfg: dict, expert: str, centroid: torch.Tensor) -> Path:
    cur = load_centroids(cfg) if centroid_path(cfg).exists() else {}
    cur[expert] = centroid
    return save_centroids(cfg, cur)


def build_ood_profile(image: Optional[np.ndarray], entropy: Sequence[float], cosdist: Sequence[float],
                      cfg: Optional[dict] = None, quality: Optional[QualityFeatures] = None) -> OODProfile:
    """Assemble an OODProfile. Quality features are computed by src.data.quality from `image` unless given.
    `entropy` should be computed on calibrated probabilities, in EXPERT_NAMES order."""
    if len(entropy) != len(EXPERT_NAMES) or len(cosdist) != len(EXPERT_NAMES):
        raise ValueError("entropy and cosdist need one value per expert (clip, dinov2, freq)")
    if quality is None:
        if image is None:
            raise ValueError("need image or quality")
        quality = extract_quality_features(image, cfg)
    cd = tuple(float(x) for x in cosdist)
    return OODProfile(quality=quality, entropy=tuple(float(x) for x in entropy), cosine_distance=cd,
                      mean_ood_score=float(np.mean(cd)))
