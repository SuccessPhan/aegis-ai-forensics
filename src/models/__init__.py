"""Heterogeneous expert pool (clip, dinov2, freq)."""
from __future__ import annotations


def build_expert(name: str, cfg: dict, backbone=None, feature_dim=None):
    """Factory. `backbone` (optional) injects a mock/pre-built backbone for clip/dinov2."""
    if name == "clip":
        from src.models.clip_expert import ClipExpert
        return ClipExpert(cfg, backbone, feature_dim)
    if name == "dinov2":
        from src.models.dinov2_expert import Dinov2Expert
        return Dinov2Expert(cfg, backbone, feature_dim)
    if name == "freq":
        from src.models.freq_expert import FreqExpert
        return FreqExpert(cfg)
    raise ValueError(f"unknown expert '{name}'")
