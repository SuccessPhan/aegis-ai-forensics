"""Three image-quality features, numpy-only, computed from PIXELS ONLY (never file headers).

Input: uint8 array (H, W, 3) or (H, W). All features are deterministic.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image

from src.core.contracts import QualityFeatures


def to_gray(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 3:
        a = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    return a


def blockiness(gray: np.ndarray, block: int = 8) -> float:
    """Blind blockiness: relative excess of mean |neighbour diff| across 8x8 block boundaries versus inside
    blocks, averaged over both axes, floored at 0. ~0 for no JPEG grid, grows with stronger compression."""
    vals = []
    for g in (gray, gray.T):
        d = np.abs(np.diff(g, axis=1)).mean(axis=0)          # diff between column j and j+1
        idx = np.arange(d.size)
        on = d[idx % block == block - 1]
        off = d[idx % block != block - 1]
        if on.size == 0 or off.size == 0 or off.mean() < 1e-9:
            vals.append(0.0)
        else:
            vals.append(on.mean() / off.mean() - 1.0)
    return float(max(0.0, np.mean(vals)))


def jpeg_quality_est(gray: np.ndarray, block: int = 8, scale: float = 1.0) -> float:
    """Heuristic quality in [0, 100]: 100 * (1 - clip(blockiness/scale, 0, 1)). Monotone in blockiness;
    a proxy for compression strength, NOT a calibrated JPEG quality factor."""
    return float(100.0 * (1.0 - min(1.0, blockiness(gray, block) / scale)))


def log_laplacian_var(gray: np.ndarray, eps: float = 1e-8) -> float:
    g = gray
    lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(np.log(lap.var() + eps))


def high_freq_energy_ratio(gray: np.ndarray, cutoff: float = 0.25) -> float:
    """Fraction of (non-DC) spectral energy at radial frequency >= cutoff cycles/pixel (Nyquist 0.5)."""
    h, w = gray.shape
    g = gray - gray.mean()
    win = np.outer(np.hanning(h), np.hanning(w))
    p = np.abs(np.fft.fft2(g * win)) ** 2
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    r = np.sqrt(fx ** 2 + fy ** 2)
    total = p.sum()
    if total <= 0:
        return 0.0
    return float(p[r >= cutoff].sum() / total)


def extract_quality_features(arr: np.ndarray, cfg: Optional[dict] = None) -> QualityFeatures:
    q = (cfg or {}).get("quality_features", {})
    gray = to_gray(arr)
    return QualityFeatures(
        jpeg_quality_est=jpeg_quality_est(gray, q.get("blockiness_block", 8), q.get("blockiness_scale", 1.0)),
        log_laplacian_var=log_laplacian_var(gray, q.get("laplacian_eps", 1e-8)),
        high_freq_energy_ratio=high_freq_energy_ratio(gray, q.get("hf_cutoff", 0.25)),
    )


def extract_from_path(path: str, cfg: Optional[dict] = None) -> QualityFeatures:
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"))
    return extract_quality_features(arr, cfg)
