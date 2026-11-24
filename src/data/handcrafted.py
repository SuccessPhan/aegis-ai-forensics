"""Hand-crafted image features for the classic machine-learning baselines (no deep learning anywhere).

The question these features answer: can classic machine-learning models, given features computed directly from the
pixels, detect AI-generated images as well as the deep experts of AEGIS (CLIP, DINOv2, the frequency CNN)?

Every feature is a fixed formula on the normalised 224x224 image (the same input the experts see):

  colour    (18) mean / std / skewness of R, G, B, H, S, V
  spectrum  (27) log-magnitude FFT averaged in 24 radial bands, spectral slope + intercept of log-power vs
                 log-frequency, high/low frequency energy ratio (generators leave periodic up-sampling traces)
  noise     (33) residuals of the 3 SRM high-pass kernels used by the frequency expert: mean |r|, std, kurtosis and an
                 8-bin histogram of the truncated residual per kernel (camera noise vs generator noise)
  texture   (10) uniform LBP(8,1) histogram of the grey image
  gradient   (4) Laplacian variance, Sobel magnitude mean / std, edge density

Total 92 values per image, cached by sha256 in data/cache/features/handcrafted_<version>.pt (float32), with the same
FeatureCache format as the CLIP / DINOv2 embeddings, so the comparison code treats it as one more feature set.

CLI: python -m src.data.handcrafted --manifest data/normalized_v2/manifest.json [--workers 4]
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy import ndimage

from src.core.config import load_config, resolve_path
from src.core.logger import get_logger

log = get_logger(__name__)

HANDCRAFTED_VERSION = "hc1"
FEATURE_SET_NAME = "handcrafted"
N_RADIAL = 24
_SRM_T = 2.0          # residual truncation for the histogram (after SRM_SCALE gain)
_EPS = 1e-8

# The 3 SRM kernels of src.models.freq_expert (copied as numpy so this module never imports torch models).
_SRM_KERNELS = [
    np.array([[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]]) / 4.0,
    np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2],
              [-1, 2, -2, 2, -1]]) / 12.0,
    np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]) / 2.0,
]
_SRM_SCALE = 10.0


def _names() -> List[str]:
    names = [f"colour_{c}_{s}" for c in ("R", "G", "B", "H", "S", "V") for s in ("mean", "std", "skew")]
    names += [f"spec_band{i:02d}" for i in range(N_RADIAL)] + ["spec_slope", "spec_intercept", "spec_hf_lf_ratio"]
    for k in range(3):
        names += [f"srm{k}_absmean", f"srm{k}_std", f"srm{k}_kurt"] + [f"srm{k}_hist{i}" for i in range(8)]
    names += [f"lbp_{i}" for i in range(10)]
    names += ["grad_lap_var", "grad_sobel_mean", "grad_sobel_std", "grad_edge_density"]
    return names


FEATURE_NAMES: List[str] = _names()
N_FEATURES = len(FEATURE_NAMES)


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx, mn = rgb.max(-1), rgb.min(-1)
    d = mx - mn
    h = np.zeros_like(mx)
    m = d > _EPS
    rm, gm, bm = m & (mx == r), m & (mx == g) & (mx != r), m & (mx == b) & (mx != r) & (mx != g)
    h[rm] = ((g - b)[rm] / d[rm]) % 6
    h[gm] = (b - r)[gm] / d[gm] + 2
    h[bm] = (r - g)[bm] / d[bm] + 4
    s = np.where(mx > _EPS, d / np.maximum(mx, _EPS), 0.0)
    return np.stack([h / 6.0, s, mx], -1)


def _moments(x: np.ndarray) -> Tuple[float, float, float, float]:
    """mean, std, skewness, excess kurtosis (population formulas, = scipy.stats skew / kurtosis defaults)."""
    x = x.ravel()
    mu = x.mean()
    d = x - mu
    d2 = d * d
    var = d2.mean()
    if var <= 1e-12:
        return float(mu), 0.0, 0.0, 0.0
    sd = np.sqrt(var)
    return float(mu), float(sd), float((d2 * d).mean() / var ** 1.5), float((d2 * d2).mean() / var ** 2 - 3.0)


def _colour(rgb: np.ndarray) -> List[float]:
    ch = np.concatenate([rgb, _rgb_to_hsv(rgb)], -1).reshape(-1, 6)
    out = []
    for j in range(6):
        mu, sd, sk, _ = _moments(ch[:, j])
        out += [mu, sd, sk]
    return out


@lru_cache(maxsize=8)
def _radial_index(h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy - h / 2, xx - w / 2)
    idx = np.minimum((r / r.max() * N_RADIAL).astype(int), N_RADIAL - 1).ravel()
    return idx, np.maximum(np.bincount(idx, None, N_RADIAL), 1)


def _spectrum(gray: np.ndarray) -> List[float]:
    f = np.abs(np.fft.fftshift(np.fft.fft2(gray - gray.mean()))).ravel()
    idx, count = _radial_index(*gray.shape)
    power = np.bincount(idx, f * f, N_RADIAL) / count
    bands = np.log1p(np.bincount(idx, f, N_RADIAL) / count)
    freq = (np.arange(N_RADIAL) + 0.5) / N_RADIAL
    sel = slice(1, N_RADIAL)  # skip DC band
    slope, intercept = np.polyfit(np.log(freq[sel]), np.log(power[sel] + _EPS), 1)
    hf = power[N_RADIAL // 2:].sum() / (power[1:N_RADIAL // 2].sum() + _EPS)
    return list(bands) + [float(slope), float(intercept), float(np.log(hf + _EPS))]


def _noise(gray: np.ndarray) -> List[float]:
    out = []
    edges = np.linspace(-_SRM_T, _SRM_T, 9)
    for k in _SRM_KERNELS:
        res = ndimage.convolve(gray, k, mode="reflect") * _SRM_SCALE
        _, sd, _, ku = _moments(res)
        out += [float(np.abs(res).mean()), sd, ku]
        out += list(np.histogram(np.clip(res, -_SRM_T, _SRM_T), edges)[0] / res.size)
    return out


def _lbp(gray: np.ndarray) -> List[float]:
    """Uniform LBP(8, 1): 9 uniform patterns (0..8 ones) + 1 bin for non-uniform patterns."""
    c = gray[1:-1, 1:-1]
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    bits = np.stack([(gray[1 + dy: gray.shape[0] - 1 + dy, 1 + dx: gray.shape[1] - 1 + dx] >= c) for dy, dx in offs])
    ones = bits.sum(0)
    transitions = (bits != np.roll(bits, 1, axis=0)).sum(0)
    code = np.where(transitions <= 2, ones, 9)
    return list(np.bincount(code.ravel(), minlength=10)[:10] / code.size)


def _gradient(gray: np.ndarray) -> List[float]:
    lap = ndimage.laplace(gray)
    gx, gy = ndimage.sobel(gray, axis=1), ndimage.sobel(gray, axis=0)
    mag = np.hypot(gx, gy)
    return [lap.var(), mag.mean(), mag.std(), float((mag > 0.5).mean())]


def extract_handcrafted(arr: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 RGB image -> (92,) float32 feature vector."""
    rgb = np.asarray(arr, dtype=np.float64)[..., :3] / 255.0
    gray = rgb @ np.array([0.299, 0.587, 0.114])
    v = _colour(rgb) + _spectrum(gray) + _noise(gray) + _lbp(gray) + _gradient(gray)
    out = np.asarray(v, dtype=np.float32)
    assert out.shape == (N_FEATURES,), out.shape
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# ------------------------------------------------------------------------------------------------------- cache
def open_cache(cfg: dict):
    from src.models.cache import FeatureCache
    path = resolve_path(cfg, "cache_dir") / "features" / f"{FEATURE_SET_NAME}_{HANDCRAFTED_VERSION}.pt"
    cache = FeatureCache(path, FEATURE_SET_NAME, HANDCRAFTED_VERSION, dtype=torch.float32)
    cache.meta.setdefault("feature_names", FEATURE_NAMES)
    return cache


def build_cache(cfg: dict, man, records: Optional[Sequence] = None, workers: int = 4, save_every: int = 2000):
    """Compute the features of every manifest record not cached yet (resumable). Returns the cache."""
    from src.models.base import load_image_array
    cache = open_cache(cfg)
    todo = [r for r in (records if records is not None else man.records) if not cache.has(r.sha256)]
    if not todo:
        log.info("handcrafted features: all %d records cached (%s)", len(cache), cache.path)
        return cache
    log.info("handcrafted features: computing %d images (%d already cached)", len(todo), len(cache))
    t0 = time.perf_counter()

    def one(r):
        return r.sha256, extract_handcrafted(load_image_array(str(man.abs_path(r))))

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for i, (sha, f) in enumerate(pool.map(one, todo), 1):
            cache.put(sha, torch.from_numpy(f))
            if i % save_every == 0 or i == len(todo):
                cache.save()
                log.info("handcrafted features: %d/%d (%.0fs)", i, len(todo), time.perf_counter() - t0)
    return cache


def main(argv: Optional[Sequence[str]] = None) -> int:
    from src.core.contracts import SplitManifest
    ap = argparse.ArgumentParser(prog="python -m src.data.handcrafted", description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)
    build_cache(load_config(a.config), SplitManifest.load(a.manifest), workers=a.workers)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
