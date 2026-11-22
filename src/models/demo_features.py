"""Demo mode: a synthetic dataset + image-derived pseudo-embeddings + bundles, so the ML-comparison notebook and its
tests run on a laptop without a GPU, without the 5 GB GenImage download and without CLIP / DINOv2 weights.

  * images   : the toy generator of src.data.toy, scaled up (`scale` x the toy counts), with all degraded variants;
  * "clip"   : fixed random projection of a 6x6 thumbnail + colour statistics + low-frequency spectrum (global appearance);
  * "dinov2" : fixed random projection of log-FFT radial spectrum + local patch statistics (texture; hurt by JPEG/blur);
  * "freq"   : log-FFT radial spectrum only, used for the third expert's logit in the demo bundles.
Feature files use the real cache format (src.models.cache.FeatureCache) and bundles use the real hand-off format
(src.learning.bundle_io), so every downstream module runs unchanged. NUMBERS FROM DEMO MODE MEAN NOTHING: the images
are synthetic textures, not photos or generator output.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from PIL import Image

from src.core.config import deep_merge, load_config
from src.core.contracts import ALL_SPLITS, EXPERT_NAMES, SplitManifest
from src.core.logger import get_logger

log = get_logger(__name__)

DEMO_DIMS = {"clip": 256, "dinov2": 192}


def _radial_spectrum(gray: np.ndarray, bins: int = 48) -> np.ndarray:
    f = np.abs(np.fft.fftshift(np.fft.fft2(gray - gray.mean())))
    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy - h / 2, xx - w / 2)
    idx = np.minimum((r / r.max() * bins).astype(int), bins - 1)
    return np.log1p(np.bincount(idx.ravel(), f.ravel(), bins) / np.maximum(np.bincount(idx.ravel(), None, bins), 1))


def _raw_views(arr: np.ndarray) -> Dict[str, np.ndarray]:
    img = Image.fromarray(arr)
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    thumb = np.asarray(img.resize((6, 6), Image.BILINEAR), dtype=np.float32).ravel() / 255.0
    gray = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    spec = _radial_spectrum(gray)
    chroma = rgb - rgb.mean(axis=2, keepdims=True)                     # colour cast (the toy generators' tint)
    colour = np.concatenate([rgb.mean((0, 1)), rgb.std((0, 1)), chroma.mean((0, 1)) * 20,
                             np.concatenate([np.histogram(chroma[..., c], 8, (-0.15, 0.15))[0] / chroma[..., c].size
                                             for c in range(3)])])
    global_view = np.concatenate([thumb, colour, np.tile(spec[:16], 6)])  # "clip": appearance + colour + low-freq spectrum
    p = gray[: gray.shape[0] // 8 * 8, : gray.shape[1] // 8 * 8].reshape(gray.shape[0] // 8, 8, -1, 8)
    patch = np.stack([p.std(axis=(1, 3)).ravel(), np.abs(np.diff(p, axis=1)).mean(axis=(1, 3)).ravel()])
    patch = np.histogram(patch[0], 16, (0, 0.3))[0] / patch.shape[1]
    return {"clip": global_view, "dinov2": np.concatenate([spec, patch]), "freq": spec}


def _projector(dim_in: int, dim_out: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, 1 / np.sqrt(dim_in), (dim_in, dim_out)).astype(np.float32)


def build_demo(out_dir: str | Path, scale: int = 6, cfg: dict | None = None, seed: int = 0
               ) -> Tuple[dict, SplitManifest]:
    """Create images, feature caches (clip, dinov2) and bundles (clean & aug head variants) under out_dir.
    Returns (cfg pointing its cache_dir to out_dir/cache, manifest). Reuses an existing demo in out_dir."""
    from src.data.quality import extract_quality_features
    from src.data.toy import load_or_build_toy
    from src.learning.bundle_io import bundle_path
    from src.models.base import load_image_array
    from src.models.cache import FeatureCache
    from src.models.heads import train_linear_head, train_records
    from src.ood.profiler import compute_centroid, cosine_distance

    out_dir = Path(out_dir).resolve()
    base = copy.deepcopy(cfg or load_config())
    counts = {k: int(v) * scale for k, v in base["toy"]["counts"].items()}
    base = deep_merge(base, {"toy": {"counts": counts, "n_total": sum(counts.values())},
                             "paths": {"cache_dir": str(out_dir / "cache")}})
    base["toy"]["split_overrides"] = {"s4": {"support_pool_frac": 0.4, "support_pool_min": 4, "real_support_frac": 0.1,
                                             "real_eval_frac": 0.15, "real_min": 10}}
    man = load_or_build_toy(out_dir / "images", base, extra_key=f"demo{scale}")
    caches = {n: FeatureCache.open(base, n) for n in ("clip", "dinov2")}
    done = all(len(c) == len(man.records) for c in caches.values()) and all(
        bundle_path(base, s, hv).exists() for s in ALL_SPLITS for hv in ("clean", "aug"))
    if done:
        log.info("demo reused from %s", out_dir)
        return base, man

    log.info("demo: computing pseudo-embeddings for %d images", len(man.records))
    views = {"clip": [], "dinov2": [], "freq": []}
    quality = []
    for r in man.records:
        arr = load_image_array(str(man.abs_path(r)))
        for k, v in _raw_views(arr).items():
            views[k].append(v)
        quality.append(extract_quality_features(arr, base).to_array())
    feats = {}
    for j, k in enumerate(("clip", "dinov2", "freq")):
        V = np.stack(views[k]).astype(np.float32)
        V = (V - V.mean(0)) / np.maximum(V.std(0), 1e-6)
        dim = DEMO_DIMS.get(k, V.shape[1])
        feats[k] = np.tanh(V @ _projector(V.shape[1], dim, seed + j)) if k in DEMO_DIMS else V
    for k in ("clip", "dinov2"):
        caches[k].meta["model_id"] = f"demo_{k}"
        for r, f in zip(man.records, feats[k]):
            caches[k].put(r.sha256, torch.from_numpy(f))
        caches[k].save()

    row = {r.sha256: i for i, r in enumerate(man.records)}
    quality = np.stack(quality).astype(np.float32)
    for hv in ("clean", "aug"):
        tr = train_records(man, hv)
        ti = [row[r.sha256] for r in tr]
        y_tr = torch.tensor([r.label for r in tr])
        clean_i = [row[r.sha256] for r in tr if r.is_clean]
        heads, cents = {}, {}
        for k in EXPERT_NAMES:
            F = torch.from_numpy(feats[k])
            heads[k] = train_linear_head(F[ti], y_tr, base)
            cents[k] = compute_centroid(F[clean_i])
        for split in ALL_SPLITS:
            recs = man.by_split(split)
            if not recs:
                continue
            ii = [row[r.sha256] for r in recs]
            logits = torch.zeros(len(ii), 3)
            cosd = torch.zeros(len(ii), 3)
            for j, k in enumerate(EXPERT_NAMES):
                F = torch.from_numpy(feats[k][ii])
                with torch.no_grad():
                    logits[:, j] = heads[k](F)
                cosd[:, j] = cosine_distance(F, cents[k])
            blob = {"sha256": [r.sha256 for r in recs], "label": torch.tensor([r.label for r in recs]),
                    "generator": [r.generator for r in recs],
                    "degradation": [None if r.is_clean else r.degradation.name for r in recs],
                    "scenario": [None if r.is_clean else r.degradation.scenario for r in recs],
                    "logits": logits, "cosdist": cosd, "quality": torch.from_numpy(quality[ii]),
                    "meta": {"cache_version": base["experts"]["extract"]["cache_version"], "split": split,
                             "head_variant": hv, "demo": True, "model_ids": {}, "created": "demo"}}
            p = bundle_path(base, split, hv)
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save(blob, p)
    log.info("demo ready at %s", out_dir)
    return base, man
