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
from typing import Dict, Optional, Tuple

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


VIEWS = ("clip", "dinov2", "freq")


def _views_and_quality(man: SplitManifest, records, cfg: dict) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    from src.data.quality import extract_quality_features
    from src.models.base import load_image_array
    views = {k: [] for k in VIEWS}
    quality = []
    for r in records:
        arr = load_image_array(str(man.abs_path(r)))
        for k, v in _raw_views(arr).items():
            views[k].append(v)
        quality.append(extract_quality_features(arr, cfg).to_array())
    return {k: np.stack(v).astype(np.float32) for k, v in views.items()}, np.stack(quality).astype(np.float32)


def _project(views: Dict[str, np.ndarray], seed: int, stats: Optional[dict] = None
             ) -> Tuple[Dict[str, np.ndarray], dict]:
    """Standardise each raw view (with the given stats, else its own) and project it; returns (features, stats)."""
    stats = dict(stats or {})
    feats = {}
    for j, k in enumerate(VIEWS):
        V = views[k]
        if k not in stats:
            stats[k] = {"mu": V.mean(0).tolist(), "sd": np.maximum(V.std(0), 1e-6).tolist()}
        V = (V - np.asarray(stats[k]["mu"], np.float32)) / np.asarray(stats[k]["sd"], np.float32)
        feats[k] = np.tanh(V @ _projector(V.shape[1], DEMO_DIMS[k], seed + j)) if k in DEMO_DIMS else V
    return feats, stats


def _freq_cache(cfg: dict):
    from src.core.config import resolve_path
    from src.models.cache import FeatureCache
    return FeatureCache(resolve_path(cfg, "cache_dir") / "features" / "demo_freq.pt", "demo_freq", "demo",
                        dtype=torch.float32)


def _heads(cfg: dict, man: SplitManifest, hv: str, feat: Dict[str, Dict[str, np.ndarray]]):
    """Expert heads + centroids of the demo, trained on train_records(man, hv). feat: {view: {sha: vector}}."""
    from src.models.heads import train_linear_head, train_records
    from src.ood.profiler import compute_centroid
    tr = train_records(man, hv)
    y_tr = torch.tensor([r.label for r in tr])
    heads, cents = {}, {}
    for k in EXPERT_NAMES:
        F = torch.from_numpy(np.stack([feat[k][r.sha256] for r in tr]))
        heads[k] = train_linear_head(F, y_tr, cfg)
        cents[k] = compute_centroid(F[[i for i, r in enumerate(tr) if r.is_clean]])
    return heads, cents


def _bundle_blob(cfg: dict, records, feat, quality: np.ndarray, heads, cents, split: str, hv: str) -> dict:
    from src.ood.profiler import cosine_distance
    logits = torch.zeros(len(records), 3)
    cosd = torch.zeros(len(records), 3)
    for j, k in enumerate(EXPERT_NAMES):
        F = torch.from_numpy(np.stack([feat[k][r.sha256] for r in records]))
        with torch.no_grad():
            logits[:, j] = heads[k](F)
        cosd[:, j] = cosine_distance(F, cents[k])
    return {"sha256": [r.sha256 for r in records], "label": torch.tensor([r.label for r in records]),
            "generator": [r.generator for r in records],
            "degradation": [None if r.is_clean else r.degradation.name for r in records],
            "scenario": [None if r.is_clean else r.degradation.scenario for r in records],
            "logits": logits, "cosdist": cosd, "quality": torch.from_numpy(quality),
            "meta": {"cache_version": cfg["experts"]["extract"]["cache_version"], "split": split,
                     "head_variant": hv, "demo": True, "model_ids": {}, "created": "demo"}}


def _load_feat(cfg: dict, records) -> Dict[str, Dict[str, np.ndarray]]:
    from src.models.cache import FeatureCache
    caches = {"clip": FeatureCache.open(cfg, "clip"), "dinov2": FeatureCache.open(cfg, "dinov2"),
              "freq": _freq_cache(cfg)}
    shas = [r.sha256 for r in records]
    return {k: dict(zip(shas, c.features_for(shas).numpy())) for k, c in caches.items()}


def build_demo(out_dir: str | Path, scale: int = 6, cfg: dict | None = None, seed: int = 0
               ) -> Tuple[dict, SplitManifest]:
    """Create images, feature caches (clip, dinov2) and bundles (clean & aug head variants) under out_dir.
    Returns (cfg pointing its cache_dir to out_dir/cache, manifest). Reuses an existing demo in out_dir."""
    from src.data.toy import load_or_build_toy
    from src.learning.bundle_io import bundle_path
    from src.models.cache import FeatureCache

    out_dir = Path(out_dir).resolve()
    base = copy.deepcopy(cfg or load_config())
    counts = {k: int(v) * scale for k, v in base["toy"]["counts"].items()}
    base = deep_merge(base, {"toy": {"counts": counts, "n_total": sum(counts.values())},
                             "paths": {"cache_dir": str(out_dir / "cache")}})
    base["toy"]["split_overrides"] = {"s4": {"support_pool_frac": 0.4, "support_pool_min": 4, "real_support_frac": 0.1,
                                             "real_eval_frac": 0.15, "real_min": 10}}
    base["demo_seed"] = seed
    man = load_or_build_toy(out_dir / "images", base, extra_key=f"demo{scale}")
    caches = {n: FeatureCache.open(base, n) for n in ("clip", "dinov2")}
    fcache = _freq_cache(base)
    done = all(len(c) >= len(man.records) for c in (*caches.values(), fcache)) and \
        "demo_stats" in caches["clip"].meta and all(
        bundle_path(base, s, hv).exists() for s in ALL_SPLITS for hv in ("clean", "aug"))
    if done:
        log.info("demo reused from %s", out_dir)
        return base, man

    log.info("demo: computing pseudo-embeddings for %d images", len(man.records))
    views, quality = _views_and_quality(man, man.records, base)
    feats, stats = _project(views, seed)
    for k, c in (("clip", caches["clip"]), ("dinov2", caches["dinov2"]), ("freq", fcache)):
        c.meta["model_id"] = f"demo_{k}"
        for r, f in zip(man.records, feats[k]):
            c.put(r.sha256, torch.from_numpy(np.ascontiguousarray(f, dtype=np.float32)))
    caches["clip"].meta["demo_stats"] = stats
    for c in (*caches.values(), fcache):
        c.save()

    feat = {k: {r.sha256: feats[k][i] for i, r in enumerate(man.records)} for k in VIEWS}
    qrow = {r.sha256: quality[i] for i, r in enumerate(man.records)}
    for hv in ("clean", "aug"):
        heads, cents = _heads(base, man, hv, feat)
        for split in ALL_SPLITS:
            recs = man.by_split(split)
            if not recs:
                continue
            blob = _bundle_blob(base, recs, feat, np.stack([qrow[r.sha256] for r in recs]), heads, cents, split, hv)
            p = bundle_path(base, split, hv)
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save(blob, p)
    log.info("demo ready at %s", out_dir)
    return base, man


# ---------------------------------------------------------------------------------------- demo external test set
DEMO_EXTERNAL_GENERATORS = ("newgen_a", "newgen_b")


def build_demo_external_raw(out_dir: str | Path, n_per_generator: int = 40, n_real: int = 60, seed: int = 0) -> Path:
    """Raw folder in the external layout (real/ as JPEG, one PNG folder per new synthetic generator) made by the toy
    synthesiser with generator names it has never been trained on. For notebooks and tests only."""
    from src.data.toy import _synth
    out = Path(out_dir)
    rng = np.random.default_rng(seed + 991)
    for gen, n in [("real", n_real)] + [(g, n_per_generator) for g in DEMO_EXTERNAL_GENERATORS]:
        d = out / gen
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            arr = _synth(rng, int(rng.integers(256, 340)), gen, "real")
            if gen == "real":
                Image.fromarray(arr).save(d / f"{i:04d}.jpg", quality=int(rng.integers(80, 96)))
            else:
                Image.fromarray(arr).save(d / f"{i:04d}.png")
    return out


def build_demo_external(cfg: dict, man: SplitManifest, ext_man: SplitManifest) -> None:
    """Demo stand-in for extract + bundle on an external manifest: pseudo-embeddings with the main demo's
    standardisation, written to the same caches, and external bundles (clean & aug) in the external bundle paths."""
    from src.evaluation.external_eval import external_bundle_path, external_split
    from src.models.cache import FeatureCache
    caches = {"clip": FeatureCache.open(cfg, "clip"), "dinov2": FeatureCache.open(cfg, "dinov2"),
              "freq": _freq_cache(cfg)}
    stats = caches["clip"].meta.get("demo_stats")
    if stats is None:
        raise RuntimeError("build_demo must run first (demo standardisation stats missing)")
    split = external_split(ext_man)
    views, quality = _views_and_quality(ext_man, ext_man.records, cfg)
    feats, _ = _project(views, int(cfg.get("demo_seed", 0)), stats)
    for k, c in caches.items():
        for r, f in zip(ext_man.records, feats[k]):
            c.put(r.sha256, torch.from_numpy(np.ascontiguousarray(f, dtype=np.float32)))
        c.save()
    feat_main = _load_feat(cfg, man.records)
    feat_ext = {k: {r.sha256: feats[k][i] for i, r in enumerate(ext_man.records)} for k in VIEWS}
    for hv in ("clean", "aug"):
        heads, cents = _heads(cfg, man, hv, feat_main)
        blob = _bundle_blob(cfg, ext_man.records, feat_ext, quality, heads, cents, split, hv)
        p = external_bundle_path(cfg, split, hv)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(blob, p)
    log.info("demo external set '%s': %d images, bundles written", split, len(ext_man.records))
