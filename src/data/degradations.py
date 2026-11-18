"""Image degradations (JPEG / Gaussian blur / resize) for val_gating (random) and test_benchmark (fixed levels).

* val_gating: continuous random Q in [30,95], sigma in [0.5,3.0] (config degradation.val).
* test_benchmark: fixed S2 JPEG Q {85,65,40}; S3 resize 0.5 bilinear (down then back up) and
  blur (k=11,s=1.5), (k=17,s=2.5).
Blur uses EXACTLY the requested odd kernel size k (explicit separable Gaussian, reflect padding).
Order when combined: blur -> resize -> jpeg.
"""
from __future__ import annotations

import zlib
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
from scipy.ndimage import correlate1d

from src.core.contracts import (SCENARIO_TRAIN_AUG, SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL_FIT, SPLIT_VAL_VERIFY,
                                DegradationSpec, ImageRecord, SplitManifest)
from src.core.logger import get_logger
from src.data.normalize import jpeg_roundtrip
from src.data.validate import sha256_file

log = get_logger(__name__)


def gaussian_kernel_1d(k: int, sigma: float) -> np.ndarray:
    if k % 2 != 1 or k < 1:
        raise ValueError(f"kernel size must be odd, got {k}")
    x = np.arange(k, dtype=np.float64) - (k - 1) / 2
    w = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return w / w.sum()


def gaussian_blur(arr: np.ndarray, k: int, sigma: float) -> np.ndarray:
    """Separable Gaussian blur with an exact k-tap kernel. uint8 (H,W,3) in, uint8 out."""
    ker = gaussian_kernel_1d(k, sigma)
    a = arr.astype(np.float64)
    a = correlate1d(a, ker, axis=0, mode="reflect")
    a = correlate1d(a, ker, axis=1, mode="reflect")
    return np.clip(np.rint(a), 0, 255).astype(np.uint8)


def resize_down_up(img: Image.Image, factor: float) -> Image.Image:
    w, h = img.size
    small = img.resize((max(1, round(w * factor)), max(1, round(h * factor))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def apply_degradation(img: Image.Image, spec: DegradationSpec) -> Image.Image:
    out = img.convert("RGB")
    if spec.blur_sigma is not None:
        k = spec.blur_kernel or auto_kernel(spec.blur_sigma)
        out = Image.fromarray(gaussian_blur(np.asarray(out), k, spec.blur_sigma))
    if spec.resize_factor is not None:
        out = resize_down_up(out, spec.resize_factor)
    if spec.jpeg_quality is not None:
        out = jpeg_roundtrip(out, spec.jpeg_quality)
    return out


def auto_kernel(sigma: float) -> int:
    """Odd kernel covering +-3 sigma."""
    return int(2 * np.ceil(3 * sigma) + 1)


def sample_val_spec(rng: np.random.Generator, cfg: dict, kind: Optional[str] = None) -> DegradationSpec:
    """Random continuous degradation for val_gating."""
    v = cfg["degradation"]["val"]
    kind = kind or str(rng.choice(v["kinds"]))
    if kind == "jpeg":
        q = int(rng.integers(v["jpeg_quality_range"][0], v["jpeg_quality_range"][1] + 1))
        return DegradationSpec(jpeg_quality=q, name=f"val_jpeg_q{q}", scenario="val")
    if kind == "blur":
        s = float(rng.uniform(*v["blur_sigma_range"]))
        k = auto_kernel(s)
        return DegradationSpec(blur_sigma=round(s, 3), blur_kernel=k, name=f"val_blur_k{k}_s{s:.2f}", scenario="val")
    if kind == "resize":
        f = float(rng.uniform(*v["resize_factor_range"]))
        return DegradationSpec(resize_factor=round(f, 3), name=f"val_resize_{f:.2f}", scenario="val")
    raise ValueError(f"unknown degradation kind {kind!r}")


def sample_train_aug_spec(rng: np.random.Generator, cfg: dict, index: int = 0) -> Optional[DegradationSpec]:
    """UnivFD/CNNSpot-style train augmentation draw (config degradation.train_aug; METHOD_SPECS question b).

    Gaussian blur with prob blur_prob (sigma ~ U[blur_sigma_range]) and, independently, JPEG with prob jpeg_prob
    (Q ~ U[jpeg_quality_range], inclusive ints); applied blur -> jpeg. Returns None when neither fires.
    The draw order is fixed (blur coin, sigma, jpeg coin, Q - always consumed) so it is reproducible.
    """
    a = cfg["degradation"]["train_aug"]
    do_blur = bool(rng.random() < float(a["blur_prob"]))
    sigma = float(rng.uniform(*a["blur_sigma_range"]))
    do_jpeg = bool(rng.random() < float(a["jpeg_prob"]))
    q = int(rng.integers(a["jpeg_quality_range"][0], a["jpeg_quality_range"][1] + 1))
    if not (do_blur or do_jpeg):
        return None
    parts, spec = [], DegradationSpec(scenario=SCENARIO_TRAIN_AUG)
    if do_blur:
        sigma = round(max(sigma, 0.01), 3)         # sigma == 0 would be a no-op / division by zero
        spec.blur_sigma, spec.blur_kernel = sigma, auto_kernel(sigma)
        parts.append(f"blur_s{sigma:.2f}")
    if do_jpeg:
        spec.jpeg_quality = q
        parts.append(f"jpeg_q{q}")
    spec.name = f"train_aug{index}_" + "_".join(parts)
    return spec


def fixed_test_specs(cfg: dict) -> Dict[str, List[DegradationSpec]]:
    """Fixed-level degradations: {'S2': [...], 'S3': [...]}."""
    t = cfg["degradation"]["test"]
    s2 = [DegradationSpec(jpeg_quality=int(q), name=f"jpeg_q{q}", scenario="S2") for q in t["s2_jpeg_quality"]]
    rs = t["s3_resize"]
    s3 = [DegradationSpec(resize_factor=float(rs["factor"]), name=f"resize_{rs['factor']}", scenario="S3")]
    for b in t["s3_blur"]:
        s3.append(DegradationSpec(blur_sigma=float(b["sigma"]), blur_kernel=int(b["kernel"]),
                                  name=f"blur_k{b['kernel']}_s{b['sigma']}", scenario="S3"))
    return {"S2": s2, "S3": s3}


def materialize_variants(manifest: SplitManifest, out_dir: str | Path, cfg: dict, seed: Optional[int] = None) -> List[ImageRecord]:
    """Create degraded PNG variants and append their records to `manifest`.

    * val_gating_fit / val_gating_verify clean sources: `variants_per_source` random val degradations.
    * test_benchmark clean sources: all fixed S2 + S3 levels.
    * train_experts clean sources: `degradation.train_aug.variants_per_source` train_aug variants (blur w.p. 0.5,
      JPEG w.p. 0.5, independently - UnivFD style). A draw where NEITHER fires produces NO new record/file
      (a byte-identical copy would duplicate a SHA-256); instead the draw is logged on the source record as
      provenance["train_aug_draws"], so the effective aug rate is exactly the UnivFD one: the clean original
      stands for the "no augmentation" case. Seeds derive from (seed, group id, 1).
    * other splits: none (heads/experts train on clean only; S4 pools stay clean).
    Variants inherit source_group_id and split (so they never cross splits).
    Per-source randomness derives from (seed, group id) => independent of iteration order.
    """
    seed = cfg.get("seed", 0) if seed is None else seed
    out_dir = Path(out_dir)
    root = Path(manifest.root) if manifest.root else out_dir
    fixed = fixed_test_specs(cfg)
    n_var = int(cfg["degradation"]["val"]["variants_per_source"])
    n_aug = int(cfg["degradation"].get("train_aug", {}).get("variants_per_source", 0))
    new: List[ImageRecord] = []
    seen_sha = {r.sha256 for r in manifest.records}
    for rec in [r for r in manifest.records if r.is_clean]:
        if rec.split in (SPLIT_VAL_FIT, SPLIT_VAL_VERIFY):
            rng = np.random.default_rng([seed, zlib.crc32(rec.source_group_id.encode())])
            kinds = list(cfg["degradation"]["val"]["kinds"])
            specs = [sample_val_spec(rng, cfg, kind=kinds[i % len(kinds)]) for i in range(n_var)]
        elif rec.split == SPLIT_TEST:
            specs = fixed["S2"] + fixed["S3"]
        elif rec.split == SPLIT_TRAIN and n_aug > 0:
            rng = np.random.default_rng([seed, zlib.crc32(rec.source_group_id.encode()), 1])
            draws = [sample_train_aug_spec(rng, cfg, i) for i in range(n_aug)]
            rec.provenance["train_aug_draws"] = [d.name if d else "none" for d in draws]
            specs = [d for d in draws if d is not None]
        else:
            continue
        if not specs:
            continue
        with Image.open(manifest.abs_path(rec)) as im:
            im.load()
            base = im.convert("RGB")
        for spec in specs:
            dst_dir = out_dir / "variants" / rec.split
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / f"{rec.sha256[:16]}__{spec.name}.png"
            apply_degradation(base, spec).save(dst, format="PNG", compress_level=0)
            sha = sha256_file(dst)
            if sha in seen_sha:    # degradation was a pixel-level no-op (e.g. tiny sigma): never duplicate a SHA-256
                dst.unlink()
                if spec.scenario == SCENARIO_TRAIN_AUG and rec.provenance.get("train_aug_draws"):
                    rec.provenance["train_aug_draws"] = [f"{n}(no-op)" if n == spec.name else n
                                                         for n in rec.provenance["train_aug_draws"]]
                log.info("variant %s of %s is identical to an existing image; skipped", spec.name, rec.sha256[:8])
                continue
            seen_sha.add(sha)
            try:
                rel = str(dst.relative_to(root))
            except ValueError:
                rel = str(dst)
            new.append(ImageRecord(
                sha256=sha, path=rel.replace("\\", "/"), label=rec.label, generator=rec.generator,
                source_group_id=rec.source_group_id, split=rec.split, degradation=spec,
                parent_sha256=rec.sha256, is_unseen_generator=rec.is_unseen_generator,
                width=rec.width, height=rec.height,
                provenance={"parent_path": rec.path}))
    manifest.records.extend(new)
    log.info("materialized %d degraded variants", len(new))
    return new
