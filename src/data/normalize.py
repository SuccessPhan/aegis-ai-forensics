"""Normalisation: Center-Crop 224x224 (offset multiple of 8) -> lossless PNG, with optional JPEG harmonisation.

Why jpeg_harmonize exists
-------------------------
In GenImage the Real images (ImageNet) are JPEG files; even when re-saved as PNG, their pixels still carry
JPEG block/quantisation history. AI images are PNG with no such history. A detector (or a 3-feature
logistic regression) could then separate classes by compression history alone - a dataset shortcut.
So, when enabled, each AI image is JPEG-compressed ONCE at a quality sampled from the Real images'
quality distribution, BEFORE cropping (same compress-then-crop order the Real images went through).

Reading the quantisation tables of Real images to build that distribution is allowed ONLY here, at data
preparation. It must never become a model feature (quality.py works from pixels only).

Normalisation modes (cfg data.normalize.mode), applied identically to Real and AI
--------------------------------------------------------------------------------
* "crop":        [jpeg_harmonize for AI] -> center-crop 224 (no resizing).
* "resize_crop": [jpeg_harmonize for AI] -> resize short side to resize_short_side (256) with PIL LANCZOS
                 (antialiased) -> center-crop 224.
  Order rationale: the Real images' history is "JPEG-compressed at source, then processed"; so AI images are
  compressed FIRST (harmonize) and only then resized/cropped, mirroring that history. Resizing before the
  harmonising compression would leave the AI JPEG grid aligned with the final pixels while Real's is not.
  Smaller-than-crop sources are still rejected upstream (min_source_size); sources whose short side is
  between 224 and 256 get (antialiased) upsampled.

Crop offsets are rounded down to a multiple of 8 so the 8x8 JPEG grid stays aligned after cropping.
In "crop" mode images are never resized here: smaller-than-crop images are rejected.
In "resize_crop" mode the multiple-of-8 rounding no longer aligns anything with a source JPEG grid (the resize
destroys it); it is kept only for consistency between the two modes.
(Note: with 4:2:0 chroma subsampling the MCU is 16 px; the brief fixes 8, which aligns the luma grid.)
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from src.core.logger import get_logger

log = get_logger(__name__)

# Standard JPEG (Annex K) luminance table, used to invert PIL's quality scaling.
_STD_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], dtype=np.float64)


def _scaled_table(q: int) -> np.ndarray:
    s = 5000 / q if q < 50 else 200 - 2 * q
    return np.clip(np.floor((_STD_LUMA * s + 50) / 100), 1, 255)


def estimate_quality_from_qtable(img: Image.Image) -> Optional[int]:
    """Quality (1..100) from a JPEG luminance quantisation table; None if the image has none.
    DATA PREPARATION ONLY - never use as a model feature."""
    qt = getattr(img, "quantization", None)
    if not qt or 0 not in qt:
        return None
    table = np.array(qt[0], dtype=np.float64)
    if table.size != 64:
        return None
    best = min(range(1, 101), key=lambda q: float(np.abs(_scaled_table(q) - table).sum()))
    return int(best)


def real_quality_pool(real_paths: Sequence[str | Path]) -> List[int]:
    """JPEG quality values of Real source files (those that are JPEGs)."""
    out: List[int] = []
    for p in real_paths:
        with Image.open(p) as im:
            q = estimate_quality_from_qtable(im)
        if q is not None:
            out.append(q)
    return out


def crop_offset(length: int, size: int, multiple: int = 8) -> int:
    """Centered offset rounded DOWN to a multiple of `multiple` (stays within [0, length-size])."""
    if length < size:
        raise ValueError(f"image side {length} < crop size {size}")
    return ((length - size) // 2 // multiple) * multiple


def center_crop(img: Image.Image, size: int = 224, multiple: int = 8) -> Tuple[Image.Image, Tuple[int, int]]:
    """Crop (no resizing). Returns (cropped, (left, top))."""
    w, h = img.size
    left, top = crop_offset(w, size, multiple), crop_offset(h, size, multiple)
    return img.crop((left, top, left + size, top + size)), (left, top)


def resize_short_side(img: Image.Image, short: int) -> Image.Image:
    """Resize so the SHORT side equals `short` (aspect kept), PIL LANCZOS (antialiased)."""
    w, h = img.size
    if w <= h:
        nw, nh = short, max(1, round(h * short / w))
    else:
        nw, nh = max(1, round(w * short / h)), short
    if (nw, nh) == (w, h):
        return img
    return img.resize((nw, nh), Image.LANCZOS)


def jpeg_roundtrip(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    out = Image.open(buf)
    out.load()
    return out


def sample_harmonize_quality(pool: Sequence[int], rng: np.random.Generator,
                             fallback_range: Sequence[int], clip: Sequence[int]) -> int:
    """Draw Q from the Real quality pool (or the config fallback range if the pool is empty)."""
    if len(pool) > 0:
        q = int(rng.choice(np.asarray(pool)))
    else:
        q = int(rng.integers(fallback_range[0], fallback_range[1] + 1))
    return int(min(max(q, clip[0]), clip[1]))


def normalize_image(src: str | Path, dst: str | Path, *, is_real: bool, cfg: dict,
                    quality_pool: Sequence[int] = (), rng: Optional[np.random.Generator] = None) -> dict:
    """Normalise one raw image to a 224x224 lossless PNG at `dst`. Returns provenance info."""
    ncfg = cfg["data"]["normalize"]
    size = int(ncfg["crop_size"])
    rng = rng or np.random.default_rng(cfg.get("seed", 0))
    with Image.open(src) as im:
        im.load()
        orig_size, orig_fmt = im.size, im.format
        rgb = im.convert("RGB")
    harm_q: Optional[int] = None
    if ncfg.get("jpeg_harmonize", True) and not is_real:
        harm_q = sample_harmonize_quality(quality_pool, rng, ncfg["harmonize_fallback_quality_range"],
                                          ncfg["harmonize_quality_clip"])
        rgb = jpeg_roundtrip(rgb, harm_q)          # compress first, then crop (as Real images were)
    mode = str(ncfg.get("mode", "crop"))
    if mode not in ("crop", "resize_crop"):
        raise ValueError(f"unknown data.normalize.mode {mode!r} (expected 'crop' or 'resize_crop')")
    if mode == "resize_crop":
        rgb = resize_short_side(rgb, int(ncfg.get("resize_short_side", 256)))
    cropped, (left, top) = center_crop(rgb, size, int(ncfg["crop_offset_multiple"]))
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(dst, format="PNG", compress_level=int(ncfg.get("png_compress_level", 0)))
    return {"source_path": str(src), "orig_width": orig_size[0], "orig_height": orig_size[1],
            "orig_format": orig_fmt, "crop_left": left, "crop_top": top, "harmonize_quality": harm_q,
            "normalize_mode": mode, "pre_crop_width": rgb.size[0], "pre_crop_height": rgb.size[1]}
