"""Synthetic toy dataset (100 images) that exercises the FULL data pipeline.

Raw images are synthesised at random sizes (Real saved as JPEG, fakes as PNG - mimicking the GenImage shortcut),
then pushed through validate -> dedup -> jpeg_harmonize -> center-crop 224 -> PNG, split, and degraded variants.
Images are synthetic textures, NOT real photos / real generator output: for pipeline & unit tests only.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from src.core.config import deep_merge, load_config
from src.core.contracts import LABEL_FAKE, LABEL_REAL, SplitManifest
from src.core.logger import get_logger
from src.data.degradations import materialize_variants
from src.data.pipeline import RawItem, prepare_records
from src.data.split import assign_splits

log = get_logger(__name__)

# per-generator (grid period, amplitude, colour tint): a synthetic, learnable "fingerprint"
_SIGNATURE = {"adm": (4, 4.0, (6, -3, 0)), "sd_v1_4": (8, 3.0, (-4, 4, 2)),
              "midjourney": (5, 5.0, (0, 6, -6)), "glide": (3, 4.0, (5, 0, 5)), "vqdm": (6, 4.5, (4, -5, 5))}


def _fallback_signature(generator: str) -> tuple:
    """Deterministic signature for a generator name without a hand-written profile (toy follows the config list)."""
    import zlib
    r = np.random.default_rng(zlib.crc32(generator.encode()))
    return (int(r.integers(3, 10)), float(r.uniform(3, 5)), tuple(int(v) for v in r.integers(-6, 7, 3)))


def _synth(rng: np.random.Generator, size: int, generator: str, real_name: str) -> np.ndarray:
    coarse = rng.random((10, 10, 3)) * 255
    base = np.asarray(Image.fromarray(coarse.astype(np.uint8)).resize((size, size), Image.BICUBIC), dtype=np.float64)
    yy, xx = np.mgrid[0:size, 0:size]
    base += (rng.uniform(-30, 30) * (xx / size) + rng.uniform(-30, 30) * (yy / size))[..., None]
    base += rng.normal(0, 3.0 if generator == real_name else 2.0, base.shape)
    if generator != real_name:
        period, amp, tint = _SIGNATURE.get(generator) or _fallback_signature(generator)
        base += amp * np.sin(2 * np.pi * xx / period)[..., None] + np.array(tint)
    return np.clip(base, 0, 255).astype(np.uint8)


def build_toy(out_dir: str | Path, cfg: Optional[dict] = None, seed: Optional[int] = None,
              with_variants: bool = True) -> SplitManifest:
    """Create the toy dataset under out_dir (images/, variants/, manifest.json) and return the manifest."""
    cfg = cfg or load_config()
    cfg = deep_merge(cfg, {"split": cfg["toy"]["split_overrides"]})
    seed = int(cfg["seed"]) if seed is None else seed
    out_dir = Path(out_dir)
    if out_dir.exists():
        for sub in ("images", "variants"):
            shutil.rmtree(out_dir / sub, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    real_name = cfg["data"]["real_name"]
    lo, hi = cfg["toy"]["raw_size_range"]
    rng = np.random.default_rng(seed)

    items = []
    with tempfile.TemporaryDirectory() as tmp:
        for gen, n in cfg["toy"]["counts"].items():
            for i in range(n):
                size = int(rng.integers(lo, hi + 1))
                arr = _synth(rng, size, gen, real_name)
                if gen == real_name:
                    p = Path(tmp) / f"{gen}_{i:03d}.jpg"
                    Image.fromarray(arr).save(p, format="JPEG", quality=int(rng.integers(60, 96)))
                    items.append(RawItem(p, gen, LABEL_REAL))
                else:
                    p = Path(tmp) / f"{gen}_{i:03d}.png"
                    Image.fromarray(arr).save(p, format="PNG")
                    items.append(RawItem(p, gen, LABEL_FAKE))
        records = prepare_records(items, out_dir, cfg)
    for r in records:  # do not leak temp paths into provenance
        r.provenance["source_path"] = Path(r.provenance["source_path"]).name
    man = assign_splits(records, cfg, seed=seed, root=str(out_dir))
    if with_variants:
        materialize_variants(man, out_dir, cfg, seed=seed)
    man.meta["toy"] = True
    man.save(out_dir / "manifest.json")
    log.info("toy dataset ready at %s (%d clean, %d total records)", out_dir,
             sum(r.is_clean for r in man.records), len(man.records))
    return man


_KEY_FILE = "toy_cache_key.json"
_KEY_CFG_SECTIONS = ("seed", "toy", "split", "degradation", "data", "quality_features", "paths")


def toy_cache_key(cfg: dict, extra: str = "") -> str:
    """Hash of the config sections that influence the toy dataset + the source of the code that builds it."""
    import hashlib
    import json
    h = hashlib.sha256()
    h.update(json.dumps({k: cfg.get(k) for k in _KEY_CFG_SECTIONS}, sort_keys=True, default=str).encode())
    here = Path(__file__).parent
    for f in sorted(list(here.glob("*.py")) + [here.parent / "core" / "contracts.py"]):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    h.update(extra.encode())
    return h.hexdigest()


def load_or_build_toy(out_dir: str | Path, cfg: Optional[dict] = None, extra_key: str = "") -> SplitManifest:
    """Reuse the toy dataset in out_dir if its stored key matches and every file exists; otherwise rebuild."""
    import json
    cfg = cfg or load_config()
    out_dir = Path(out_dir)
    key = toy_cache_key(cfg, extra_key)
    key_path, man_path = out_dir / _KEY_FILE, out_dir / "manifest.json"
    try:
        if key_path.exists() and man_path.exists() and json.loads(key_path.read_text())["key"] == key:
            man = SplitManifest.load(man_path)
            if man.records and all(man.abs_path(r).exists() for r in man.records):
                log.info("toy dataset reused from %s", out_dir)
                return man
    except Exception as exc:  # corrupt / partial cache -> rebuild
        log.warning("toy cache unusable (%s); rebuilding", exc)
    if key_path.exists():
        key_path.unlink()
    man = build_toy(out_dir, cfg)
    key_path.write_text(json.dumps({"key": key}))   # written last: a crash mid-build leaves no valid key
    return man
