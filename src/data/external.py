"""External test sets (v4): images from generators of OTHER families / newer tools, never seen in training.

Why: the S4 "unseen" generators of GenImage (Midjourney, VQDM) are diffusion models like the training generators
(ADM, SD v1.4), and all GenImage images share ImageNet content. S4 therefore measures generalisation inside one family.
An external set (FLUX, DALL-E 3, Midjourney v6, GANs, real photos that are not ImageNet, ...) measures it outside.

Input layout -- one folder per generator, real images in `real/` (any depth, jpg/jpeg/png/webp):

    <root>/real/...            label 0 (also accepted: 0_real, nature, authentic)
    <root>/flux/...            label 1, generator "flux"
    <root>/dalle3/...          label 1, generator "dalle3"

The images go through the SAME normalisation as GenImage (src.data.normalize.normalize_image: 224x224 PNG, AI
images JPEG-harmonised to the Real quality distribution), so the comparison with S1-S4 is like for like. With
`harmonize=False` AI images keep their original encoding: the "web" setting, closest to what the web demo receives.

Output: <out>/images/*.png and <out>/manifest.json, a normal SplitManifest whose records all have split
"ext_<name>" (is_unseen_generator=True). Every existing step then runs on it unchanged:

    python -m src.models.extract --manifest <out>/manifest.json --expert clip   --splits ext_<name>
    python -m src.models.extract --manifest <out>/manifest.json --expert dinov2 --splits ext_<name>
    python -m src.models.bundle  --manifest <out>/manifest.json --split ext_<name> --head-variant aug \\
                                 --out data/cache/bundles/aug/ext_<name>.pt

CLI: python -m src.data.external --root data/raw/external/mytest --name mytest [--no-harmonize] [--max-per-generator N]
"""
from __future__ import annotations

import argparse
import copy
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.core.config import load_config, resolve_path
from src.core.contracts import ImageRecord, SplitManifest
from src.core.logger import get_logger
from src.data.normalize import normalize_image, real_quality_pool
from src.data.validate import sha256_file, validate_image

log = get_logger(__name__)

REAL_NAMES = {"real", "0_real", "nature", "authentic"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def split_name(name: str) -> str:
    return f"ext_{name}"


def scan_folder(root: str | Path, max_per_generator: Optional[int] = None, seed: int = 0) -> Dict[str, List[Path]]:
    """{generator: [image paths]} from the folder layout above (sorted, seeded subset if max_per_generator)."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"external test set folder not found: {root}")
    out: Dict[str, List[Path]] = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXT)
        if not files:
            continue
        gen = "real" if d.name.lower() in REAL_NAMES else d.name.lower().replace(" ", "_")
        if max_per_generator and len(files) > max_per_generator:
            rng = np.random.default_rng([seed, zlib.crc32(gen.encode())])
            files = sorted(files[i] for i in rng.choice(len(files), max_per_generator, replace=False))
        out.setdefault(gen, []).extend(files)
    if not out:
        raise ValueError(f"no images found under {root}: expected sub-folders real/, <generator>/")
    return out


def ingest_external(root: str | Path, out: str | Path, name: str, cfg: Optional[dict] = None, harmonize: bool = True,
                    max_per_generator: Optional[int] = None, quality_pool: Optional[Sequence[int]] = None
                    ) -> SplitManifest:
    """Normalise an external folder into <out> and return its manifest (also saved to <out>/manifest.json).
    The JPEG-harmonisation quality pool comes from the set's own real images, else from `quality_pool`, else from the
    config fallback range."""
    cfg = copy.deepcopy(cfg or load_config())
    cfg["data"]["normalize"]["jpeg_harmonize"] = bool(harmonize)
    seed = int(cfg.get("seed", 0))
    out = Path(out)
    min_size = int(cfg["data"]["normalize"]["min_source_size"])
    groups = scan_folder(root, max_per_generator, seed)
    pool = list(quality_pool or [])
    if "real" in groups:
        pool = real_quality_pool(groups["real"]) or pool
    split = split_name(name)
    records: List[ImageRecord] = []
    seen: set = set()
    skipped = 0
    for gen, files in groups.items():
        is_real = gen == "real"
        for src in files:
            ok, why = validate_image(src, min_size)
            if not ok:
                skipped += 1
                log.debug("skip %s: %s", src, why)
                continue
            raw = sha256_file(src)
            if raw in seen:
                skipped += 1
                continue
            seen.add(raw)
            gid = f"{name}:{gen}:{raw[:16]}"
            dst = out / "images" / gen / f"{raw[:24]}.png"
            prov = normalize_image(src, dst, is_real=is_real, cfg=cfg, quality_pool=pool,
                                   rng=np.random.default_rng([seed, zlib.crc32(gid.encode())]))
            records.append(ImageRecord(sha256=sha256_file(dst), path=str(dst.relative_to(out)).replace("\\", "/"),
                                       label=0 if is_real else 1, generator=gen, source_group_id=gid, split=split,
                                       is_unseen_generator=not is_real, provenance={**prov, "external_set": name}))
    man = SplitManifest(records=records, seed=seed, root=str(out))
    man.meta.update({"source": "external", "external_set": name, "split": split, "harmonize": bool(harmonize),
                     "normalize_mode": cfg["data"]["normalize"].get("mode"), "raw_root": str(root)})
    man.save(out / "manifest.json")
    counts = {g: sum(1 for r in records if r.generator == g) for g in sorted({r.generator for r in records})}
    log.info("external set '%s' (%s): %d images %s, %d skipped -> %s", name,
             "harmonised" if harmonize else "raw encoding", len(records), counts, skipped, out / "manifest.json")
    return man


def default_out(cfg: dict, name: str, harmonize: bool = True) -> Path:
    return resolve_path(cfg, "cache_dir").parent / "external" / (name if harmonize else f"{name}_raw")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.data.external", description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True, help="folder with real/ and one sub-folder per generator")
    ap.add_argument("--name", required=True, help="short name, e.g. newgen (split becomes ext_<name>)")
    ap.add_argument("--out", default=None, help="default: data/external/<name>[_raw]")
    ap.add_argument("--no-harmonize", action="store_true", help="keep the original encoding of AI images")
    ap.add_argument("--max-per-generator", type=int, default=None)
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    out = a.out or default_out(cfg, a.name, not a.no_harmonize)
    ingest_external(a.root, out, a.name, cfg, harmonize=not a.no_harmonize, max_per_generator=a.max_per_generator)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
