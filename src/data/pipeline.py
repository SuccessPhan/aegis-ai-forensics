"""End-to-end data preparation: raw items -> validate -> dedup -> normalise -> ImageRecords (with provenance).

Splitting and degraded variants are separate steps (split.assign_splits, degradations.materialize_variants).
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

from src.core.contracts import LABEL_FAKE, LABEL_REAL, ImageRecord
from src.core.logger import get_logger
from src.data.normalize import normalize_image, real_quality_pool
from src.data.validate import dedup_records, sha256_file, validate_image

log = get_logger(__name__)


@dataclass
class RawItem:
    path: Path
    generator: str      # cfg data.real_name for Real images
    label: int


def prepare_records(items: Sequence[RawItem], out_dir: str | Path, cfg: dict) -> List[ImageRecord]:
    """Normalise raw items into out_dir/images/*.png and return deduplicated clean ImageRecords (split unset)."""
    out_dir = Path(out_dir)
    seed = int(cfg.get("seed", 0))
    unseen = set(cfg["data"]["generators"]["unseen"])
    min_size = int(cfg["data"]["normalize"]["min_source_size"])

    valid: List[RawItem] = []
    for it in items:
        ok, why = validate_image(it.path, min_size)
        if ok:
            valid.append(it)
        else:
            log.warning("skip %s: %s", it.path, why)

    # raw-level dedup (before any processing / split)
    raw_sha = {it.path: sha256_file(it.path) for it in valid}
    seen_raw: dict = {}
    uniq: List[RawItem] = []
    for it in sorted(valid, key=lambda i: str(i.path)):
        if raw_sha[it.path] in seen_raw:
            log.info("raw duplicate dropped: %s", it.path)
            continue
        seen_raw[raw_sha[it.path]] = it.path
        uniq.append(it)

    pool = real_quality_pool([it.path for it in uniq if it.label == LABEL_REAL])  # data-prep only
    records: List[ImageRecord] = []
    for it in uniq:
        gid = f"{it.generator}:{raw_sha[it.path][:16]}"
        rng = np.random.default_rng([seed, zlib.crc32(gid.encode())])
        dst = out_dir / "images" / f"{it.generator}_{raw_sha[it.path][:16]}.png"
        prov = normalize_image(it.path, dst, is_real=(it.label == LABEL_REAL), cfg=cfg, quality_pool=pool, rng=rng)
        prov["raw_sha256"] = raw_sha[it.path]
        prov["jpeg_harmonize"] = bool(cfg["data"]["normalize"]["jpeg_harmonize"]) and it.label == LABEL_FAKE
        records.append(ImageRecord(
            sha256=sha256_file(dst), path=str(dst.relative_to(out_dir)).replace("\\", "/"), label=it.label,
            generator=it.generator, source_group_id=gid, is_unseen_generator=it.generator in unseen,
            width=int(cfg["data"]["normalize"]["crop_size"]), height=int(cfg["data"]["normalize"]["crop_size"]),
            provenance=prov))
    kept, dropped = dedup_records(records)   # post-normalisation dedup (e.g. different raws, same crop)
    if dropped:
        log.info("post-normalisation dedup dropped %d", len(dropped))
    return kept


__all__ = ["RawItem", "prepare_records", "LABEL_FAKE", "LABEL_REAL"]
