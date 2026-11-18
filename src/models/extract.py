"""Batched, resumable embedding extraction for frozen experts (clip, dinov2).

CLI: python -m src.models.extract --manifest <path> --expert clip|dinov2 [--splits a b ...]
Features are cached as float16 in cfg.paths.cache_dir/features/<expert>_<cache_version>.pt, keyed by sha256.
Samples already cached are skipped (resume); the cache is flushed every `save_every` batches.
"""
from __future__ import annotations

import argparse
from typing import List, Optional, Sequence

import torch

from src.core.config import load_config
from src.core.contracts import ALL_SPLITS, ImageRecord, SplitManifest
from src.core.logger import get_logger
from src.models.base import load_image_array
from src.models.cache import FeatureCache

log = get_logger(__name__)


def extract_records(expert, man: SplitManifest, records: Sequence[ImageRecord], cache: FeatureCache,
                    batch_size: int, save_every: int = 20) -> int:
    """Extract features for records not yet in `cache`. Returns number of newly computed samples."""
    todo, seen = [], set()
    for r in records:
        if not cache.has(r.sha256) and r.sha256 not in seen:
            seen.add(r.sha256)
            todo.append(r)
    log.info("%s: %d records, %d to extract", expert.name, len(records), len(todo))
    for bi, i in enumerate(range(0, len(todo), batch_size)):
        chunk = todo[i:i + batch_size]
        x = expert.preprocess([load_image_array(str(man.abs_path(r))) for r in chunk])
        feats = expert.extract_features(x)
        for r, f in zip(chunk, feats):
            cache.put(r.sha256, f)
        if (bi + 1) % save_every == 0:
            cache.save()
            log.info("%s: %d/%d extracted", expert.name, min(i + batch_size, len(todo)), len(todo))
    if todo:
        cache.meta["model_id"] = getattr(expert, "model_id", "")
        cache.save()
    return len(todo)


def main(argv: Optional[List[str]] = None) -> int:
    from src.models import build_expert
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--expert", required=True, choices=["clip", "dinov2"])
    ap.add_argument("--splits", nargs="*", default=None, help=f"default: all of {ALL_SPLITS}")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    cfg = load_config()
    man = SplitManifest.load(a.manifest)
    splits = set(a.splits) if a.splits else set(ALL_SPLITS)
    records = [r for r in man.records if r.split in splits]
    expert = build_expert(a.expert, cfg).to(a.device)
    cache = FeatureCache.open(cfg, a.expert)
    extract_records(expert, man, records, cache, int(cfg["experts"]["extract"]["batch_size"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
