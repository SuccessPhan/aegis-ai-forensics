"""Hand-off bundle for the gating / evaluation layer.

CLI: python -m src.models.bundle --manifest <path> --split <name> [--out <file>]

File ``<cache_dir>/bundles/<split>.pt`` is a dict (all lists/tensors aligned, n = records of the split,
including degraded variants, in manifest order):
  sha256      list[str] (n)
  label       LongTensor[n]            0 real / 1 fake
  generator   list[str] (n)
  degradation list[str | None] (n)     DegradationSpec.name, None if clean
  scenario    list[str | None] (n)     DegradationSpec.scenario ("S2"/"S3"/"val"), None if clean
  logits      FloatTensor[n, 3]        RAW (uncalibrated) logits, columns = EXPERT_NAMES (clip, dinov2, freq)
  cosdist     FloatTensor[n, 3]        1 - cos(feature, clean train_experts centroid), same column order
  quality     FloatTensor[n, 3]        columns = QUALITY_FEATURE_NAMES (from src/data/quality.py)
  meta        dict                     cache_version, model_ids {expert: id}, split, created (ISO date)
Calibration and entropy (on calibrated probabilities) are done downstream
(src.ood.profiler.predictive_entropy).
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from src.core.config import load_config, resolve_path
from src.core.contracts import EXPERT_NAMES, ImageRecord, SplitManifest
from src.core.logger import get_logger
from src.data.quality import extract_quality_features
from src.models.base import load_image_array
from src.models.cache import FeatureCache
from src.ood.profiler import cosine_distance

log = get_logger(__name__)


def build_bundle(man: SplitManifest, records: Sequence[ImageRecord], split: str, cfg: dict,
                 experts: Dict[str, object], caches: Dict[str, FeatureCache],
                 centroids: Dict[str, torch.Tensor], batch_size: int = 64) -> dict:
    """`experts` maps name -> expert with head/weights loaded; clip/dinov2 read features from `caches`."""
    n = len(records)
    logits = torch.zeros(n, 3)
    cosd = torch.zeros(n, 3)
    shas = [r.sha256 for r in records]
    for j, name in enumerate(EXPERT_NAMES):
        if name == "freq":
            continue
        feats = caches[name].features_for(shas)
        with torch.no_grad():
            logits[:, j] = experts[name].predict_logits(feats).cpu()
        cosd[:, j] = cosine_distance(feats, centroids[name])
    quality = torch.zeros(n, 3)
    fj = EXPERT_NAMES.index("freq")
    freq = experts["freq"]
    for i in range(0, n, batch_size):
        chunk = records[i:i + batch_size]
        arrs = [load_image_array(str(man.abs_path(r))) for r in chunk]
        quality[i:i + len(chunk)] = torch.from_numpy(
            np.stack([extract_quality_features(a, cfg).to_array() for a in arrs]))
        f, lg = freq.features_and_logits(freq.preprocess(arrs))
        logits[i:i + len(chunk), fj] = lg
        cosd[i:i + len(chunk), fj] = cosine_distance(f, centroids["freq"])
    return {
        "sha256": shas,
        "label": torch.tensor([r.label for r in records], dtype=torch.long),
        "generator": [r.generator for r in records],
        "degradation": [None if r.is_clean else r.degradation.name for r in records],
        "scenario": [None if r.is_clean else r.degradation.scenario for r in records],
        "logits": logits, "cosdist": cosd, "quality": quality,
        "meta": {"cache_version": cfg["experts"]["extract"]["cache_version"],
                 "model_ids": {k: getattr(e, "model_id", "") for k, e in experts.items()},
                 "split": split, "created": datetime.now().isoformat(timespec="seconds")},
    }


def main(argv: Optional[List[str]] = None) -> int:
    from src.models import build_expert
    from src.models.heads import load_head_into
    from src.ood.profiler import load_centroids
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    cfg = load_config()
    man = SplitManifest.load(a.manifest)
    records = [r for r in man.records if r.split == a.split]
    if not records:
        raise SystemExit(f"no records in split '{a.split}'")
    experts = {}
    for name in EXPERT_NAMES:
        e = build_expert(name, cfg).to(a.device)
        load_head_into(cfg, e)
        experts[name] = e.eval()
    caches = {n: FeatureCache.open(cfg, n) for n in ("clip", "dinov2")}
    bundle = build_bundle(man, records, a.split, cfg, experts, caches, load_centroids(cfg))
    out = Path(a.out) if a.out else resolve_path(cfg, "cache_dir") / "bundles" / f"{a.split}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out)
    log.info("bundle %s: %d samples -> %s", a.split, len(records), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
