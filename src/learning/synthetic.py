"""Small structured synthetic bundles for tests / dry runs (no real data needed).

Structure: the freq expert is accurate on good-quality images but outputs confident inverted (anti-correlated) logits when
jpeg_quality_est is low (bad quality); CLIP/DINOv2 are moderately accurate everywhere. A gating that routes on
quality should therefore beat the naive average. Unseen-generator samples (S4) get larger cosdist.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from src.learning.bundle_io import Bundle


def make_synthetic_bundle(n: int = 400, seed: int = 0, bad_frac: float = 0.3, unseen: bool = False,
                          scenarios: bool = False, clean_only: bool = False, tag: str = "x") -> Bundle:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    bad = rng.random(n) < (0.0 if clean_only else bad_frac)
    q = np.where(bad, rng.uniform(0.0, 0.3, n), rng.uniform(0.7, 1.0, n))
    sign = 2.0 * y - 1.0
    clip = sign * 1.2 + rng.normal(0, 1.3, n)
    dino = sign * 1.2 + rng.normal(0, 1.3, n)
    freq = np.where(bad, -sign * 4.0 + rng.normal(0, 1.0, n), sign * 4.0 + rng.normal(0, 1.0, n))
    quality = np.stack([q, rng.normal(0, 1, n), np.where(bad, 0.1, 0.6) + rng.normal(0, 0.02, n)], 1)
    cos = rng.uniform(0.1, 0.3, (n, 3)) + (0.3 if unseen else 0.0)
    deg = [None if not b else "jpeg_q40" for b in bad]
    scen = [None if d is None else "S2" for d in deg] if scenarios else [None] * n
    gen = [("midjourney" if unseen else "adm") if yy else "real" for yy in y]
    return Bundle([f"{tag}{seed}_{i:05d}" for i in range(n)], y.astype(np.int64), gen, deg, scen,
                  np.stack([clip, dino, freq], 1), cos, quality, {"synthetic": True})
