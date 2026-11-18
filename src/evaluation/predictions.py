"""Per-sample prediction store written by `python -m src.evaluation.ablation --save-predictions`.

Layout: one compressed archive per head variant, `<out>/predictions/predictions_<head_variant>.npz`.
Samples are grouped by the evaluation slice (variant, scenario, level, generator, N, seed); '-' stands for None.
For every group the archive holds
    <group>##sha256  (n,) str      <group>##y  (n,) int
    <group>##p_<method>  (n,) float64      (every method evaluated on exactly that slice, e.g. clip, aegis, ...)
    <group>##w_<method>  (n,3) float64     (gating weights; only for aegis / aegis_noadapt)
Non-adaptive baselines of S4 live in the group with N='-' and seed='-'; aegis_noadapt in N='0'; aegis and
static_learned_support in groups with N=<n> and seed=<s>. Every group of the same (head_variant, scenario, level,
generator) shares the same eval sample order, so predictions of different groups are paired by position.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

SEP, FIELD = "::", "##"
META_KEY = "_meta"


def group_key(variant: str, scenario: str, level: Any, generator: Any, N: Any, seed: Any) -> str:
    return SEP.join("-" if v is None else str(v) for v in (variant, scenario, level, generator, N, seed))


class PredictionSink:
    """Collects (sha256, y, p, weights) per method and slice while the ablation runs."""

    def __init__(self, meta: Optional[Dict[str, str]] = None) -> None:
        self.groups: Dict[str, Dict[str, np.ndarray]] = {}
        self.meta: Dict[str, str] = dict(meta or {})          # e.g. {'fusion': 'logit'}

    def add(self, key: str, method: str, sha256, y, p, weights: Optional[np.ndarray] = None) -> None:
        g = self.groups.setdefault(key, {})
        sha = np.asarray([str(s) for s in sha256])
        if "sha256" in g and not np.array_equal(g["sha256"], sha):
            raise ValueError(f"sample order mismatch inside group {key}")
        g["sha256"], g["y"] = sha, np.asarray(y, dtype=np.int64)
        g[f"p_{method}"] = np.asarray(p, dtype=np.float64).copy()
        if weights is not None:
            g[f"w_{method}"] = np.asarray(weights, dtype=np.float64).copy()

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrs = {f"{k}{FIELD}{f}": v for k, g in self.groups.items() for f, v in g.items()}
        arrs.update({f"{META_KEY}{FIELD}{k}": np.asarray(str(v)) for k, v in self.meta.items()})
        np.savez_compressed(path, **arrs)
        return path


def load_predictions(path: str | Path) -> Dict[str, Dict[str, np.ndarray]]:
    """Inverse of PredictionSink.save -> {group_key: {field: array}}."""
    out: Dict[str, Dict[str, np.ndarray]] = {}
    with np.load(Path(path), allow_pickle=False) as z:
        for name in z.files:
            k, f = name.split(FIELD)
            if k != META_KEY:
                out.setdefault(k, {})[f] = z[name]
    return out


def parse_key(key: str) -> Dict[str, Any]:
    v, sc, lvl, gen, n, seed = key.split(SEP)
    nn = lambda s: None if s == "-" else s            # noqa: E731
    return {"variant": v, "scenario": sc, "level": nn(lvl), "generator": nn(gen),
            "N": None if n == "-" else int(n), "seed": None if seed == "-" else int(seed)}


def load_meta(path: str | Path) -> Dict[str, str]:
    """Run metadata stored next to the predictions (e.g. {'fusion': 'logit'}); {} for older files."""
    with np.load(Path(path), allow_pickle=False) as z:
        return {n.split(FIELD)[1]: str(z[n]) for n in z.files if n.split(FIELD)[0] == META_KEY}
