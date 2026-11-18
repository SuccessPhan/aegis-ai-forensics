"""Loader / validator / writer for the expert-detector hand-off bundles (data/cache/bundles/<split>.pt).

Bundle dict: sha256 list[str]; label LongTensor[n]; generator list[str]; degradation list[str|None];
scenario list[str|None]; logits FloatTensor[n,3] (raw, uncalibrated, EXPERT_NAMES order); cosdist FloatTensor[n,3];
quality FloatTensor[n,3] (QUALITY_FEATURE_NAMES order); meta dict.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from src.core.config import Config, resolve_path
from src.core.contracts import ALL_SPLITS
from src.core.logger import get_logger

log = get_logger(__name__)

_LIST_KEYS = ("sha256", "generator", "degradation", "scenario")
_TENSOR_KEYS = ("label", "logits", "cosdist", "quality")


class BundleError(ValueError):
    """Raised when a bundle file does not follow the hand-off format."""


@dataclass
class Bundle:
    sha256: List[str]
    label: np.ndarray                       # int64 [n]
    generator: List[str]
    degradation: List[Optional[str]]
    scenario: List[Optional[str]]
    logits: np.ndarray                      # float64 [n,3]
    cosdist: np.ndarray                     # float64 [n,3]
    quality: np.ndarray                     # float64 [n,3]
    meta: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.sha256)

    @property
    def mean_ood(self) -> np.ndarray:
        """Mean OOD score = mean of the 3 expert cosine distances."""
        return self.cosdist.mean(axis=1)

    def subset(self, sel: Sequence[int] | np.ndarray) -> "Bundle":
        sel = np.asarray(sel)
        idx = np.flatnonzero(sel) if sel.dtype == bool else sel.astype(int)
        return Bundle([self.sha256[i] for i in idx], self.label[idx], [self.generator[i] for i in idx],
                      [self.degradation[i] for i in idx], [self.scenario[i] for i in idx],
                      self.logits[idx], self.cosdist[idx], self.quality[idx], dict(self.meta))

    def concat(self, other: "Bundle") -> "Bundle":
        return Bundle(self.sha256 + other.sha256, np.concatenate([self.label, other.label]),
                      self.generator + other.generator, self.degradation + other.degradation,
                      self.scenario + other.scenario, np.concatenate([self.logits, other.logits]),
                      np.concatenate([self.cosdist, other.cosdist]), np.concatenate([self.quality, other.quality]),
                      dict(self.meta))

    def clean_mask(self) -> np.ndarray:
        return np.array([d is None for d in self.degradation], dtype=bool)


def validate_bundle(d: Dict[str, Any], name: str = "bundle") -> None:
    """Raise BundleError listing every format problem found."""
    problems: List[str] = []
    for k in _LIST_KEYS + _TENSOR_KEYS:
        if k not in d:
            problems.append(f"missing key '{k}'")
    if problems:
        raise BundleError(f"{name}: " + "; ".join(problems))
    n = len(d["sha256"])
    for k in _LIST_KEYS:
        if len(d[k]) != n:
            problems.append(f"len({k})={len(d[k])} != n={n}")
    if len(set(d["sha256"])) != n:
        problems.append("duplicate sha256 values")
    if tuple(d["label"].shape) != (n,):
        problems.append(f"label shape {tuple(d['label'].shape)} != ({n},)")
    else:
        bad = set(np.unique(np.asarray(d["label"])).tolist()) - {0, 1}
        if bad:
            problems.append(f"labels outside {{0,1}}: {sorted(bad)}")
    for k in ("logits", "cosdist", "quality"):
        t = d[k]
        if tuple(t.shape) != (n, 3):
            problems.append(f"{k} shape {tuple(t.shape)} != ({n}, 3)")
        elif not bool(torch.isfinite(t.float()).all()):
            problems.append(f"{k} contains NaN/Inf")
    if problems:
        raise BundleError(f"{name}: " + "; ".join(problems))


def _to_np(t: Any, dtype) -> np.ndarray:
    return t.detach().cpu().numpy().astype(dtype) if isinstance(t, torch.Tensor) else np.asarray(t, dtype=dtype)


def bundle_from_dict(d: Dict[str, Any], name: str = "bundle") -> Bundle:
    d = dict(d)
    for k in _TENSOR_KEYS:                   # tolerate numpy / list inputs
        if k in d and not isinstance(d[k], torch.Tensor):
            d[k] = torch.as_tensor(np.asarray(d[k]))
    validate_bundle(d, name)
    return Bundle(list(d["sha256"]), _to_np(d["label"], np.int64), list(d["generator"]),
                  list(d["degradation"]), list(d["scenario"]), _to_np(d["logits"], np.float64),
                  _to_np(d["cosdist"], np.float64), _to_np(d["quality"], np.float64), dict(d.get("meta") or {}))


def load_bundle(path: str | Path) -> Bundle:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"bundle not found: {p}")
    try:
        d = torch.load(p, map_location="cpu", weights_only=True)
    except Exception:                        # meta may hold non-tensor objects
        d = torch.load(p, map_location="cpu", weights_only=False)
    b = bundle_from_dict(d, p.name)
    log.info("loaded bundle %s (n=%d)", p.name, len(b))
    return b


def bundle_path(cfg: Config, split: str, head_variant: Optional[str] = None) -> Path:
    """data/cache/bundles/<head_variant>/<split>.pt; legacy data/cache/bundles/<split>.pt when no variant given."""
    if split not in ALL_SPLITS:
        raise ValueError(f"unknown split {split!r}")
    base = resolve_path(cfg, "cache_dir") / "bundles"
    return (base / head_variant if head_variant else base) / f"{split}.pt"


def load_split(cfg: Config, split: str, head_variant: Optional[str] = None) -> Bundle:
    return load_bundle(bundle_path(cfg, split, head_variant))


def save_bundle(b: Bundle, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"sha256": b.sha256, "label": torch.as_tensor(b.label, dtype=torch.long), "generator": b.generator,
                "degradation": b.degradation, "scenario": b.scenario,
                "logits": torch.as_tensor(b.logits, dtype=torch.float32),
                "cosdist": torch.as_tensor(b.cosdist, dtype=torch.float32),
                "quality": torch.as_tensor(b.quality, dtype=torch.float32), "meta": b.meta}, path)
