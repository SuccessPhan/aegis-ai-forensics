"""Export of the fitted AEGIS system (calibration + feature scaler + gating MLP + OOD threshold) for serving.

`fit_system` reuses the ablation's `_System` (variant full: calibrated + OOD features), fitted ONLY on val_gating_fit,
so the exported system is bit-identical to the one evaluated in the ablation (same seed). The file is a plain
torch.save dict (tensors, numbers, strings, lists/dicts) and loads with weights_only=True.

CLI: python -m src.learning.export --head-variant aug --out data/cache/serving/aegis_v3_aug.pt
(reports AUROC on val_gating_verify only; test_benchmark is never read here).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch

from src.core.config import Config, REPO_ROOT, load_config
from src.core.contracts import EXPERT_NAMES, QUALITY_FEATURE_NAMES, SPLIT_VAL_FIT, SPLIT_VAL_VERIFY
from src.core.logger import get_logger
from src.learning.bundle_io import Bundle, load_split
from src.learning.features import FeaturePipeline
from src.learning.flags import compute_flags
from src.learning.fusion import get_fusion
from src.learning.gating import GatingController

log = get_logger(__name__)

FORMAT_VERSION = 1
DEFAULT_MODEL_VERSION = "aegis-v3-aug-srm"
_HASH_SECTIONS = ("seed", "gating", "gating_train", "calibration", "flags")


def config_hash(cfg: Config) -> str:
    """sha256 (first 16 hex) of the config sections that determine the fitted system."""
    sub = {k: cfg.get(k) for k in _HASH_SECTIONS}
    return hashlib.sha256(json.dumps(sub, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def _arr3(x, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.shape[-1] != 3:
        raise ValueError(f"{name} must have last dimension 3, got shape {a.shape}")
    return a


@dataclass
class AegisSystem:
    cfg: Config
    pipe: FeaturePipeline
    gating: GatingController
    ood_threshold: float
    fusion: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ prediction
    def predict_batch(self, logits, cosdist, quality) -> Dict[str, Any]:
        """Inputs (n,3) each, columns in EXPERT_NAMES / QUALITY_FEATURE_NAMES order. Returns arrays/lists."""
        lg, cd, ql = (np.atleast_2d(_arr3(a, n)) for a, n in ((logits, "logits"), (cosdist, "cosdist"),
                                                              (quality, "quality")))
        n = len(lg)
        if not (len(cd) == len(ql) == n):
            raise ValueError("logits, cosdist, quality must have the same number of rows")
        b = Bundle([str(i) for i in range(n)], np.zeros(n, dtype=np.int64), [""] * n, [None] * n, [None] * n,
                   lg, cd, ql)
        fs = self.pipe.transform(b)
        w = self.gating.weights(fs)
        p_final = self.gating.predict(fs)
        flag, reasons = compute_flags(fs.p, p_final, fs.mean_ood, self.ood_threshold, self.cfg)
        return {"p_ai": p_final, "weights": w, "p_cal": fs.p, "ood_score": fs.mean_ood, "warning_flag": flag,
                "flag_reasons": reasons, "ood_threshold": self.ood_threshold}

    def predict_one(self, logits, cosdist, quality) -> Dict[str, Any]:
        r = self.predict_batch(np.asarray(logits).reshape(1, 3), np.asarray(cosdist).reshape(1, 3),
                               np.asarray(quality).reshape(1, 3))
        return _row(r, 0)

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        cal = self.pipe.calibrator
        torch.save({"format_version": FORMAT_VERSION, "cfg": self.cfg, "fusion": self.fusion,
                    "ood_threshold": float(self.ood_threshold),
                    "temperatures": [float(t) for t in cal.T],
                    "scaler_mean": [float(x) for x in self.pipe.mean], "scaler_std": [float(x) for x in self.pipe.std],
                    "gating_state": {k: v.detach().cpu() for k, v in self.gating.net.state_dict().items()},
                    "gating_seed": int(self.gating.seed), "metadata": self.metadata}, p)
        log.info("AEGIS system saved to %s", p)

    @classmethod
    def load(cls, path: str | Path) -> "AegisSystem":
        d = torch.load(Path(path), map_location="cpu", weights_only=True)
        if d.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"unsupported export format {d.get('format_version')!r} (expected {FORMAT_VERSION})")
        cfg = d["cfg"]
        pipe = FeaturePipeline(cfg, calibrate=True, use_ood=True)
        pipe.calibrator.T = np.asarray(d["temperatures"], dtype=float)
        pipe.mean = np.asarray(d["scaler_mean"], dtype=float)
        pipe.std = np.asarray(d["scaler_std"], dtype=float)
        gating = GatingController(cfg, seed=d["gating_seed"], fusion=d["fusion"])
        gating.net.load_state_dict(d["gating_state"])
        gating.net.eval()
        sysm = cls(cfg, pipe, gating, float(d["ood_threshold"]), d["fusion"], dict(d["metadata"]))
        sysm.metadata["loaded_config_hash_ok"] = sysm.metadata.get("config_hash") == config_hash(cfg)
        return sysm


def _row(r: Dict[str, Any], i: int) -> Dict[str, Any]:
    p = float(r["p_ai"][i])
    return {"p_ai": p, "label": "AI" if p >= 0.5 else "Real", "confidence": max(p, 1.0 - p),
            "warning_flag": bool(r["warning_flag"][i]), "flag_reasons": list(r["flag_reasons"][i]),
            "weights": {e: float(r["weights"][i, k]) for k, e in enumerate(EXPERT_NAMES)},
            "p_cal": {e: float(r["p_cal"][i, k]) for k, e in enumerate(EXPERT_NAMES)},
            "ood_score": float(r["ood_score"][i]), "ood_threshold": float(r["ood_threshold"])}


def fit_system(cfg: Config, head_variant: Optional[str] = None, fit: Optional[Bundle] = None,
               model_version: str = DEFAULT_MODEL_VERSION) -> AegisSystem:
    """Fit exactly like ablation `_System(variant=full)` on val_gating_fit (loaded from the cache unless `fit` given)."""
    from src.evaluation.ablation import _System      # local import: avoids a learning <-> evaluation import cycle
    if fit is None:
        fit = load_split(cfg, SPLIT_VAL_FIT, head_variant)
    s = _System(cfg, fit, calibrate=True, use_ood=True, seed=cfg.get("seed", 0))
    meta = {"model_version": model_version, "head_variant": head_variant,
            "cache_version": ((cfg.get("experts") or {}).get("extract") or {}).get("cache_version"),
            "fit_date": _dt.date.today().isoformat(), "n_fit": int(len(fit)), "fit_split": SPLIT_VAL_FIT,
            "fusion": s.fusion, "config_hash": config_hash(cfg),
            "temperatures": {e: float(t) for e, t in zip(EXPERT_NAMES, s.pipe.calibrator.T)},
            "expert_order": list(EXPERT_NAMES), "quality_order": list(QUALITY_FEATURE_NAMES)}
    return AegisSystem(cfg, s.pipe, s.gating, float(s.thr), s.fusion, meta)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    from src.evaluation.metrics import compute_metrics
    ap = argparse.ArgumentParser(prog="python -m src.learning.export", description="Fit + export the AEGIS system")
    ap.add_argument("--head-variant", default="aug")
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "cache" / "serving" / "aegis_v3_aug.pt"))
    ap.add_argument("--config", default=None)
    ap.add_argument("--model-version", default=DEFAULT_MODEL_VERSION)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    system = fit_system(cfg, a.head_variant, model_version=a.model_version)
    system.save(a.out)
    for k, v in system.metadata.items():
        log.info("metadata %s = %s", k, v)
    log.info("ood_threshold = %.6f", system.ood_threshold)
    ver = load_split(cfg, SPLIT_VAL_VERIFY, a.head_variant)         # verification split only (never test_benchmark)
    r = AegisSystem.load(a.out).predict_batch(ver.logits, ver.cosdist, ver.quality)
    rep = compute_metrics(ver.label, r["p_ai"], cfg, SPLIT_VAL_VERIFY, "aegis", r["warning_flag"], {})
    log.info("val_gating_verify (n=%d): AUROC=%.4f accuracy=%.4f ece=%.4f flagged_rate=%.3f", len(ver), rep.auroc,
             rep.accuracy, rep.ece, rep.flagged_rate)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
