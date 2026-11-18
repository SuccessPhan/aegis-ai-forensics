"""Dataset shortcut audit: can 3 pixel-only quality features alone predict Real vs Fake?

A cross-validated Logistic Regression on (jpeg_quality_est, log_laplacian_var, high_freq_energy_ratio).
PASS iff |accuracy - 0.5| <= tol AND |AUROC - 0.5| <= tol (tol from config shortcut_audit.tolerance).
CLI (real data later):  python -m src.data.shortcut_audit --manifest <path> [--split test_benchmark]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.core.config import load_config
from src.core.contracts import LABEL_FAKE, LABEL_REAL, S4_SPLITS, SplitManifest
from src.core.logger import get_logger
from src.data.quality import extract_quality_features

log = get_logger(__name__)


@dataclass
class AuditResult:
    accuracy: float
    auroc: float
    n: int
    tolerance: float
    passed: bool
    seen: Optional[bool] = None          # per-generator audits: True = seen generator, False = unseen (S4 pool)


def audit_features(X: np.ndarray, y: np.ndarray, cfg: Optional[dict] = None) -> AuditResult:
    cfg = cfg or load_config()
    a = cfg["shortcut_audit"]
    y = np.asarray(y)
    folds = int(min(a["cv_folds"], np.bincount(y).min()))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(a["seed"]))
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    proba = cross_val_predict(clf, X, y, cv=skf, method="predict_proba")[:, 1]
    acc = float(accuracy_score(y, (proba >= 0.5).astype(int)))
    auc = float(roc_auc_score(y, proba))
    tol = float(a["tolerance"])
    return AuditResult(acc, auc, int(len(y)), tol, bool(abs(acc - 0.5) <= tol and abs(auc - 0.5) <= tol))


def audit_images(arrays: Sequence[np.ndarray], labels: Sequence[int], cfg: Optional[dict] = None) -> AuditResult:
    cfg = cfg or load_config()
    X = np.stack([extract_quality_features(a, cfg).to_array() for a in arrays])
    return audit_features(X, np.asarray(labels), cfg)


def _audit_records(man: SplitManifest, recs: list, cfg: dict) -> AuditResult:
    """Balanced (seeded subsample) CV audit of the given clean records."""
    rng = np.random.default_rng(int(cfg["shortcut_audit"]["seed"]))
    real = [r for r in recs if r.label == LABEL_REAL]
    fake = [r for r in recs if r.label == LABEL_FAKE]
    n = min(len(real), len(fake))
    if n < 4:
        raise ValueError(f"too few samples to audit (real={len(real)}, fake={len(fake)})")
    pick: List = [real[i] for i in rng.permutation(len(real))[:n]] + [fake[i] for i in rng.permutation(len(fake))[:n]]
    arrays = []
    for r in pick:
        with Image.open(man.abs_path(r)) as im:
            arrays.append(np.asarray(im.convert("RGB")))
    return audit_images(arrays, [r.label for r in pick], cfg)


def _clean_seen(man: SplitManifest, split: Optional[str]) -> list:
    return [r for r in man.records if r.is_clean and not r.is_unseen_generator and (split is None or r.split == split)]


def audit_manifest(man: SplitManifest, cfg: Optional[dict] = None, split: Optional[str] = None) -> AuditResult:
    """Audit clean seen-generator images (optionally one split), balancing classes by seeded subsampling."""
    cfg = cfg or load_config()
    return _audit_records(man, _clean_seen(man, split), cfg)


def audit_per_generator(man: SplitManifest, cfg: Optional[dict] = None, split: Optional[str] = None,
                        include_unseen: bool = True) -> Dict[str, AuditResult]:
    """Real vs EACH generator separately on clean images. Seen generators use the seen pool (S1, optionally one
    split); unseen generators (midjourney, glide) use the clean images of the S4 pools (Real from the S4 pools too).
    Results carry `seen` True/False. Generators with too few samples are skipped (logged)."""
    cfg = cfg or load_config()
    recs = _clean_seen(man, split)
    real = [r for r in recs if r.label == LABEL_REAL]
    out: Dict[str, AuditResult] = {}
    for gen in sorted({r.generator for r in recs if r.label == LABEL_FAKE}):
        try:
            res = _audit_records(man, real + [r for r in recs if r.generator == gen and r.label == LABEL_FAKE], cfg)
            res.seen = True
            out[gen] = res
        except ValueError as e:
            log.warning("per-generator audit skipped for %s: %s", gen, e)
    if include_unseen:
        s4 = [r for r in man.records if r.is_clean and r.split in S4_SPLITS]
        s4_real = [r for r in s4 if r.label == LABEL_REAL]
        for gen in sorted({r.generator for r in s4 if r.label == LABEL_FAKE}):
            try:
                res = _audit_records(man, s4_real + [r for r in s4 if r.generator == gen and r.label == LABEL_FAKE], cfg)
                res.seen = False
                out[gen] = res
            except ValueError as e:
                log.warning("per-generator audit skipped for unseen %s: %s", gen, e)
    return out


def audit_full(man: SplitManifest, cfg: Optional[dict] = None, split: Optional[str] = None) -> dict:
    """Pooled (seen) + per-generator (seen AND unseen) audit.
    `passed` = pooled AND every SEEN per-generator audit within tolerance (unseen generators are reported in
    `per_generator` with seen=False and listed in `failed_unseen`, but do not affect `passed`).
    `status`: "pass"/"fail" when shortcut_audit.enforce is true; "reported" when false (figures only, not a gate;
    `passed` still holds the raw verdict so the CLI --strict can force exit 1)."""
    cfg = cfg or load_config()
    pooled = audit_manifest(man, cfg, split)
    per = audit_per_generator(man, cfg, split)
    seen_res = {g: r for g, r in per.items() if r.seen}
    passed = bool(pooled.passed and all(r.passed for r in seen_res.values()))
    enforce = bool(cfg["shortcut_audit"].get("enforce", True))
    return {**asdict(pooled), "passed": passed, "pooled_passed": pooled.passed,
            "per_generator": {g: asdict(r) for g, r in per.items()},
            "failed_generators": sorted(g for g, r in seen_res.items() if not r.passed),
            "failed_unseen": sorted(g for g, r in per.items() if r.seen is False and not r.passed),
            "enforce": enforce,
            "status": ("pass" if passed else "fail") if enforce else "reported"}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="AEGIS dataset-shortcut audit (3 quality features, CV logistic regression)")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default=None, help="restrict to one split (default: all seen-generator clean images)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--strict", action="store_true", help="exit 1 when the audit fails even if enforce=false")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    man = SplitManifest.load(args.manifest)
    res = audit_full(man, cfg, args.split)
    res["mode"] = man.meta.get("normalize_mode")
    log.info("shortcut audit: %s", json.dumps(res))
    if not res["enforce"] and not args.strict:
        return 0
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
