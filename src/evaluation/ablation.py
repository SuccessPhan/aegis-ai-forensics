"""Final ablation table: 6 methods x S1-S4 (+ calibration / OOD-profiler variants) -> CSV + Markdown.

Everything (temperature, scaler, gating, static LR, OOD threshold) is fitted on val_gating_fit BEFORE the test bundle
is touched. `test` (test_benchmark) and the S4 pools are only read inside `run_ablation`, once.

Scenario slicing of the test bundle (via bundle.degradation / bundle.scenario):
  S1 = clean samples; S2 = scenario 'S2' per degradation level (+ 'all'); S3 = scenario 'S3' per level (+ 'all');
  S4 = s4_eval, per N in config fewshot.n_values (support drawn from s4_support, averaged over `s4_seeds`).
Variants: full (calibrated + OOD features), nocal (T=1), noood (gating sees only p_cal), nocal_noood.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.core.config import Config
from src.core.contracts import EXPERT_NAMES
from src.core.logger import get_logger
from src.evaluation.metrics import compute_metrics
from src.learning.baselines import NaiveAverage, SingleExpert, StaticLearnedWeights
from src.learning.bundle_io import Bundle
from src.learning.features import FeaturePipeline
from src.learning.fewshot import adapt, select_support
from src.learning.flags import compute_flags, fit_ood_threshold
from src.learning.gating import GatingController

log = get_logger(__name__)

METHODS = ("clip", "dinov2", "freq", "naive_avg", "static_lr", "aegis")
_NUM = ("accuracy", "precision", "recall", "f1", "auroc", "ece", "selective_accuracy", "flagged_rate")
COLUMNS = ["scenario", "level", "N", "method", "variant", "n", *_NUM]


class _System:
    """One (calibrate, use_ood) configuration fitted on val_gating_fit."""

    def __init__(self, cfg: Config, fit: Bundle, calibrate: bool, use_ood: bool, seed: int):
        self.cfg = cfg
        self.pipe = FeaturePipeline(cfg, calibrate, use_ood).fit(fit)
        self.fit_fs = self.pipe.transform(fit)
        self.fit_y = fit.label
        self.thr = fit_ood_threshold(fit.mean_ood, cfg)
        self.models: Dict[str, Any] = {e: SingleExpert(e) for e in EXPERT_NAMES}
        self.models["naive_avg"] = NaiveAverage()
        self.models["static_lr"] = StaticLearnedWeights().fit(self.fit_fs, fit.label)
        self.gating = GatingController(cfg, seed=seed).fit(self.fit_fs, fit.label)
        self.models["aegis"] = self.gating
        self.seed = seed


def _row(sysm: _System, method: str, model: Any, b: Bundle, fs, scenario: str, level: str, N: Optional[int],
         variant: str):
    p = model.predict(fs)
    flags, _ = compute_flags(fs.p, p, fs.mean_ood, sysm.thr, sysm.cfg)
    return compute_metrics(b.label, p, sysm.cfg, scenario, method, flags,
                           {"level": level, "N": N, "variant": variant})


def _mean_reports(reps):
    r = reps[0]
    if len(reps) == 1:
        return r
    for k in _NUM:
        vals = [getattr(x, k) for x in reps if getattr(x, k) is not None]
        setattr(r, k, float(np.nanmean(vals)) if vals else None)
    r.extras = dict(r.extras, n_seeds=len(reps))
    return r


def _slices(test: Bundle):
    yield "S1", "clean", test.subset(test.clean_mask())
    for sc in ("S2", "S3"):
        m = np.array([s == sc for s in test.scenario])
        if not m.any():
            log.warning("test bundle has no %s samples", sc)
            continue
        sub = test.subset(m)
        for lvl in sorted({d for d in sub.degradation if d}):
            yield sc, lvl, sub.subset(np.array([d == lvl for d in sub.degradation]))
        yield sc, "all", sub


def run_ablation(cfg: Config, fit: Bundle, test: Bundle, s4_support: Optional[Bundle] = None,
                 s4_eval: Optional[Bundle] = None, out_dir: Optional[str | Path] = None,
                 s4_seeds: Optional[Sequence[int]] = None, n_values: Optional[Sequence[int]] = None
                 ) -> List[Dict[str, Any]]:
    """`fit` = val_gating_fit bundle, `test` = test_benchmark bundle (read once, here)."""
    seed = cfg.get("seed", 0)
    s4_seeds = list(s4_seeds) if s4_seeds else [seed]
    n_values = list(n_values) if n_values else cfg["fewshot"]["n_values"]
    variants = {"full": (True, True), "nocal": (False, True), "noood": (True, False), "nocal_noood": (False, False)}
    reports = []
    for vname, (cal, ood) in variants.items():
        sysm = _System(cfg, fit, cal, ood, seed)
        methods = METHODS if vname in ("full", "nocal") else ("aegis",)
        for sc, lvl, b in _slices(test):
            fs = sysm.pipe.transform(b)
            for m in methods:
                reports.append(_row(sysm, m, sysm.models[m], b, fs, sc, lvl, None, vname))
        if s4_support is None or s4_eval is None:
            continue
        fs_eval = sysm.pipe.transform(s4_eval)
        for m in methods:
            if m != "aegis":
                reports.append(_row(sysm, m, sysm.models[m], s4_eval, fs_eval, "S4", "all", None, vname))
        for n in n_values:
            per_seed = []
            for s in s4_seeds:
                sup = select_support(s4_support, n, s)
                res = adapt(sysm.gating, sysm.fit_fs, sysm.fit_y, sysm.pipe.transform(sup), sup.label, cfg, seed=s)
                per_seed.append(_row(sysm, "aegis", res.model, s4_eval, fs_eval, "S4", "all", n, vname))
                per_seed[-1].extras["regime"] = res.regime
            reports.append(_mean_reports(per_seed))
        # reference: AEGIS without adaptation (N-independent)
        reports.append(_row(sysm, "aegis_noadapt", sysm.gating, s4_eval, fs_eval, "S4", "all", 0, vname))
    rows = [_flatten(r) for r in reports]
    if out_dir is not None:
        write_tables(rows, out_dir)
    return rows


def _flatten(r) -> Dict[str, Any]:
    d = {"scenario": r.scenario, "level": r.extras.get("level"), "N": r.extras.get("N"), "method": r.method,
         "variant": r.extras.get("variant"), "n": r.n}
    d.update({k: getattr(r, k) for k in _NUM})
    if "regime" in r.extras:
        d["regime"] = r.extras["regime"]
    return d


def _fmt(v: Any) -> str:
    return "" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


def write_tables(rows: List[Dict[str, Any]], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "ablation.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS + ["regime"], extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    lines = ["# AEGIS ablation (cells: accuracy / AUROC; variant = full)", ""]
    main = [r for r in rows if r["variant"] == "full"]
    meths = [m for m in (*METHODS, "aegis_noadapt") if any(r["method"] == m for r in main)]
    lines += ["| scenario | level | N | " + " | ".join(meths) + " |", "|" + "---|" * (3 + len(meths))]
    keys = list(dict.fromkeys((r["scenario"], r["level"], r["N"]) for r in main))
    for sc, lvl, n in keys:
        cells = []
        for m in meths:
            hit = [r for r in main if (r["scenario"], r["level"], r["N"], r["method"]) == (sc, lvl, n, m)]
            if not hit and m == "aegis_noadapt":
                hit = []
            cells.append(f"{hit[0]['accuracy']:.3f} / {hit[0]['auroc']:.3f}" if hit else "-")
        lines.append(f"| {sc} | {lvl} | {_fmt(n)} | " + " | ".join(cells) + " |")
    lines += ["", "## All metrics (long format)", "", "| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    lines += ["| " + " | ".join(_fmt(r.get(c)) for c in COLUMNS) + " |" for r in rows]
    (out / "ablation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("ablation tables written to %s (%d rows)", out, len(rows))
