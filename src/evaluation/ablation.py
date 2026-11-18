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
S4_ONLY_METHODS = ("static_learned_support",)    # S4 only, variants full/nocal, per N and seed
_NUM = ("accuracy", "precision", "recall", "f1", "auroc", "ece", "selective_accuracy", "flagged_rate")
COLUMNS = ["head_variant", "scenario", "level", "generator", "N", "method", "variant", "n", *_NUM]


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


def fit_static_with_support(fit_fs, fit_y, sup_fs, sup_y, cfg: Config) -> StaticLearnedWeights:
    """Static LR stacking refit on val_gating_fit + support (support repeated `fewshot.support_repeat` times,
    exactly like the AEGIS full refit). Support labels are ground truth from the manifest."""
    rep = int(cfg["fewshot"].get("support_repeat", 1))
    X = fit_fs
    for _ in range(rep):
        X = X.concat(sup_fs)
    y_all = np.concatenate([np.asarray(fit_y)] + [np.asarray(sup_y)] * rep)
    return StaticLearnedWeights().fit(X, y_all)


def _mean_reports(reps):
    r = reps[0]
    if len(reps) > 1:
        r.extras = dict(r.extras, support_sha256={k: v for x in reps for k, v in x.extras["support_sha256"].items()})
    if len(reps) == 1:
        return r
    for k in _NUM:
        raw = [getattr(x, k) for x in reps if getattr(x, k) is not None]
        arr = np.array(raw, dtype=float)
        arr = arr[~np.isnan(arr)]                          # all-NaN -> NaN without a numpy 'empty slice' warning
        setattr(r, k, None if not raw else (float(arr.mean()) if arr.size else float("nan")))
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
                 s4_seeds: Optional[Sequence[int]] = None, n_values: Optional[Sequence[int]] = None,
                 head_variant: Optional[str] = None, s4_generators: Optional[Sequence[str]] = None
                 ) -> List[Dict[str, Any]]:
    """`fit` = val_gating_fit bundle, `test` = test_benchmark bundle (read once, here).
    S4 is reported per unseen generator (support = N/2 AI of that generator + N/2 Real; eval = that generator + Real)
    plus a pooled row group (generator='all'). `s4_generators` defaults to the AI generators found in s4_eval."""
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
        for gname, gen in _s4_groups(s4_eval, s4_generators):
            ev = s4_eval if gen is None else s4_eval.subset(
                np.array([lb == 0 or g == gen for lb, g in zip(s4_eval.label, s4_eval.generator)]))
            reports += _s4_reports(cfg, sysm, vname, methods, s4_support, ev, gname, gen, n_values, s4_seeds)
    rows = [_flatten(r) for r in reports]
    for r in rows:
        r["head_variant"] = head_variant
    if out_dir is not None:
        write_tables(rows, out_dir)
    return rows


def _s4_groups(s4_eval: Bundle, generators: Optional[Sequence[str]]):
    gens = list(generators) if generators else sorted({g for g, lb in zip(s4_eval.generator, s4_eval.label) if lb == 1})
    return [(g, g) for g in gens] + [("all", None)]


def _s4_reports(cfg, sysm, vname, methods, s4_support, ev, gname, gen, n_values, s4_seeds):
    reports = []
    fs_eval = sysm.pipe.transform(ev)
    for m in methods:
        if m != "aegis":
            reports.append(_row(sysm, m, sysm.models[m], ev, fs_eval, "S4", "all", None, vname))
    for n in n_values:
        per_seed, per_seed_sl = [], []
        for s in s4_seeds:
            sup = select_support(s4_support, n, s, generator=gen)
            sup_fs = sysm.pipe.transform(sup)
            shas = [str(x) for x in sup.sha256]
            res = adapt(sysm.gating, sysm.fit_fs, sysm.fit_y, sup_fs, sup.label, cfg, seed=s)
            per_seed.append(_row(sysm, "aegis", res.model, ev, fs_eval, "S4", "all", n, vname))
            per_seed[-1].extras.update(regime=res.regime, support_sha256={s: shas})
            if vname in ("full", "nocal"):
                # fair baseline: same support set (same seed/sample), same support_repeat as AEGIS refit
                sl = fit_static_with_support(sysm.fit_fs, sysm.fit_y, sup_fs, sup.label, cfg)
                per_seed_sl.append(_row(sysm, "static_learned_support", sl, ev, fs_eval, "S4", "all", n, vname))
                per_seed_sl[-1].extras["support_sha256"] = {s: shas}
        reports.append(_mean_reports(per_seed))
        if per_seed_sl:
            reports.append(_mean_reports(per_seed_sl))
    reports.append(_row(sysm, "aegis_noadapt", sysm.gating, ev, fs_eval, "S4", "all", 0, vname))
    for r in reports:
        r.extras["generator"] = gname
    return reports


def _flatten(r) -> Dict[str, Any]:
    d = {"scenario": r.scenario, "level": r.extras.get("level"), "N": r.extras.get("N"), "method": r.method,
         "variant": r.extras.get("variant"), "generator": r.extras.get("generator"), "n": r.n}
    d.update({k: getattr(r, k) for k in _NUM})
    if "support_sha256" in r.extras:
        d["support_sha256"] = r.extras["support_sha256"]      # {seed: [sha256,...]}; not a CSV column
    if "regime" in r.extras:
        d["regime"] = r.extras["regime"]
    return d


def _fmt(v: Any) -> str:
    return "" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


BASELINES = ("clip", "dinov2", "freq", "naive_avg", "static_lr", "static_learned_support")


def _isnum(v) -> bool:
    return v is not None and not (isinstance(v, float) and np.isnan(v))


def _score(r: Dict[str, Any]) -> float:
    """Ranking score for 'strongest baseline': AUROC, falling back to accuracy when AUROC is undefined."""
    return r["auroc"] if _isnum(r["auroc"]) else (r["accuracy"] if _isnum(r["accuracy"]) else -1.0)


def _key(r: Dict[str, Any]):
    return (r["scenario"], r["level"], r.get("generator"), r["N"])


def summarize_best_baseline(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per (scenario, level, generator, N, head_variant): AEGIS (variant=full) vs the strongest baseline over ALL head
    variants (so AEGIS on the 'clean' head is also compared with e.g. dinov2 on the 'aug' head). Reporting only."""
    full = [r for r in rows if r["variant"] == "full"]
    out = []
    for r in full:
        if r["method"] != "aegis":
            continue
        sc, lvl, gen, n = _key(r)
        cands = [c for c in full if c["method"] in BASELINES and _key(c)[:3] == (sc, lvl, gen) and c["N"] in (None, n)]
        if not cands:
            continue
        best = max(cands, key=_score)
        hv = best.get("head_variant")
        d_auc = r["auroc"] - best["auroc"] if _isnum(r["auroc"]) and _isnum(best["auroc"]) else None
        d_acc = r["accuracy"] - best["accuracy"] if _isnum(r["accuracy"]) and _isnum(best["accuracy"]) else None
        out.append({"scenario": sc, "level": lvl, "generator": gen, "N": n, "head_variant": r.get("head_variant"),
                    "aegis_accuracy": r["accuracy"], "aegis_auroc": r["auroc"],
                    "best_baseline": f"{best['method']}_{hv}" if hv else best["method"],
                    "best_accuracy": best["accuracy"], "best_auroc": best["auroc"],
                    "delta_auroc": d_auc, "delta_accuracy": d_acc})
    return out


SUMMARY_COLUMNS = ["scenario", "level", "generator", "N", "head_variant", "aegis_accuracy", "aegis_auroc",
                   "best_baseline", "best_accuracy", "best_auroc", "delta_auroc", "delta_accuracy"]


def _main_table(rows: List[Dict[str, Any]], title: str) -> List[str]:
    lines = [title, ""]
    main = [r for r in rows if r["variant"] == "full"]
    meths = [m for m in (*METHODS, *S4_ONLY_METHODS, "aegis_noadapt") if any(r["method"] == m for r in main)]
    lines += ["| scenario | level | generator | N | " + " | ".join(meths) + " |", "|" + "---|" * (4 + len(meths))]
    idx = {(*_key(r), r["method"]): r for r in main}
    for key in dict.fromkeys(_key(r) for r in main):
        cells = [f"{idx[(*key, m)]['accuracy']:.3f} / {idx[(*key, m)]['auroc']:.3f}" if (*key, m) in idx else "-"
                 for m in meths]
        lines.append(f"| {key[0]} | {key[1]} | {_fmt(key[2])} | {_fmt(key[3])} | " + " | ".join(cells) + " |")
    return lines


def _long_table(rows: List[Dict[str, Any]]) -> List[str]:
    cols = [c for c in COLUMNS if c != "head_variant"]
    return ["## All metrics (long format)", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols),
            *("| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |" for r in rows)]


def _summary_md(summ: List[Dict[str, Any]]) -> List[str]:
    cols = SUMMARY_COLUMNS
    return ["# AEGIS (full) vs strongest baseline across all head variants", "",
            "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols),
            *("| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |" for r in summ)]


def write_tables(rows: List[Dict[str, Any]], out_dir: str | Path) -> None:
    """ablation.csv (all rows, with head_variant); ablation_<variant>.md per head variant; ablation_summary.md/.csv
    (AEGIS vs strongest baseline over all variants); ablation.md = everything concatenated."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "ablation.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS + ["regime"], extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    hvs = list(dict.fromkeys(r.get("head_variant") for r in rows))
    parts: List[str] = []
    for hv in hvs:
        sub = [r for r in rows if r.get("head_variant") == hv]
        lines = _main_table(sub, f"# AEGIS ablation, head variant = {hv or 'default'} "
                                 "(cells: accuracy / AUROC; variant = full)") + [""] + _long_table(sub)
        if hv:
            (out / f"ablation_{hv}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        parts += lines + [""]
    summ = summarize_best_baseline(rows)
    sl = _summary_md(summ)
    (out / "ablation_summary.md").write_text("\n".join(sl) + "\n", encoding="utf-8")
    with open(out / "ablation_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        w.writeheader()
        w.writerows(summ)
    (out / "ablation.md").write_text("\n".join(sl + [""] + parts) + "\n", encoding="utf-8")
    log.info("ablation tables written to %s (%d rows)", out, len(rows))


# ------------------------------------------------------------------------------------------------------- CLI
REQUIRED_SPLITS = ("val_gating_fit", "test_benchmark", "s4_support", "s4_eval")
DEFAULT_VARIANTS = ("clean", "aug")


class MissingBundlesError(FileNotFoundError):
    """Raised with the list of every missing bundle file."""

    def __init__(self, missing: Sequence[Path]):
        self.missing = [str(m) for m in missing]
        super().__init__("missing bundle files:\n  " + "\n  ".join(self.missing))


def run_all(cfg: Config, head_variants: Optional[Sequence[str]] = None, out_dir: str | Path = "reports",
            s4_seeds: Optional[Sequence[int]] = None) -> List[Dict[str, Any]]:
    """Load bundles, fit on val_gating_fit, evaluate test_benchmark once per head variant, write tables.
    head_variants None -> every variant in DEFAULT_VARIANTS that has a bundle directory; legacy flat layout
    (bundles/<split>.pt, head_variant=None) when none exists."""
    from src.learning.bundle_io import bundle_path, load_split
    if head_variants:
        hvs: List[Optional[str]] = list(head_variants)
    else:
        hvs = [v for v in DEFAULT_VARIANTS if bundle_path(cfg, "val_gating_fit", v).parent.is_dir()] or [None]
    missing = [bundle_path(cfg, s, hv) for hv in hvs for s in REQUIRED_SPLITS if not bundle_path(cfg, s, hv).exists()]
    if missing:
        raise MissingBundlesError(missing)
    rows: List[Dict[str, Any]] = []
    for hv in hvs:
        b = {s: load_split(cfg, s, hv) for s in REQUIRED_SPLITS}
        rows += run_ablation(cfg, b["val_gating_fit"], b["test_benchmark"], b["s4_support"], b["s4_eval"],
                             s4_seeds=s4_seeds, head_variant=hv)
    write_tables(rows, out_dir)
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    from src.core.config import REPO_ROOT, load_config
    ap = argparse.ArgumentParser(prog="python -m src.evaluation.ablation", description="AEGIS final ablation")
    ap.add_argument("--head-variants", nargs="+", default=None, help="e.g. clean aug (default: auto-detect)")
    ap.add_argument("--out", default=str(REPO_ROOT / "reports"))
    ap.add_argument("--config", default=None)
    ap.add_argument("--s4-seeds", nargs="+", type=int, default=None)
    a = ap.parse_args(argv)
    try:
        run_all(load_config(a.config), a.head_variants, a.out, a.s4_seeds)
    except MissingBundlesError as e:
        log.error("%s", e)
        return 2
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
