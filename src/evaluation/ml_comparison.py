"""Machine learning vs the deep image models of AEGIS.

Question (machine-learning course): can classic machine-learning models detect AI-generated images as well as, or
better than, the deep image models AEGIS is built from? Three models (src.learning.classic_models: RBF-SVM, random
forest, XGBoost) are trained on three kinds of input and put next to the AEGIS models on the SAME test images:

  feature sets   handcrafted  92 features computed from the pixels (src.data.handcrafted): no deep network at all
                 clip, dinov2 frozen embeddings (same features as the AEGIS experts, a different classifier)
                 clip+dinov2  concatenation (optional)
  image models   clip_expert, dinov2_expert, freq_expert (SRM CNN), static_lr, aegis -- read from the bundles of the
                 same head variant (src.learning.bundle_io), i.e. exactly the models of the main ablation

Splits (as the main ablation, src.evaluation.ablation):
  train      = train_experts (`clean`: clean only; `aug`: clean + train_aug variants, as src.models.heads.train_records)
  validation = val_gating_fit: hyperparameter choice + Platt calibration of the machine-learning models
  test       = test_benchmark, read once after every model is fixed:
               S1 clean; S2 per JPEG level + 'all'; S3 per blur/resize level + 'all'
  S4         = s4_eval (unseen generators, NO few-shot adaptation for any method), per generator with a disjoint
               balanced Real part (same rule as ablation.partition_real) + 'all'

Significance: paired cluster bootstrap (by source image) of delta = metric(ML model) - metric(AEGIS), reusing
src.evaluation.significance.bootstrap_metrics.

Also provides `run_fusion_comparison`: the three models as meta-classifiers on the 12-d gating input, against Static LR
and AEGIS (fit on val_gating_fit, selected on val_gating_verify, evaluated on the test slices).

CLI: python -m src.evaluation.ml_comparison --manifest M [--feature-sets handcrafted clip dinov2]
         [--head-variants clean aug] [--models rbf_svm random_forest xgboost] [--out reports/ml_comparison] [--B 2000]
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from src.core.config import Config, load_config
from src.core.contracts import SPLIT_S4_EVAL, SPLIT_TEST, SPLIT_VAL_FIT, SplitManifest
from src.core.logger import get_logger
from src.evaluation.metrics import expected_calibration_error
from src.evaluation.significance import bootstrap_metrics, point_metrics, summarize_delta
from src.learning.classic_models import DEFAULT_MODELS, describe_models, tune

log = get_logger(__name__)

FEATURE_SETS = ("handcrafted", "clip", "dinov2")
FEATURE_PARTS = ("handcrafted", "clip", "dinov2")
HEAD_VARIANTS = ("clean", "aug")
DEEP_SET = "image_models"           # feature_set value of the AEGIS image-model rows
DEEP_METHODS: Dict[str, str] = {
    "clip_expert": "CLIP ViT-L/14 (frozen) + linear head",
    "dinov2_expert": "DINOv2 ViT-B/14 (frozen) + linear head",
    "freq_expert": "SRM high-pass CNN (1.17 M params, trained from scratch)",
    "static_lr": "3 experts, fixed learned weights (logistic regression on calibrated logits)",
    "aegis": "3 experts + per-image gating network (full AEGIS, no few-shot adaptation)",
}
REFERENCE = "aegis"                 # significance: delta = ML model - AEGIS
RESULT_COLUMNS = ["head_variant", "feature_set", "group", "scenario", "level", "generator", "model", "n", "auroc",
                  "accuracy", "f1", "ece", "brier"]
SIG_COLUMNS = ["head_variant", "feature_set", "scenario", "level", "generator", "model", "metric", "model_value",
               "reference_value", "delta", "ci_lo", "ci_hi", "p_value", "significant", "n", "n_clusters", "B"]


# ------------------------------------------------------------------------------------------------------ data
@dataclass
class EmbeddingSplit:
    X: np.ndarray
    y: np.ndarray
    sha256: List[str]
    cluster: List[str]                  # parent sha (degraded variants share their clean image's id)
    scenario: List[Optional[str]]
    level: List[Optional[str]]          # DegradationSpec.name, None if clean
    generator: List[str]

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, m: np.ndarray) -> "EmbeddingSplit":
        idx = np.flatnonzero(m) if np.asarray(m).dtype == bool else np.asarray(m, dtype=int)
        pick = lambda lst: [lst[i] for i in idx]  # noqa: E731
        return EmbeddingSplit(self.X[idx], self.y[idx], pick(self.sha256), pick(self.cluster), pick(self.scenario),
                              pick(self.level), pick(self.generator))


def open_caches(cfg: Config, feature_set: str, man: Optional[SplitManifest] = None) -> list:
    """Feature caches of a feature set ('handcrafted', 'clip', 'dinov2' or a '+' combination). Hand-crafted features
    are computed on first use when a manifest is given (resumable, a few minutes on a CPU)."""
    from src.models.cache import FeatureCache
    names = feature_set.split("+")
    for n in names:
        if n not in FEATURE_PARTS:
            raise ValueError(f"feature set parts must be one of {FEATURE_PARTS}, got {n!r}")
    caches = []
    for n in names:
        if n == "handcrafted":
            from src.data.handcrafted import build_cache, open_cache
            caches.append(build_cache(cfg, man) if man is not None else open_cache(cfg))
        else:
            caches.append(FeatureCache.open(cfg, n))
    return caches


def load_split(man: SplitManifest, records, caches) -> EmbeddingSplit:
    shas = [r.sha256 for r in records]
    X = np.concatenate([c.features_for(shas).numpy() for c in caches], axis=1).astype(np.float32)
    clean = [r.is_clean for r in records]
    return EmbeddingSplit(
        X, np.array([r.label for r in records], dtype=int), shas,
        [r.parent_sha256 or r.sha256 for r in records],
        [None if c else r.degradation.scenario for r, c in zip(records, clean)],
        [None if c else r.degradation.name for r, c in zip(records, clean)],
        [r.generator for r in records])


def load_all(cfg: Config, man: SplitManifest, feature_set: str, head_variant: str) -> Dict[str, EmbeddingSplit]:
    from src.models.heads import train_records
    caches = open_caches(cfg, feature_set, man)
    return {"train": load_split(man, train_records(man, head_variant), caches),
            "val": load_split(man, man.by_split(SPLIT_VAL_FIT), caches),
            "test": load_split(man, man.by_split(SPLIT_TEST), caches),
            "s4": load_split(man, man.by_split(SPLIT_S4_EVAL), caches)}


# ----------------------------------------------------------------------------------------------------- slices
def _balanced_real(s4: EmbeddingSplit, gens: Sequence[str], seed: int) -> Dict[str, np.ndarray]:
    """Disjoint Real parts per unseen generator, same rule as src.evaluation.ablation.partition_real."""
    gens = sorted(gens)
    real = sorted(np.flatnonzero(s4.y == 0), key=lambda i: s4.sha256[i])
    perm = [real[i] for i in np.random.default_rng(seed).permutation(len(real))]
    want = {g: int(sum(1 for lb, gg in zip(s4.y, s4.generator) if lb == 1 and gg == g)) for g in gens}
    if sum(want.values()) > len(perm):
        want = {g: len(perm) // max(len(gens), 1) for g in gens}
    out, pos = {}, 0
    for g in gens:
        out[g] = np.array(perm[pos:pos + want[g]], dtype=int)
        pos += want[g]
    return out


def iter_slices(test: EmbeddingSplit, s4: Optional[EmbeddingSplit], seed: int = 0
                ) -> Iterator[Tuple[str, str, str, str, np.ndarray]]:
    """(scenario, level, generator, source, index array into test/s4) -- same slicing as the ablation."""
    sc = np.array([s or "" for s in test.scenario])
    lv = np.array([s or "" for s in test.level])
    yield "S1", "clean", "all", "test", np.flatnonzero(sc == "")
    for scen in ("S2", "S3"):
        m = sc == scen
        if not m.any():
            continue
        for lvl in sorted(set(lv[m])):
            yield scen, lvl, "all", "test", np.flatnonzero(m & (lv == lvl))
        yield scen, "all", "all", "test", np.flatnonzero(m)
    if s4 is None or len(s4) == 0:
        return
    gens = sorted({g for g, lb in zip(s4.generator, s4.y) if lb == 1})
    real = _balanced_real(s4, gens, seed)
    for g in gens:
        ai = np.array([i for i, (lb, gg) in enumerate(zip(s4.y, s4.generator)) if lb == 1 and gg == g], dtype=int)
        yield "S4", "all", g, "s4", np.concatenate([ai, real[g]])
    yield "S4", "all", "all", "s4", np.arange(len(s4))


def slice_metrics(y: np.ndarray, p: np.ndarray, ece_bins: int = 15) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    pred = (p >= 0.5).astype(int)
    tp, fp, fn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum()), \
        int(((pred == 0) & (y == 1)).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    return {**point_metrics(y, p), "f1": float(f1), "ece": expected_calibration_error(y, p, ece_bins),
            "brier": float(np.mean((p - y) ** 2))}


# ------------------------------------------------------------------------------------------ AEGIS image models
def deep_predictions(cfg: Config, head_variant: str, seed: Optional[int] = None,
                     bundles: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Dict[str, float]]]:
    """P(fake) of every AEGIS image model, keyed by image sha256: {'test'|'s4': {method: {sha: p}}}.
    Fitted exactly like the main ablation (calibration + Static LR + gating on val_gating_fit); S4 without
    adaptation, so every method in the comparison sees the unseen generators cold."""
    from src.core.contracts import EXPERT_NAMES
    from src.evaluation.ablation import _System
    seed = int(cfg.get("seed", 0) if seed is None else seed)
    if bundles is None:
        from src.learning.bundle_io import bundle_path, load_split as load_bundle
        need = ["val_gating_fit", "test_benchmark", "s4_eval"]
        missing = [s for s in need if not bundle_path(cfg, s, head_variant).exists()]
        if missing:
            raise FileNotFoundError(f"bundles {missing} ({head_variant}) not found: run notebook 02 "
                                    f"(src.models.bundle) first")
        bundles = {s: load_bundle(cfg, s, head_variant) for s in need}
    sysm = _System(cfg, bundles["val_gating_fit"], True, True, seed)
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for key, split in (("test", "test_benchmark"), ("s4", "s4_eval")):
        b = bundles.get(split)
        if b is None:
            continue
        fs = sysm.pipe.transform(b)
        preds = {f"{e}_expert": fs.p[:, j] for j, e in enumerate(EXPERT_NAMES)}
        preds["static_lr"] = sysm.models["static_lr"].predict(fs)
        preds["aegis"] = sysm.models["aegis"].predict(fs)
        out[key] = {m: dict(zip(b.sha256, np.asarray(p, dtype=float))) for m, p in preds.items()}
    return out


def align_predictions(by_sha: Dict[str, Dict[str, Dict[str, float]]], data: Dict[str, EmbeddingSplit]
                      ) -> Dict[str, Dict[str, np.ndarray]]:
    """{split: {method: {sha: p}}} -> {split: {method: array in the order of data[split].sha256}}."""
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for key, methods in by_sha.items():
        shas = data[key].sha256
        out[key] = {}
        for m, d in methods.items():
            miss = sum(1 for s in shas if s not in d)
            if miss:
                raise KeyError(f"{miss} {key} images missing from the {m} predictions: bundles and manifest differ")
            out[key][m] = np.array([d[s] for s in shas], dtype=float)
    return out


# ---------------------------------------------------------------------------------------------------- runner
def compare_on_embeddings(cfg: Config, data: Dict[str, EmbeddingSplit], models: Sequence[str] = DEFAULT_MODELS,
                          B: int = 2000, seed: Optional[int] = None, tags: Optional[Dict[str, Any]] = None,
                          fixed: Optional[Dict[str, Dict[str, np.ndarray]]] = None, fixed_rows: bool = True
                          ) -> Dict[str, List[Dict[str, Any]]]:
    """Tune every ML model on train/val, then evaluate test + s4 once, next to the `fixed` predictions of the AEGIS
    image models (aligned with data[split]). Returns {'results', 'significance', 'tuning', 'timing'} row lists.
    `fixed_rows=False` skips the result rows of the fixed models (they are written once per head variant)."""
    seed = int(cfg.get("seed", 0) if seed is None else seed)
    tags = dict(tags or {})
    fixed = fixed or {}
    tr, va = data["train"], data["val"]
    tuned = {m: tune(m, tr.X, tr.y, va.X, va.y, seed=seed, cfg=cfg) for m in models}

    # ---- test / S4 are read only from here on, every model is fixed
    preds: Dict[str, Dict[str, np.ndarray]] = {"test": {}, "s4": {}}
    timing = []
    for m, tm in tuned.items():
        pt, secs = tm.predict_timed(data["test"].X)
        preds["test"][m] = pt
        if len(data["s4"]):
            preds["s4"][m] = tm.predict_proba1(data["s4"].X)
        timing.append({**tags, "model": m, "params": json.dumps(tm.params), "val_auroc": tm.val_auroc,
                       "fit_seconds": tm.fit_seconds,
                       "predict_ms_per_1k": 1000.0 * secs / max(len(data["test"]), 1) * 1000.0})

    ece_bins = int((cfg.get("evaluation") or {}).get("ece_bins", 15))
    results, sig = [], []
    for scen, lvl, gen, src, idx in iter_slices(data["test"], data["s4"], seed):
        sub = data[src].subset(idx)
        if len(sub) == 0:
            continue
        cell = {"scenario": scen, "level": lvl, "generator": gen}
        deep = fixed.get(src, {})
        if fixed_rows:
            for m in DEEP_METHODS:
                if m in deep:
                    results.append({**tags, "feature_set": DEEP_SET, "group": "deep", **cell, "model": m,
                                    "n": len(sub), **slice_metrics(sub.y, deep[m][idx], ece_bins)})
        for m in tuned:
            results.append({**tags, "group": "ml", **cell, "model": m, "n": len(sub),
                            **slice_metrics(sub.y, preds[src][m][idx], ece_bins)})
        if B <= 0 or REFERENCE not in deep or len(np.unique(sub.y)) < 2:
            continue
        cl = np.unique(np.array(sub.cluster), return_inverse=True)[1]
        scores = {m: preds[src][m][idx] for m in tuned}
        scores[REFERENCE] = deep[REFERENCE][idx]
        boot = bootstrap_metrics(sub.y, scores, B=B, seed=seed, clusters=cl)
        ref_pt = point_metrics(sub.y, scores[REFERENCE])
        for m in tuned:
            mp = point_metrics(sub.y, scores[m])
            for met in ("auroc", "accuracy"):
                res = summarize_delta(boot[m][met] - boot[REFERENCE][met], mp[met] - ref_pt[met])
                sig.append({**tags, **cell, "model": m, "metric": met, "model_value": mp[met],
                            "reference_value": ref_pt[met], **res, "n": len(sub),
                            "n_clusters": int(cl.max() + 1), "B": B})
    trials = [{**tags, **t} for tm in tuned.values() for t in tm.trials]
    return {"results": results, "significance": sig, "tuning": trials, "timing": timing}


def run_comparison(cfg: Config, man: SplitManifest, feature_sets: Sequence[str] = FEATURE_SETS,
                   head_variants: Sequence[str] = HEAD_VARIANTS, models: Sequence[str] = DEFAULT_MODELS,
                   out_dir: Optional[str | Path] = None, B: int = 2000, seed: Optional[int] = None,
                   include_image_models: bool = True) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {"results": [], "significance": [], "tuning": [], "timing": []}
    for hv in head_variants:
        by_sha = deep_predictions(cfg, hv, seed) if include_image_models else {}
        for i, fs in enumerate(feature_sets):
            log.info("=== head_variant=%s feature_set=%s", hv, fs)
            data = load_all(cfg, man, fs, hv)
            log.info("train %d | val %d | test %d | s4 %d  (dim %d)", len(data["train"]), len(data["val"]),
                     len(data["test"]), len(data["s4"]), data["train"].X.shape[1])
            part = compare_on_embeddings(cfg, data, models, B, seed, {"head_variant": hv, "feature_set": fs},
                                         fixed=align_predictions(by_sha, data), fixed_rows=(i == 0))
            for k in out:
                out[k] += part[k]
    if out_dir is not None:
        write_outputs(out, out_dir)
    return out


# ------------------------------------------------------------------------------------------ fusion-level variant
def run_fusion_comparison(cfg: Config, head_variant: Optional[str] = "aug", models: Sequence[str] = DEFAULT_MODELS,
                          seed: Optional[int] = None, bundles: Optional[Dict[str, Any]] = None
                          ) -> List[Dict[str, Any]]:
    """The ML models as meta-classifiers on the 12-d gating input vs Static LR / naive average / AEGIS (variant 'full').
    Fit on val_gating_fit, hyperparameters on val_gating_verify (if that bundle exists, else default params), test once.
    `bundles` may be passed directly ({split: Bundle}); otherwise they are loaded from the cache."""
    from src.evaluation.ablation import _slices, _System
    from src.learning.classic_models import SklearnStacker, tune_stacker
    seed = int(cfg.get("seed", 0) if seed is None else seed)
    if bundles is None:
        from src.learning.bundle_io import bundle_path, load_split as load_bundle
        need = ["val_gating_fit", "test_benchmark"]
        opt = [s for s in ("val_gating_verify", "s4_eval") if bundle_path(cfg, s, head_variant).exists()]
        bundles = {s: load_bundle(cfg, s, head_variant) for s in need + opt}
    sysm = _System(cfg, bundles["val_gating_fit"], True, True, seed)
    ver = bundles.get("val_gating_verify")
    ver_fs = sysm.pipe.transform(ver) if ver is not None else None
    stackers, chosen = {}, {}
    for m in models:
        if ver_fs is not None:
            st, params, _ = tune_stacker(m, sysm.fit_fs, sysm.fit_y, ver_fs, ver.label, seed, cfg)
        else:
            st, params = SklearnStacker(m, seed, cfg).fit(sysm.fit_fs, sysm.fit_y), {}
        stackers[f"stack_{m}"], chosen[f"stack_{m}"] = st, params
    base = {k: sysm.models[k] for k in ("naive_avg", "static_lr", "aegis")}
    ece_bins = int((cfg.get("evaluation") or {}).get("ece_bins", 15))
    rows = []
    slices = list(_slices(bundles["test_benchmark"]))
    if bundles.get("s4_eval") is not None:
        slices.append(("S4", "all", bundles["s4_eval"]))
    for sc, lvl, b in slices:
        fs = sysm.pipe.transform(b)
        for name, model in {**base, **stackers}.items():
            p = model.predict(fs)
            rows.append({"head_variant": head_variant, "scenario": sc, "level": lvl, "method": name,
                         "params": json.dumps(chosen.get(name, {})), "n": len(b),
                         **slice_metrics(b.label, p, ece_bins)})
    return rows


# ---------------------------------------------------------------------------------------------------- output
def _write_csv(rows: List[Dict[str, Any]], path: Path, columns: Optional[List[str]] = None) -> None:
    if not rows:
        return
    cols = columns or list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def summary_table(results: List[Dict[str, Any]], metric: str = "auroc") -> List[str]:
    """Markdown: one row per (head_variant, feature_set, model), AEGIS image models first; columns = pooled
    scenarios."""
    cols = [("S1", "clean", "all"), ("S2", "all", "all"), ("S3", "all", "all")]
    cols += sorted({(r["scenario"], r["level"], r["generator"]) for r in results if r["scenario"] == "S4"},
                   key=lambda c: (c[2] == "all", c[2]))
    idx = {(r["head_variant"], r["feature_set"], r["model"], r["scenario"], r["level"], r["generator"]): r[metric]
           for r in results}
    keys = list(dict.fromkeys((r["head_variant"], r["feature_set"], r["model"]) for r in results))
    keys.sort(key=lambda k: (k[0] != "clean", k[1] != DEEP_SET))      # stable: keeps the run order otherwise
    head = ["head", "features", "model"] + [f"{s} {g if s == 'S4' else l}" for s, l, g in cols]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for hv, fs, m in keys:
        vals = [idx.get((hv, fs, m, *c)) for c in cols]
        lines.append("| " + " | ".join([str(hv), fs, m] + ["-" if v is None else f"{v:.3f}" for v in vals]) + " |")
    return lines


def write_outputs(out: Dict[str, List[Dict[str, Any]]], out_dir: str | Path) -> None:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    _write_csv(out["results"], d / "results.csv", RESULT_COLUMNS)
    _write_csv(out["significance"], d / "significance.csv", SIG_COLUMNS)
    _write_csv(out["tuning"], d / "tuning.csv")
    _write_csv(out["timing"], d / "timing.csv")
    md = ["# Machine learning vs the deep image models of AEGIS", "",
          f"Rows with features `{DEEP_SET}` are the AEGIS models (deep learning); the other rows are the "
          "machine-learning models on hand-crafted features or on frozen embeddings. AUROC on test_benchmark / "
          "s4_eval (S4 without few-shot adaptation for every method). Significance (significance.csv) is ML model "
          f"minus `{REFERENCE}`.", ""]
    md += summary_table(out["results"], "auroc") + ["", "## Accuracy", ""] + summary_table(out["results"], "accuracy")
    present = {r["model"] for r in out["results"]}
    md += ["", "## Models", "", "| model | group | description |", "|---|---|---|"]
    md += [f"| {m} | deep learning (AEGIS) | {d} |" for m, d in DEEP_METHODS.items() if m in present]
    md += [f"| {r['model']} | machine learning ({r['family']}) | {r['note']}; grid {r['grid']} |"
           for r in describe_models([m for m in DEFAULT_MODELS if m in present])]
    (d / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log.info("ML comparison written to %s", d)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.evaluation.ml_comparison", description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--feature-sets", nargs="+", default=list(FEATURE_SETS))
    ap.add_argument("--head-variants", nargs="+", default=list(HEAD_VARIANTS), choices=list(HEAD_VARIANTS))
    ap.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    ap.add_argument("--out", default="reports/ml_comparison")
    ap.add_argument("--B", type=int, default=2000, help="bootstrap resamples (0 = skip significance)")
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    run_comparison(cfg, SplitManifest.load(a.manifest), a.feature_sets, a.head_variants, a.models, a.out, a.B)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
