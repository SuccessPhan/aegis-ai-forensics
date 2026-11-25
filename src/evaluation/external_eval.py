"""Evaluation on an external test set (v4): generators of other families / newer tools, never seen in training.

For every head variant it scores, on the external images:
  * the AEGIS image models (clip / dinov2 / freq experts, Static LR, AEGIS) from the external bundle, fitted exactly
    like the main ablation on val_gating_fit (no few-shot adaptation);
  * the three machine-learning models (RBF-SVM, random forest, XGBoost) on hand-crafted features and on the CLIP /
    DINOv2 embeddings, tuned on train_experts / val_gating_fit of the MAIN dataset -- the same models as notebook 04.

AUROC / accuracy are reported per external generator (its AI images vs every real image of the evaluation) and
pooled, next to the in-family S4 numbers, so the report can show "unseen, same family" vs "unseen, other family".

Real images: the external set's own `real/` folder when it has one. Sets without real images (e.g. Synthbuster) use
`real_from_main=True`: the clean real images of test_benchmark are added, so the AUROC measures how the new AI images
are separated from the real images the models already know (state this in the report).

CLI: python -m src.evaluation.external_eval --manifest data/normalized_v2/manifest.json \\
         --external data/external/newgen/manifest.json [--real-from-main] [--head-variants clean aug] [--B 1000]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.core.config import Config, load_config, resolve_path
from src.core.contracts import EXPERT_NAMES, SPLIT_TEST, SplitManifest
from src.core.logger import get_logger
from src.evaluation.metrics import expected_calibration_error
from src.evaluation.ml_comparison import (DEEP_METHODS, DEEP_SET, FEATURE_SETS, HEAD_VARIANTS, REFERENCE, load_all,
                                          open_caches)
from src.evaluation.significance import bootstrap_metrics, point_metrics, summarize_delta
from src.learning.classic_models import DEFAULT_MODELS, tune

log = get_logger(__name__)

RESULT_COLUMNS = ["external_set", "head_variant", "feature_set", "group", "generator", "model", "n", "n_ai", "n_real",
                  "auroc", "accuracy", "balanced_accuracy", "f1", "ece"]


def external_split(ext_man: SplitManifest) -> str:
    splits = {r.split for r in ext_man.records}
    if len(splits) != 1:
        raise ValueError(f"an external manifest holds exactly one split, got {sorted(splits)}")
    return splits.pop()


def external_bundle_path(cfg: Config, split: str, head_variant: str) -> Path:
    return resolve_path(cfg, "cache_dir") / "bundles" / head_variant / f"{split}.pt"


def _missing_embeddings(cfg: Config, shas: Sequence[str]) -> List[str]:
    from src.models.cache import FeatureCache
    out = []
    for n in ("clip", "dinov2"):
        c = FeatureCache.open(cfg, n)
        if any(not c.has(s) for s in shas):
            out.append(n)
    return out


def check_ready(cfg: Config, ext_man: SplitManifest, head_variants: Sequence[str], feature_sets: Sequence[str]
                ) -> List[str]:
    """Commands still to run before the evaluation (empty list = ready)."""
    split = external_split(ext_man)
    mpath = Path(ext_man.root) / "manifest.json"
    todo = []
    need_emb = {p for fs in feature_sets for p in fs.split("+") if p in ("clip", "dinov2")}
    for n in sorted(set(_missing_embeddings(cfg, [r.sha256 for r in ext_man.records])) & need_emb):
        todo.append(f"python -m src.models.extract --manifest {mpath} --expert {n} --splits {split}")
    for hv in head_variants:
        if not external_bundle_path(cfg, split, hv).exists():
            todo.append(f"python -m src.models.bundle --manifest {mpath} --split {split} --head-variant {hv} "
                        f"--out {external_bundle_path(cfg, split, hv)}")
    return todo


# ------------------------------------------------------------------------------------------------- predictions
def image_model_predictions(cfg: Config, head_variant: str, bundles: Sequence[Any], seed: int = 0
                            ) -> Dict[str, Dict[str, float]]:
    """{method: {sha: P(fake)}} of the AEGIS image models on the given bundles (fitted on val_gating_fit)."""
    from src.evaluation.ablation import _System
    from src.learning.bundle_io import load_split
    sysm = _System(cfg, load_split(cfg, "val_gating_fit", head_variant), True, True, seed)
    out: Dict[str, Dict[str, float]] = {m: {} for m in DEEP_METHODS}
    for b in bundles:
        fs = sysm.pipe.transform(b)
        preds = {f"{e}_expert": fs.p[:, j] for j, e in enumerate(EXPERT_NAMES)}
        preds["static_lr"] = sysm.models["static_lr"].predict(fs)
        preds["aegis"] = sysm.models["aegis"].predict(fs)
        for m, p in preds.items():
            out[m].update(zip(b.sha256, np.asarray(p, dtype=float)))
    return out


def evaluation_items(ext_man: SplitManifest, man: Optional[SplitManifest], real_from_main: bool
                     ) -> Tuple[List[str], np.ndarray, List[str], List[str]]:
    """(sha256, label, generator, origin) of the images to score: every external image (+ main test clean reals)."""
    recs = [(r.sha256, r.label, r.generator, "external") for r in ext_man.records]
    has_real = any(lb == 0 for _, lb, _, _ in recs)
    if real_from_main or not has_real:
        if man is None:
            raise ValueError("the external set has no real images: pass the main manifest with real_from_main=True")
        recs += [(r.sha256, 0, "real", "main_test") for r in man.by_split(SPLIT_TEST) if r.is_clean and r.label == 0]
        log.info("external evaluation: adding %d clean real images of test_benchmark",
                 sum(1 for x in recs if x[3] == "main_test"))
    if not any(lb == 1 for _, lb, _, _ in recs):
        raise ValueError("the external set has no AI images")
    shas, labels, gens, origin = zip(*recs)
    return list(shas), np.array(labels, dtype=int), list(gens), list(origin)


def _metrics(y: np.ndarray, p: np.ndarray, ece_bins: int) -> Dict[str, float]:
    pt = point_metrics(y, p)
    pred = (p >= 0.5).astype(int)
    tpr = float((pred[y == 1] == 1).mean()) if (y == 1).any() else float("nan")
    tnr = float((pred[y == 0] == 0).mean()) if (y == 0).any() else float("nan")
    tp, fp, fn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum()), \
        int(((pred == 0) & (y == 1)).sum())
    return {**pt, "balanced_accuracy": (tpr + tnr) / 2, "f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "ece": expected_calibration_error(y, p, ece_bins)}


def _slices(labels: np.ndarray, gens: List[str]) -> List[Tuple[str, np.ndarray]]:
    real = np.flatnonzero(labels == 0)
    out = [(g, np.concatenate([np.flatnonzero((labels == 1) & (np.array(gens) == g)), real]))
           for g in sorted({g for g, lb in zip(gens, labels) if lb == 1})]
    return out + [("all", np.arange(len(labels)))]


# ----------------------------------------------------------------------------------------------------- runner
def evaluate_external(cfg: Config, man: SplitManifest, ext_man: SplitManifest,
                      head_variants: Sequence[str] = HEAD_VARIANTS, feature_sets: Sequence[str] = FEATURE_SETS,
                      models: Sequence[str] = DEFAULT_MODELS, real_from_main: bool = False, B: int = 1000,
                      seed: Optional[int] = None, out_dir: Optional[str | Path] = None,
                      bundles: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, List[Dict[str, Any]]]:
    """Score every model on the external set. `bundles` ({head_variant: {'external': Bundle, 'test': Bundle}}) may be
    passed directly; otherwise they are loaded from the cache (run the commands of `check_ready` first)."""
    from src.learning.bundle_io import load_bundle, load_split
    seed = int(cfg.get("seed", 0) if seed is None else seed)
    name = ext_man.meta.get("external_set", external_split(ext_man))
    split = external_split(ext_man)
    if bundles is None:
        todo = check_ready(cfg, ext_man, head_variants, feature_sets)
        if todo:
            raise FileNotFoundError("external set not prepared yet; run:\n  " + "\n  ".join(todo))
    shas, y, gens, origin = evaluation_items(ext_man, man, real_from_main)
    ece_bins = int((cfg.get("evaluation") or {}).get("ece_bins", 15))
    slices = _slices(y, gens)
    results: List[Dict[str, Any]] = []
    sig: List[Dict[str, Any]] = []

    if "handcrafted" in feature_sets:
        from src.data.handcrafted import build_cache
        build_cache(cfg, ext_man)
        if "main_test" in origin:
            build_cache(cfg, man)

    for hv in head_variants:
        hb = (bundles or {}).get(hv) or {}
        ext_b = hb.get("external") or load_bundle(external_bundle_path(cfg, split, hv))
        bl = [ext_b]
        if "main_test" in origin:
            bl.append(hb.get("test") or load_split(cfg, SPLIT_TEST, hv))
        deep_by_sha = image_model_predictions(cfg, hv, bl, seed)
        scores: Dict[Tuple[str, str], np.ndarray] = {}
        for m, d in deep_by_sha.items():
            scores[(DEEP_SET, m)] = np.array([d[s] for s in shas], dtype=float)
        for fs in feature_sets:
            data = load_all(cfg, man, fs, hv)
            caches = open_caches(cfg, fs)
            X = np.concatenate([c.features_for(shas).numpy() for c in caches], axis=1).astype(np.float32)
            for m in models:
                tm = tune(m, data["train"].X, data["train"].y, data["val"].X, data["val"].y, seed=seed, cfg=cfg)
                scores[(fs, m)] = tm.predict_proba1(X)
        for gen, idx in slices:
            yy = y[idx]
            base = {"external_set": name, "head_variant": hv, "generator": gen, "n": len(idx),
                    "n_ai": int(yy.sum()), "n_real": int((yy == 0).sum())}
            for (fs, m), p in scores.items():
                results.append({**base, "feature_set": fs, "group": "deep" if fs == DEEP_SET else "ml", "model": m,
                                **_metrics(yy, p[idx], ece_bins)})
            if B > 0 and len(np.unique(yy)) == 2:
                ml_keys = [k for k in scores if k[0] != DEEP_SET]
                sc = {f"{fs}|{m}": scores[(fs, m)][idx] for fs, m in ml_keys}
                sc[REFERENCE] = scores[(DEEP_SET, REFERENCE)][idx]
                boot = bootstrap_metrics(yy, sc, B=B, seed=seed)
                ref = point_metrics(yy, sc[REFERENCE])
                for k in sc:
                    if k == REFERENCE:
                        continue
                    fs, m = k.split("|")
                    mp = point_metrics(yy, sc[k])
                    sig.append({**base, "feature_set": fs, "model": m, "metric": "auroc", "model_value": mp["auroc"],
                                "reference_value": ref["auroc"],
                                **summarize_delta(boot[k]["auroc"] - boot[REFERENCE]["auroc"],
                                                  mp["auroc"] - ref["auroc"]), "B": B})
    out = {"results": results, "significance": sig}
    if out_dir is not None:
        write_outputs(out, out_dir, name)
    return out


def summary_lines(results: List[Dict[str, Any]], metric: str = "auroc") -> List[str]:
    gens = sorted({r["generator"] for r in results}, key=lambda g: (g == "all", g))
    keys = list(dict.fromkeys((r["head_variant"], r["feature_set"], r["model"]) for r in results))
    keys.sort(key=lambda k: (k[0] != "clean", k[1] != DEEP_SET))
    idx = {(r["head_variant"], r["feature_set"], r["model"], r["generator"]): r[metric] for r in results}
    head = ["head", "features", "model"] + gens
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for k in keys:
        vals = [idx.get((*k, g)) for g in gens]
        lines.append("| " + " | ".join(list(k) + ["-" if v is None else f"{v:.3f}" for v in vals]) + " |")
    return lines


def write_outputs(out: Dict[str, List[Dict[str, Any]]], out_dir: str | Path, name: str) -> None:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    for key, cols in (("results", RESULT_COLUMNS), ("significance", None)):
        rows = out[key]
        if not rows:
            continue
        cols = cols or list(dict.fromkeys(k for r in rows for k in r))
        with (d / f"{key}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    r0 = out["results"][0]
    md = [f"# External test set `{name}`", "",
          f"AI images per generator vs {r0['n_real']} real images. AEGIS image models (`{DEEP_SET}`) and the three "
          "machine-learning models trained on the main GenImage splits; no adaptation to the new generators.", "",
          "## AUROC", ""] + summary_lines(out["results"], "auroc") + \
         ["", "## Balanced accuracy (threshold 0.5)", ""] + summary_lines(out["results"], "balanced_accuracy")
    (d / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log.info("external evaluation written to %s", d)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.evaluation.external_eval", description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", required=True, help="main GenImage manifest (training data of the models)")
    ap.add_argument("--external", required=True, help="external manifest from src.data.external")
    ap.add_argument("--head-variants", nargs="+", default=list(HEAD_VARIANTS), choices=list(HEAD_VARIANTS))
    ap.add_argument("--feature-sets", nargs="+", default=list(FEATURE_SETS))
    ap.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    ap.add_argument("--real-from-main", action="store_true", help="add the clean real images of test_benchmark")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--out", default=None, help="default: reports/external/<name>")
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    ext = SplitManifest.load(a.external)
    todo = check_ready(cfg, ext, a.head_variants, a.feature_sets)
    if todo:
        print("Run these first (GPU recommended for extract / bundle):\n  " + "\n  ".join(todo))
        return 1
    name = ext.meta.get("external_set", external_split(ext))
    evaluate_external(cfg, SplitManifest.load(a.manifest), ext, a.head_variants, a.feature_sets, a.models,
                      a.real_from_main, a.B, out_dir=a.out or Path("reports/external") / name)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
