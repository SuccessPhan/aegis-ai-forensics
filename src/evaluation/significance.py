"""Paired bootstrap significance of AEGIS vs every baseline (post-hoc ANALYSIS of saved predictions).

Input : <pred_dir>/predictions_<head_variant>.npz written by `ablation --save-predictions` (variant 'full' only is used).
Output: significance.csv, significance.md.

This module fits / tunes / selects NOTHING: it only re-reads predictions that were produced by models fitted on
val_gating_fit and evaluated once on test_benchmark / s4_eval. It never feeds back into any decision.

Method
  * One bootstrap per evaluation set (head_variant, scenario, level, generator): B resamples with replacement
    (default B = 2000, seed = 0; overridable via cfg['evaluation']['bootstrap'] = {n_resamples, seed}). The SAME resampled
    sample counts are used for every method, group and seed of that evaluation set (paired).
  * Optional cluster bootstrap (`--manifest`): when a manifest is given, samples are resampled by source image
    (parent_sha256 or own sha256), so a clean image and its degraded variants move together (matters for the 'all'
    cells of S2/S3). Without a manifest every sample is its own cluster (plain iid bootstrap).
  * Metrics: AUROC (tie-aware, computed from resample weights) and Accuracy (threshold 0.5), identical to compute_metrics.
  * Delta = metric(AEGIS) - metric(baseline). CI = 2.5 / 97.5 percentiles of the bootstrap deltas (95%).
    p-value (two-sided) = min(1, 2 * min(P_b[delta_b <= 0], P_b[delta_b >= 0])); p = 0 means < 1/B.
  * S4 (several support seeds): for every resample the delta is the MEAN over seeds s of
    metric(AEGIS_s) - metric(baseline_s), all on the same resampled eval indices. static_learned_support is paired
    seed by seed (same support set); seed-free baselines (clip, dinov2, freq, naive_avg, static_lr, quality_only) and
    aegis_noadapt are the same for every seed. The point estimate is the same mean over seeds (matches ablation.csv).
    NB: this captures eval-set sampling noise only, not the variance across support sets (the seed spread is reported
    as `seed_sd` of the per-seed deltas).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

from src.core.logger import get_logger
from src.evaluation.predictions import load_predictions, parse_key

log = get_logger(__name__)

DEFAULT_B = 2000          # ASSUMPTION: not in configs/default.yaml; overridable via evaluation.bootstrap.n_resamples
DEFAULT_SEED = 0          # ASSUMPTION: evaluation.bootstrap.seed
CHUNK = 200
BASELINES = ("clip", "dinov2", "freq", "naive_avg", "static_lr", "quality_only")
S4_BASELINES = ("static_learned_support", "aegis_noadapt")
STRONG_POOL = ("clip", "dinov2", "freq", "naive_avg", "static_lr", "static_learned_support")   # quality_only = floor
METRICS = ("auroc", "accuracy")
COLUMNS = ["head_variant", "scenario", "level", "generator", "N", "baseline", "metric", "aegis", "baseline_value",
           "delta", "ci_lo", "ci_hi", "p_value", "significant", "n", "n_clusters", "n_seeds", "seed_sd", "B"]


# ---------------------------------------------------------------------------------------------- bootstrap core
def _auc_weighted(C: np.ndarray, y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Tie-aware AUROC for each row of the (b, n) non-negative sample-weight matrix C."""
    n = len(p)
    order = np.argsort(p, kind="mergesort")
    ps, ys = p[order], y[order]
    new = np.concatenate([[True], ps[1:] != ps[:-1]])
    gid = np.cumsum(new) - 1
    starts = np.flatnonzero(new)
    ends = np.concatenate([starts[1:], [n]])
    Cs = C[:, order]
    neg = Cs * (ys == 0)
    S = np.concatenate([np.zeros((C.shape[0], 1)), np.cumsum(neg, axis=1)], axis=1)
    val = 0.5 * (S[:, starts] + S[:, ends])[:, gid]               # negatives below + half of the tied group
    pos = Cs * (ys == 1)
    den = pos.sum(1) * neg.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (pos * val).sum(1) / den, np.nan)


def _acc_weighted(C: np.ndarray, y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return (C @ ((p >= 0.5).astype(int) == y).astype(float)) / C.sum(1)


def bootstrap_metrics(y: np.ndarray, preds: Dict[str, np.ndarray], B: int = DEFAULT_B, seed: int = DEFAULT_SEED,
                      clusters: Optional[np.ndarray] = None, chunk: int = CHUNK) -> Dict[str, Dict[str, np.ndarray]]:
    """Paired bootstrap of AUROC / accuracy for every named prediction vector (same resample for all).
    Returns {name: {'auroc': (B,), 'accuracy': (B,)}}. `clusters` (n,) int ids -> cluster bootstrap."""
    y = np.asarray(y).astype(int)
    n = len(y)
    cl = np.arange(n) if clusters is None else np.unique(clusters, return_inverse=True)[1]
    K = int(cl.max()) + 1
    rng = np.random.default_rng(seed)
    out = {k: {m: np.empty(B) for m in METRICS} for k in preds}
    for s in range(0, B, chunk):
        b = min(chunk, B - s)
        draw = rng.integers(0, K, (b, K))
        cnt = np.bincount((draw + np.arange(b)[:, None] * K).ravel(), minlength=b * K).reshape(b, K)
        C = cnt[:, cl].astype(float)
        for k, p in preds.items():
            p = np.asarray(p, dtype=float)
            out[k]["auroc"][s:s + b] = _auc_weighted(C, y, p)
            out[k]["accuracy"][s:s + b] = _acc_weighted(C, y, p)
    return out


def point_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    return {"auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
            "accuracy": float(((np.asarray(p) >= 0.5).astype(int) == y).mean())}


def summarize_delta(delta_b: np.ndarray, point: float) -> Dict[str, float]:
    d = delta_b[~np.isnan(delta_b)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = min(1.0, 2.0 * min(float((d <= 0).mean()), float((d >= 0).mean())))
    return {"delta": point, "ci_lo": float(lo), "ci_hi": float(hi), "p_value": p,
            "significant": bool(lo > 0 or hi < 0)}


# --------------------------------------------------------------------------------------------- cell assembly
def _cluster_ids(shas: Sequence[str], parent: Optional[Dict[str, str]]) -> Optional[np.ndarray]:
    if not parent:
        return None
    return np.unique(np.array([parent.get(s, s) for s in shas]), return_inverse=True)[1]


def analyse_head_variant(groups: Dict[str, Dict[str, np.ndarray]], head_variant: Optional[str], B: int = DEFAULT_B,
                         seed: int = DEFAULT_SEED, parent: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """All (scenario, level, generator, N) cells of variant 'full' for one head variant -> list of result rows."""
    parsed = {k: parse_key(k) for k in groups}
    full = {k: v for k, v in parsed.items() if v["variant"] == "full"}
    sets: Dict[Tuple, List[str]] = {}
    for k, v in full.items():
        sets.setdefault((v["scenario"], v["level"], v["generator"]), []).append(k)
    rows: List[Dict[str, Any]] = []
    for (sc, lvl, gen), keys in sorted(sets.items(), key=lambda t: tuple(map(str, t[0]))):
        ref = groups[keys[0]]
        y, shas = ref["y"], ref["sha256"]
        preds = {(k, f[2:]): groups[k][f] for k in keys for f in groups[k] if f.startswith("p_")}
        for k in keys:
            if not np.array_equal(groups[k]["sha256"], shas):
                raise ValueError(f"eval samples differ between groups of {sc}/{lvl}/{gen}")
        cl = _cluster_ids(shas, parent)
        boot = bootstrap_metrics(y, {f"{k}|{m}": p for (k, m), p in preds.items()}, B, seed, cl)
        pt = {(k, m): point_metrics(y, p) for (k, m), p in preds.items()}
        n_cl = int(len(np.unique(cl))) if cl is not None else len(y)
        key_of = {(parsed[k]["N"], parsed[k]["seed"], m): k for (k, m) in preds}

        def get(N, s, m):
            k = key_of[(N, s, m)]
            return boot[f"{k}|{m}"], pt[(k, m)]

        ad_keys = sorted((N, s) for (N, s, m) in key_of if m == "aegis")
        cells = sorted({N for N, _ in ad_keys}, key=lambda v: (v is not None, v))
        for N in cells:
            seeds = [s for n_, s in ad_keys if n_ == N]
            a = [get(N, s, "aegis") for s in seeds]
            names: List[Tuple[str, Any]] = []
            for m in BASELINES:
                if (None, None, m) in key_of:
                    names.append((m, lambda s, m=m: get(None, None, m)))
            if N is not None:
                if (N, seeds[0], "static_learned_support") in key_of:
                    names.append(("static_learned_support", lambda s: get(N, s, "static_learned_support")))
                if (0, None, "aegis_noadapt") in key_of:
                    names.append(("aegis_noadapt", lambda s: get(0, None, "aegis_noadapt")))
            for bname, fetch in names:
                bs = [fetch(s) for s in seeds]
                for met in METRICS:
                    d_seed = [pa[met] - pb[met] for (_, pa), (_, pb) in zip(a, bs)]
                    d_b = np.mean([ba[met] - bb[met] for (ba, _), (bb, _) in zip(a, bs)], axis=0)
                    res = summarize_delta(d_b, float(np.mean(d_seed)))
                    rows.append({"head_variant": head_variant, "scenario": sc, "level": lvl, "generator": gen, "N": N,
                                 "baseline": bname, "metric": met,
                                 "aegis": float(np.mean([pa[met] for _, pa in a])),
                                 "baseline_value": float(np.mean([pb[met] for _, pb in bs])), **res,
                                 "n": len(y), "n_clusters": n_cl, "n_seeds": len(seeds),
                                 "seed_sd": float(np.std(d_seed)) if len(seeds) > 1 else None, "B": B})
    return rows


# --------------------------------------------------------------------------------------------------- output
def _fmt(v: Any, nd: int = 4) -> str:
    return "" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def _cell_str(r: Optional[Dict[str, Any]]) -> str:
    if r is None:
        return "-"
    p = "<0.001" if r["p_value"] < 0.001 else f"{r['p_value']:.3f}"
    return f"{r['delta']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] p={p}{' *' if r['significant'] else ''}"


def compact_table(rows: List[Dict[str, Any]], metric: str = "auroc") -> List[str]:
    """Per cell: AEGIS vs strongest baseline (by point AUROC among clip/dinov2/freq/naive_avg/static_lr/
    static_learned_support), vs static_lr, vs naive_avg and (S4) vs static_learned_support."""
    lines = []
    for hv in dict.fromkeys(r["head_variant"] for r in rows):
        sub = [r for r in rows if r["head_variant"] == hv and r["metric"] == metric]
        lines += [f"## head variant = {hv}  (delta {metric}: AEGIS - baseline; [95% CI]; p two-sided; * = CI excludes 0)",
                  "", "| scenario | level | generator | N | strongest baseline | vs strongest | vs static_lr | vs naive_avg "
                  "| vs static_learned_support |", "|---|---|---|---|---|---|---|---|---|"]
        cells = dict.fromkeys((r["scenario"], r["level"], r["generator"], r["N"]) for r in sub)
        for c in cells:
            cr = {r["baseline"]: r for r in sub if (r["scenario"], r["level"], r["generator"], r["N"]) == c}
            pool = [cr[b] for b in STRONG_POOL if b in cr]
            best = min(pool, key=lambda r: r["delta"]) if pool else None      # smallest delta = strongest baseline
            lines.append(f"| {c[0]} | {_fmt(c[1])} | {_fmt(c[2])} | {_fmt(c[3])} | {best['baseline'] if best else '-'} | "
                         f"{_cell_str(best)} | {_cell_str(cr.get('static_lr'))} | {_cell_str(cr.get('naive_avg'))} | "
                         f"{_cell_str(cr.get('static_learned_support'))} |")
        lines.append("")
    return lines


def write_outputs(rows: List[Dict[str, Any]], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "significance.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    B = rows[0]["B"] if rows else DEFAULT_B
    head = ["# AEGIS significance (paired bootstrap)", "",
            f"B = {B} resamples, shared across methods; percentile 95% CI; two-sided p = min(1, 2*min tail). "
            "S4: delta averaged over support seeds on the same resampled eval samples. Post-hoc analysis of saved "
            "predictions; nothing is fitted or selected here.", ""]
    (out / "significance.md").write_text("\n".join(head + compact_table(rows, "auroc") + [""]
                                                   + ["# Accuracy", ""] + compact_table(rows, "accuracy")) + "\n",
                                         encoding="utf-8")
    log.info("significance written to %s (%d rows)", out, len(rows))


def load_parent_map(manifest_path: str | Path) -> Dict[str, str]:
    """sha256 -> source key (parent_sha256 for degraded variants, own sha256 for clean images)."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        recs = json.load(f)["records"]
    return {r["sha256"]: (r.get("parent_sha256") or r["sha256"]) for r in recs}


def run(pred_dir: str | Path, out_dir: str | Path, B: Optional[int] = None, seed: Optional[int] = None,
        manifest: Optional[str | Path] = None, cfg: Optional[dict] = None) -> List[Dict[str, Any]]:
    bc = ((cfg or {}).get("evaluation") or {}).get("bootstrap") or {}
    B = int(B or bc.get("n_resamples", DEFAULT_B))
    seed = int(bc.get("seed", DEFAULT_SEED) if seed is None else seed)
    parent = load_parent_map(manifest) if manifest else None
    rows: List[Dict[str, Any]] = []
    files = sorted(Path(pred_dir).glob("predictions_*.npz"))
    if not files:
        raise FileNotFoundError(f"no predictions_*.npz in {pred_dir}")
    for f in files:
        hv = f.stem[len("predictions_"):]
        log.info("bootstrap %s (B=%d)", f.name, B)
        rows += analyse_head_variant(load_predictions(f), hv, B, seed, parent)
    write_outputs(rows, out_dir)
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    from src.core.config import load_config
    ap = argparse.ArgumentParser(prog="python -m src.evaluation.significance")
    ap.add_argument("--pred", required=True, help="directory with predictions_<variant>.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=None, help="split manifest -> cluster bootstrap by source image")
    ap.add_argument("--B", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)
    run(a.pred, a.out, a.B, a.seed, a.manifest, load_config(a.config))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
