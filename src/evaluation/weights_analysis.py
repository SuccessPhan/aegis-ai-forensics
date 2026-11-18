"""Does the AEGIS gating actually move its weights with the degradation level? (descriptive analysis, nothing is tuned)

(a) val_gating_verify (continuous random JPEG Q in [30,95] / blur sigma in [0.5,3.0]; NOT used to fit the gating, so it
    is a legitimate place to describe behaviour): mean (+-sd) of w_clip, w_dinov2, w_freq per JPEG-Q bin (width 10, last
    bin 90-95) and per blur-sigma bin (width 0.5), plus a "clean" reference. Q / sigma are taken from the split manifest
    (`degradation.jpeg_quality`, `degradation.blur_sigma`) joined on sha256 -- no parsing of file names. The system is
    refitted exactly like the ablation does (variant 'full': calibration + OOD features, fit on val_gating_fit only).
(b) test_benchmark at the fixed levels: mean +- sd of w per level, read from the predictions saved by
    `ablation --save-predictions` (the very same single test evaluation; nothing is re-run on the test set).
Per-expert AUROC (calibrated probabilities) is reported over the same bins/levels next to the weights, to see whether
the weight falls where the expert gets worse. Outputs: weights_by_jpeg.png, weights_by_blur.png (top row weights,
bottom row expert AUROC, one column per head variant, 150 dpi), weights_summary.csv, weights_summary.md.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

from src.core.config import Config
from src.core.contracts import EXPERT_NAMES
from src.core.logger import get_logger
from src.evaluation.predictions import load_predictions, parse_key
from src.learning.bundle_io import Bundle

log = get_logger(__name__)

JPEG_EDGES = [30, 40, 50, 60, 70, 80, 90, 95]            # bins [30,40) ... [90,95]
BLUR_EDGES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]              # bins [0.5,1.0) ... [2.5,3.0]
TEST_LEVELS = ["clean", "jpeg_q85", "jpeg_q65", "jpeg_q40", "blur_k11_s1.5", "blur_k17_s2.5", "resize_0.5"]
COLUMNS = ["head_variant", "split", "axis", "bin", "n"] + [f"w_{e}_mean" for e in EXPERT_NAMES] + \
          [f"w_{e}_sd" for e in EXPERT_NAMES] + [f"auroc_{e}" for e in EXPERT_NAMES] + ["auroc_aegis"]


def load_degradation_params(manifest_path: str | Path) -> Dict[str, Dict[str, Optional[float]]]:
    """sha256 -> {'jpeg_quality','blur_sigma','resize_factor'} (all None for clean images)."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        recs = json.load(f)["records"]
    keys = ("jpeg_quality", "blur_sigma", "resize_factor")
    return {r["sha256"]: {k: (r.get("degradation") or {}).get(k) for k in keys} for r in recs}


def _bin_labels(edges: Sequence[float], fmt: str) -> List[str]:
    return [f"{fmt.format(a)}-{fmt.format(b)}" for a, b in zip(edges[:-1], edges[1:])]


def _bin_index(v: float, edges: Sequence[float]) -> Optional[int]:
    if v < edges[0] or v > edges[-1]:
        return None
    return min(int(np.searchsorted(edges, v, side="right")) - 1, len(edges) - 2)


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def _row(hv, split, axis, label, y, w, p) -> Dict[str, Any]:
    r: Dict[str, Any] = {"head_variant": hv, "split": split, "axis": axis, "bin": label, "n": int(len(y))}
    for k, e in enumerate(EXPERT_NAMES):
        r[f"w_{e}_mean"] = float(w[:, k].mean()) if len(y) else float("nan")
        r[f"w_{e}_sd"] = float(w[:, k].std()) if len(y) else float("nan")
        r[f"auroc_{e}"] = _auc(y, p[:, k]) if len(y) else float("nan")
    r["auroc_aegis"] = _auc(y, (w * p).sum(1)) if len(y) else float("nan")
    return r


def verify_curves(y: np.ndarray, w: np.ndarray, p: np.ndarray, shas: Sequence[str],
                  params: Dict[str, Dict[str, Optional[float]]], hv: Optional[str]) -> List[Dict[str, Any]]:
    """Rows per JPEG-Q bin, blur-sigma bin and 'clean' for arrays (y, w (n,3), p (n,3)) aligned with `shas`."""
    prm = [params.get(s, {}) for s in shas]
    q = np.array([d.get("jpeg_quality") if d.get("jpeg_quality") is not None else np.nan for d in prm], dtype=float)
    sg = np.array([d.get("blur_sigma") if d.get("blur_sigma") is not None else np.nan for d in prm], dtype=float)
    rs = np.array([d.get("resize_factor") is not None for d in prm])
    clean = np.isnan(q) & np.isnan(sg) & ~rs
    rows: List[Dict[str, Any]] = []
    for axis, vals, other, edges, fmt in (("jpeg", q, sg, JPEG_EDGES, "{:g}"), ("blur", sg, q, BLUR_EDGES, "{:.1f}")):
        pure = ~np.isnan(vals) & np.isnan(other) & ~rs                  # single-degradation samples only
        idx = np.array([(_bin_index(v, edges) if m else None) for v, m in zip(vals, pure)], dtype=object)
        idx = np.array([-1 if i is None else i for i in idx], dtype=int)
        for i, lab in enumerate(_bin_labels(edges, fmt)):
            m = idx == i
            rows.append(_row(hv, "val_gating_verify", axis, lab, y[m], w[m], p[m]))
        rows.append(_row(hv, "val_gating_verify", axis, "clean", y[clean], w[clean], p[clean]))
    return rows


def test_level_rows(groups: Dict[str, Dict[str, np.ndarray]], hv: Optional[str]) -> List[Dict[str, Any]]:
    """Per fixed test level (variant 'full'), from saved predictions (aegis weights + calibrated expert p)."""
    rows = []
    for lvl in TEST_LEVELS:
        for k, g in groups.items():
            d = parse_key(k)
            ok_sc = d["scenario"] == "S1" if lvl == "clean" else d["scenario"] in ("S2", "S3")
            if d["variant"] != "full" or d["level"] != lvl or not ok_sc or "w_aegis" not in g:
                continue
            p = np.stack([g[f"p_{e}"] for e in EXPERT_NAMES], 1)
            rows.append(_row(hv, "test_benchmark", "level", lvl, g["y"], g["w_aegis"], p))
            break
    return rows


# --------------------------------------------------------------------------------------------------- output
def plot_axis(rows: List[Dict[str, Any]], axis: str, path: str | Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hvs = list(dict.fromkeys(r["head_variant"] for r in rows))
    fig, axes = plt.subplots(2, len(hvs), figsize=(5.2 * len(hvs), 7), squeeze=False, sharex="col")
    colors = {"clip": "tab:blue", "dinov2": "tab:green", "freq": "tab:red"}
    for j, hv in enumerate(hvs):
        sub = [r for r in rows if r["head_variant"] == hv and r["axis"] == axis and r["split"] == "val_gating_verify"]
        labels = [r["bin"] for r in sub]
        x = np.arange(len(sub))
        for e in EXPERT_NAMES:
            mu = np.array([r[f"w_{e}_mean"] for r in sub])
            sd = np.array([r[f"w_{e}_sd"] for r in sub])
            axes[0, j].plot(x, mu, marker="o", color=colors.get(e), label=f"w_{e}")
            axes[0, j].fill_between(x, mu - sd, mu + sd, color=colors.get(e), alpha=0.12)
            axes[1, j].plot(x, [r[f"auroc_{e}"] for r in sub], marker="s", color=colors.get(e), label=f"AUROC {e}")
        axes[1, j].plot(x, [r["auroc_aegis"] for r in sub], marker="^", color="k", ls="--", label="AUROC AEGIS")
        axes[0, j].set_title(f"head variant = {hv}")
        axes[0, j].set_ylabel("mean gating weight (band = +-1 sd)")
        axes[1, j].set_ylabel("AUROC on val_gating_verify bin")
        axes[1, j].set_xticks(x)
        axes[1, j].set_xticklabels(labels, rotation=35, ha="right")
        axes[1, j].set_xlabel("JPEG quality Q bin" if axis == "jpeg" else "Gaussian blur sigma bin")
        for a in axes[:, j]:
            a.grid(alpha=0.3)
            a.legend(fontsize=8)
    fig.suptitle(f"AEGIS gating weights vs {'JPEG quality' if axis == 'jpeg' else 'blur sigma'} (val_gating_verify)")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return Path(path)


def _md(rows: List[Dict[str, Any]]) -> List[str]:
    lines = ["# AEGIS gating weights by degradation level", "",
             "Weights: mean +- sd of w over the samples of the bin/level (variant full). AUROC per expert: calibrated "
             "probability of that expert alone in the same bin/level. val_gating_verify is NOT used to fit the gating; "
             "test_benchmark numbers come from the saved predictions of the single test evaluation.", ""]
    for hv in dict.fromkeys(r["head_variant"] for r in rows):
        for split, axis in (("val_gating_verify", "jpeg"), ("val_gating_verify", "blur"), ("test_benchmark", "level")):
            sub = [r for r in rows if r["head_variant"] == hv and r["split"] == split and r["axis"] == axis]
            if not sub:
                continue
            lines += [f"## head variant = {hv}, {split}, by {axis}", "",
                      "| bin | n | " + " | ".join(f"w_{e}" for e in EXPERT_NAMES) + " | "
                      + " | ".join(f"AUROC {e}" for e in EXPERT_NAMES) + " | AUROC AEGIS |",
                      "|" + "---|" * (3 + 2 * len(EXPERT_NAMES) + 1)]
            for r in sub:
                ws = " | ".join(f"{r[f'w_{e}_mean']:.3f} +- {r[f'w_{e}_sd']:.3f}" for e in EXPERT_NAMES)
                au = " | ".join(f"{r[f'auroc_{e}']:.3f}" for e in EXPERT_NAMES)
                lines.append(f"| {r['bin']} | {r['n']} | {ws} | {au} | {r['auroc_aegis']:.3f} |")
            lines.append("")
    return lines


def write_outputs(rows: List[Dict[str, Any]], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "weights_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    (out / "weights_summary.md").write_text("\n".join(_md(rows)) + "\n", encoding="utf-8")
    plot_axis(rows, "jpeg", out / "weights_by_jpeg.png")
    plot_axis(rows, "blur", out / "weights_by_blur.png")
    log.info("weights analysis written to %s", out)


def analyse(cfg: Config, hv: Optional[str], fit: Bundle, verify: Bundle,
            params: Dict[str, Dict[str, Optional[float]]], test_groups: Optional[Dict[str, Dict[str, np.ndarray]]]
            ) -> List[Dict[str, Any]]:
    """Fit the full AEGIS system on `fit` (val_gating_fit) exactly like the ablation, describe it on `verify`."""
    from src.evaluation.ablation import _System
    sysm = _System(cfg, fit, True, True, cfg.get("seed", 0))
    fs = sysm.pipe.transform(verify)
    rows = verify_curves(verify.label, sysm.gating.weights(fs), fs.p, verify.sha256, params, hv)
    if test_groups is not None:
        rows += test_level_rows(test_groups, hv)
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    from src.core.config import REPO_ROOT, load_config
    from src.learning.bundle_io import load_split
    ap = argparse.ArgumentParser(prog="python -m src.evaluation.weights_analysis")
    ap.add_argument("--pred", default=None, help="predictions dir from ablation --save-predictions (test table)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=str(REPO_ROOT / "data" / "normalized_v1" / "manifest.json"))
    ap.add_argument("--head-variants", nargs="+", default=["clean", "aug"])
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    params = load_degradation_params(a.manifest)
    rows: List[Dict[str, Any]] = []
    for hv in a.head_variants:
        tg = None
        if a.pred and (Path(a.pred) / f"predictions_{hv}.npz").exists():
            tg = load_predictions(Path(a.pred) / f"predictions_{hv}.npz")
        rows += analyse(cfg, hv, load_split(cfg, "val_gating_fit", hv), load_split(cfg, "val_gating_verify", hv),
                        params, tg)
    write_outputs(rows, a.out)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
