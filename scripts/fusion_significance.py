"""Paired bootstrap: AEGIS gating vs every classic-ML meta-classifier trained on the same 12-d gating input.

`src.evaluation.ml_comparison.run_fusion_comparison` reports point metrics only. This script repeats exactly its protocol
(fit on val_gating_fit, hyper-parameters on val_gating_verify, test_benchmark evaluated once) but keeps the per-sample
predictions, so differences can be tested with the same source-clustered paired bootstrap used elsewhere
(`src.evaluation.significance`). It does not change any model or choice; it only adds confidence intervals.

Usage (from the repo root):
    .venv/Scripts/python scripts/fusion_significance.py --out reports/ml_comparison
Outputs: fusion_significance.csv (delta AUROC, CI, p for every stack_* model vs AEGIS) and a short summary in the log.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # same convention as the other scripts
from src.core.config import load_config, resolve_path  # noqa: E402
from src.core.logger import get_logger  # noqa: E402
from src.evaluation.ablation import _slices, _System  # noqa: E402
from src.evaluation.ml_comparison import DEFAULT_MODELS, REFERENCE  # noqa: E402
from src.evaluation.significance import (DEFAULT_B, _cluster_ids, bootstrap_metrics, load_parent_map,  # noqa: E402
                                         point_metrics, summarize_delta)
from src.learning.bundle_io import bundle_path, load_split as load_bundle  # noqa: E402
from src.learning.classic_models import SklearnStacker, tune_stacker  # noqa: E402

log = get_logger("fusion_significance")


def collect(cfg: dict, head_variant: str, seed: int) -> Dict[tuple, dict]:
    """{(scenario, level): {'y', 'shas', 'preds': {method: p}}} for AEGIS, baselines and every stack_* model."""
    need = ["val_gating_fit", "test_benchmark"]
    opt = [s for s in ("val_gating_verify",) if bundle_path(cfg, s, head_variant).exists()]
    bundles = {s: load_bundle(cfg, s, head_variant) for s in need + opt}
    sysm = _System(cfg, bundles["val_gating_fit"], True, True, seed)
    ver = bundles.get("val_gating_verify")
    ver_fs = sysm.pipe.transform(ver) if ver is not None else None
    models = {k: sysm.models[k] for k in ("naive_avg", "static_lr", "aegis")}
    for m in DEFAULT_MODELS:
        if m == REFERENCE:
            continue
        if ver_fs is not None:
            st, _params, _ = tune_stacker(m, sysm.fit_fs, sysm.fit_y, ver_fs, ver.label, seed, cfg)
        else:
            st = SklearnStacker(m, seed, cfg).fit(sysm.fit_fs, sysm.fit_y)
        models[f"stack_{m}"] = st
    out: Dict[tuple, dict] = {}
    for sc, lvl, b in _slices(bundles["test_benchmark"]):
        fs = sysm.pipe.transform(b)
        out[(sc, lvl)] = {"y": np.asarray(b.label).astype(int), "shas": list(b.sha256),
                          "preds": {name: np.asarray(mdl.predict(fs), dtype=float) for name, mdl in models.items()}}
    return out


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="reports/ml_comparison")
    ap.add_argument("--head-variants", nargs="+", default=["clean", "aug"])
    ap.add_argument("--B", type=int, default=DEFAULT_B)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args(argv)

    cfg = load_config()
    seed = int(cfg.get("seed", 0) if a.seed is None else a.seed)
    parent = load_parent_map(resolve_path(cfg, "manifest"))
    rows: List[dict] = []
    for hv in a.head_variants:
        log.info("head variant %s: fitting AEGIS and the stacking models", hv)
        for (sc, lvl), d in collect(cfg, hv, seed).items():
            if lvl not in ("clean", "all"):
                continue
            clusters = _cluster_ids(d["shas"], parent)
            boot = bootstrap_metrics(d["y"], d["preds"], B=a.B, seed=seed, clusters=clusters)
            pt = {k: point_metrics(d["y"], p) for k, p in d["preds"].items()}
            for name in d["preds"]:
                if name == "aegis":
                    continue
                delta_b = boot["aegis"]["auroc"] - boot[name]["auroc"]
                r = summarize_delta(delta_b, pt["aegis"]["auroc"] - pt[name]["auroc"])
                rows.append({"head_variant": hv, "scenario": sc, "level": lvl, "baseline": name,
                             "aegis_auroc": pt["aegis"]["auroc"], "baseline_auroc": pt[name]["auroc"], "n": len(d["y"]),
                             "n_clusters": int(len(np.unique(clusters))) if clusters is not None else len(d["y"]), **r})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "fusion_significance.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    log.info("wrote %s (%d rows)", path, len(rows))
    for r in rows:
        if r["baseline"] in ("stack_random_forest", "stack_hist_gb", "static_lr", "naive_avg"):
            log.info("%-5s %s %-5s AEGIS - %-20s dAUROC %+.4f [%+.4f, %+.4f] p=%.3f %s", r["head_variant"], r["scenario"],
                     r["level"], r["baseline"], r["delta"], r["ci_lo"], r["ci_hi"], r["p_value"],
                     "*" if r["significant"] else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
