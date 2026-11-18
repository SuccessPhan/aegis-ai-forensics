"""Choose gating.fusion ("prob" | "logit") using ONLY val_gating_fit (fit) and val_gating_verify (decision).

    python -m src.evaluation.select_fusion [--head-variants clean aug] [--out reports/fusion_selection]

For each fusion the whole system (temperatures, scaler, gating MLP, static LR) is fitted on val_gating_fit; AEGIS and
the matching Static Learned Weights are scored on val_gating_verify, split into clean / JPEG / blur / overall.
Paired bootstrap of delta = metric(logit) - metric(prob) (cluster by source image through the manifest when available).

Rule: select "logit" only if overall delta AUROC (AEGIS) > 0 with a 95% CI that excludes 0 AND
ECE(logit) - ECE(prob) <= evaluation.champion_challenger.max_ece_increase; otherwise "prob". With several head
variants the global choice is "logit" only if every variant selects "logit".

SAFETY: test_benchmark (and the S4 pools) are never loaded here. `load_allowed` refuses any other split and
`assert_not_test_bundle` rejects a bundle that carries test_benchmark metadata / S2-S3 scenario tags.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.core.config import Config, deep_merge
from src.core.contracts import SPLIT_TEST, SPLIT_VAL_FIT, SPLIT_VAL_VERIFY
from src.core.logger import get_logger
from src.evaluation.metrics import expected_calibration_error
from src.evaluation.significance import (DEFAULT_B, DEFAULT_SEED, bootstrap_metrics, load_parent_map, point_metrics,
                                         summarize_delta)
from src.learning.bundle_io import Bundle

log = get_logger(__name__)

ALLOWED_SPLITS = (SPLIT_VAL_FIT, SPLIT_VAL_VERIFY)
FUSIONS = ("prob", "logit")
SLICES = ("clean", "jpeg", "blur", "overall")
MODELS = ("aegis", "static_lr")


class ForbiddenSplitError(RuntimeError):
    """Raised when fusion selection would touch test_benchmark (or any split other than the two val_gating parts)."""


def assert_not_test_bundle(b: Bundle, name: str = "bundle") -> None:
    if b.meta.get("split") == SPLIT_TEST or any(s in ("S2", "S3") for s in b.scenario):
        raise ForbiddenSplitError(f"{name} looks like test_benchmark; fusion selection must not read it")


def load_allowed(cfg: Config, split: str, head_variant: Optional[str] = None) -> Bundle:
    if split not in ALLOWED_SPLITS:
        raise ForbiddenSplitError(f"fusion selection may only load {ALLOWED_SPLITS}, got {split!r}")
    from src.learning.bundle_io import load_split
    b = load_split(cfg, split, head_variant)
    assert_not_test_bundle(b, split)
    return b


def slice_masks(b: Bundle) -> Dict[str, np.ndarray]:
    kind = np.array([("clean" if d is None else str(d).lower().split("_")[0]) for d in b.degradation])
    return {"clean": kind == "clean", "jpeg": kind == "jpeg", "blur": kind == "blur",
            "overall": np.ones(len(b), dtype=bool)}


def _fit_predict(cfg: Config, fusion: str, fit: Bundle, verify: Bundle) -> Dict[str, np.ndarray]:
    """Fit everything on `fit` under `fusion`; return {model: p on verify}."""
    from src.evaluation.ablation import _System
    c = deep_merge(cfg, {"gating": {"fusion": fusion}})
    sysm = _System(c, fit, True, True, c.get("seed", 0))
    fs = sysm.pipe.transform(verify)
    return {"aegis": np.asarray(sysm.gating.predict(fs)), "static_lr": np.asarray(sysm.models["static_lr"].predict(fs))}


def decide(overall_aegis: Dict[str, Any], ece_increase: float, max_ece_increase: float) -> Dict[str, Any]:
    """Pure selection rule (see module docstring)."""
    ok_auc = overall_aegis["delta"] > 0 and overall_aegis["ci_lo"] > 0
    ok_ece = ece_increase <= max_ece_increase
    choice = "logit" if ok_auc and ok_ece else "prob"
    if choice == "logit":
        why = "logit: overall delta AUROC > 0 with CI excluding 0 and ECE increase within limit"
    elif not ok_auc:
        why = "prob: overall delta AUROC (logit - prob) is not > 0 with a CI excluding 0"
    else:
        why = f"prob: ECE increase {ece_increase:.4f} > max_ece_increase {max_ece_increase}"
    return {"selected": choice, "auroc_criterion": bool(ok_auc), "ece_criterion": bool(ok_ece), "reason": why}


def select_for_variant(cfg: Config, fit: Bundle, verify: Bundle, parent: Optional[Dict[str, str]] = None,
                       B: Optional[int] = None, seed: Optional[int] = None,
                       head_variant: Optional[str] = None) -> Dict[str, Any]:
    assert_not_test_bundle(fit, "fit")
    assert_not_test_bundle(verify, "verify")
    bc = (cfg.get("evaluation") or {}).get("bootstrap") or {}
    B = int(B or bc.get("n_resamples", DEFAULT_B))
    seed = int(bc.get("seed", DEFAULT_SEED) if seed is None else seed)
    max_ece = float(cfg["evaluation"]["champion_challenger"]["max_ece_increase"])
    preds = {f: _fit_predict(cfg, f, fit, verify) for f in FUSIONS}
    y = np.asarray(verify.label)
    shas = verify.sha256
    clusters = None
    if parent:
        clusters = np.unique(np.array([parent.get(s, s) for s in shas]), return_inverse=True)[1]
    rows: List[Dict[str, Any]] = []
    overall_ece: Dict[str, Dict[str, float]] = {}
    for sname, m in slice_masks(verify).items():
        if m.sum() < 2 or len(np.unique(y[m])) < 2:
            log.warning("fusion selection: slice %s has too few samples/classes (n=%d); skipped", sname, int(m.sum()))
            continue
        cl = None if clusters is None else np.unique(clusters[m], return_inverse=True)[1]
        for model in MODELS:
            pp = {f: preds[f][model][m] for f in FUSIONS}
            boot = bootstrap_metrics(y[m], pp, B, seed, cl)
            pt = {f: point_metrics(y[m], pp[f]) for f in FUSIONS}
            ece = {f: expected_calibration_error(y[m], pp[f], cfg["evaluation"]["ece_bins"]) for f in FUSIONS}
            for met in ("auroc", "accuracy"):
                res = summarize_delta(boot["logit"][met] - boot["prob"][met], pt["logit"][met] - pt["prob"][met])
                rows.append({"head_variant": head_variant, "slice": sname, "model": model, "metric": met,
                             "prob": pt["prob"][met], "logit": pt["logit"][met], **res, "n": int(m.sum()),
                             "ece_prob": ece["prob"], "ece_logit": ece["logit"], "B": B})
            if sname == "overall":
                overall_ece[model] = ece
    ov = [r for r in rows if r["slice"] == "overall" and r["model"] == "aegis" and r["metric"] == "auroc"]
    if not ov:
        raise ValueError("verify bundle needs both classes for the overall slice")
    e = overall_ece["aegis"]
    dec = decide(ov[0], e["logit"] - e["prob"], max_ece)
    return {"head_variant": head_variant, "rows": rows, "ece_increase": e["logit"] - e["prob"],
            "max_ece_increase": max_ece, "clustered": clusters is not None, **dec}


def _md(result: Dict[str, Any]) -> List[str]:
    lines = ["# Fusion selection (val_gating_fit -> val_gating_verify; test_benchmark never read)", "",
             f"**Selected fusion: `{result['selected']}`**", "",
             "Rule: logit only if overall delta AUROC (AEGIS, logit - prob) > 0 with 95% CI excluding 0 and ECE "
             f"increase <= {result['max_ece_increase']}.", ""]
    for v in result["variants"]:
        lines += [f"## head variant = {v['head_variant']}: `{v['selected']}`", "", v["reason"], "",
                  f"ECE increase (overall AEGIS) = {v['ece_increase']:+.4f}; cluster bootstrap = {v['clustered']}", "",
                  "| slice | model | metric | prob | logit | delta | CI95 | p | n | ECE prob | ECE logit |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in v["rows"]:
            lines.append(f"| {r['slice']} | {r['model']} | {r['metric']} | {r['prob']:.4f} | {r['logit']:.4f} | "
                         f"{r['delta']:+.4f} | [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['p_value']:.3f} | {r['n']} | "
                         f"{r['ece_prob']:.4f} | {r['ece_logit']:.4f} |")
        lines.append("")
    return lines


def run(cfg: Config, head_variants: Optional[Sequence[Optional[str]]] = None,
        out_dir: str | Path = "reports/fusion_selection", manifest: Optional[str | Path] = None,
        B: Optional[int] = None, loader=load_allowed) -> Dict[str, Any]:
    from src.core.config import resolve_path
    hvs = list(head_variants) if head_variants else [None]
    mp = Path(manifest) if manifest else resolve_path(cfg, "manifest")
    parent = load_parent_map(mp) if mp.exists() else None
    if parent is None:
        log.warning("manifest %s not found: iid bootstrap (no clustering by source image)", mp)
    variants = []
    for hv in hvs:
        variants.append(select_for_variant(cfg, loader(cfg, SPLIT_VAL_FIT, hv), loader(cfg, SPLIT_VAL_VERIFY, hv),
                                           parent, B, head_variant=hv))
    selected = "logit" if all(v["selected"] == "logit" for v in variants) else "prob"
    result = {"selected": selected, "max_ece_increase": variants[0]["max_ece_increase"],
              "splits_used": list(ALLOWED_SPLITS), "variants": variants}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fusion_selection.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    (out / "fusion_selection.md").write_text("\n".join(_md(result)) + "\n", encoding="utf-8")
    log.info("fusion selection -> %s (selected=%s)", out, selected)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    from src.core.config import REPO_ROOT, load_config
    ap = argparse.ArgumentParser(prog="python -m src.evaluation.select_fusion")
    ap.add_argument("--head-variants", nargs="+", default=None)
    ap.add_argument("--out", default=str(REPO_ROOT / "reports" / "fusion_selection"))
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--B", type=int, default=None)
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)
    run(load_config(a.config), a.head_variants, a.out, a.manifest, a.B)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
