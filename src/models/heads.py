"""Linear heads for frozen CLIP / DINOv2 features + artifact persistence + training CLI.

Variants: `clean` = clean train_experts only; `aug` = clean + train_aug variants of train_experts.
The OOD centroid is always computed on CLEAN train_experts features (same meaning for both variants).
Artifacts (under cfg.paths.cache_dir): heads/<expert>_<variant>_<ver>.pt, heads/freq_<input>_<variant>_<ver>.pt,
centroids_<version>.pt (keys clip, dinov2, freq_<input>_<variant>).
CLI: python -m src.models.heads --manifest M --expert clip|dinov2|freq [--variant clean|aug] [--freq-input ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from src.core.config import load_config, resolve_path
from src.core.contracts import SCENARIO_TRAIN_AUG, SPLIT_TRAIN, SPLIT_VAL_FIT, SplitManifest
from src.core.logger import get_logger

log = get_logger(__name__)

HEAD_VARIANTS = ("clean", "aug")
FREQ_INPUTS = ("log_magnitude_fft", "srm")


class LinearHead(nn.Module):
    """Feature standardisation (stats stored from the training set) + Linear(d, 1) -> logit."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))
        self.fc = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc((x - self.mean) / self.std).squeeze(-1)


def train_linear_head(features: torch.Tensor, labels: torch.Tensor, cfg: dict,
                      device: str | torch.device = "cpu") -> LinearHead:
    hc = cfg["experts"]["head"]
    torch.manual_seed(int(cfg.get("seed", 0)))
    x, y = features.float().to(device), labels.float().to(device)
    head = LinearHead(x.shape[1]).to(device)
    head.mean.copy_(x.mean(0))
    head.std.copy_(x.std(0).clamp_min(1e-6))
    opt = torch.optim.Adam(head.parameters(), lr=hc["lr"], weight_decay=hc["weight_decay"])
    n, bs = len(x), int(hc["batch_size"])
    for _ in range(int(hc["epochs"])):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = F.binary_cross_entropy_with_logits(head(x[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return head.cpu().eval()


def check_variant(variant: str) -> str:
    if variant not in HEAD_VARIANTS:
        raise ValueError(f"head variant must be one of {HEAD_VARIANTS}, got '{variant}'")
    return variant


def with_freq_input(cfg: dict, freq_input: Optional[str]) -> dict:
    """Shallow copy of cfg with experts.freq.input overridden (cfg itself is not modified)."""
    if not freq_input:
        return cfg
    if freq_input not in FREQ_INPUTS:
        raise ValueError(f"unknown freq input '{freq_input}'")
    return {**cfg, "experts": {**cfg["experts"], "freq": {**cfg["experts"]["freq"], "input": freq_input}}}


def head_path(cfg: dict, expert: str, variant: str = "clean", freq_input: Optional[str] = None) -> Path:
    """heads/<expert>_<variant>_<ver>.pt ; freq: heads/freq_<input>_<variant>_<ver>.pt (never overwrite)."""
    check_variant(variant)
    v = cfg["experts"]["extract"]["cache_version"]
    if expert == "freq":
        name = f"freq_{freq_input or cfg['experts']['freq']['input']}_{variant}_{v}.pt"
    else:
        name = f"{expert}_{variant}_{v}.pt"
    return resolve_path(cfg, "cache_dir") / "heads" / name


def centroid_key(expert: str, freq_input: Optional[str] = None, variant: str = "clean") -> str:
    """Key in the centroid file. clip/dinov2 centroids do not depend on the variant (frozen backbone, clean
    images); the freq centroid depends on the trained CNN, hence on (input, variant)."""
    if expert == "freq":
        return f"freq_{freq_input}_{variant}"
    return expert


def train_records(man: SplitManifest, variant: str = "clean"):
    """train_experts records for head training: `clean` = clean only; `aug` = clean + train_aug variants
    (never val/test variants, never other degradation scenarios)."""
    check_variant(variant)
    recs = man.by_split(SPLIT_TRAIN)
    if variant == "clean":
        return [r for r in recs if r.is_clean]
    return [r for r in recs if r.is_clean or (r.degradation and r.degradation.scenario == SCENARIO_TRAIN_AUG)]


def save_head(cfg: dict, expert: str, state: dict, dim: int, model_id: str = "",
              variant: str = "clean", freq_input: Optional[str] = None) -> Path:
    p = head_path(cfg, expert, variant, freq_input)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": state, "dim": dim, "model_id": model_id, "variant": variant,
                "freq_input": (freq_input or cfg["experts"]["freq"]["input"]) if expert == "freq" else None}, p)
    return p


def load_head_into(cfg: dict, expert, variant: str = "clean") -> None:
    """Load the saved head (clip/dinov2) or CNN weights (freq; input = expert.input_mode) into `expert`."""
    fi = getattr(expert, "input_mode", None) if expert.name == "freq" else None
    blob = torch.load(head_path(cfg, expert.name, variant, fi), map_location="cpu", weights_only=True)
    if expert.name == "freq":
        expert.net.load_state_dict(blob["state"])
        expert.net.eval()
    else:
        head = LinearHead(blob["dim"])
        head.load_state_dict(blob["state"])
        expert.head = head.eval().to(expert.device)
        expert.feature_dim = blob["dim"]
    expert.model_id = blob.get("model_id") or expert.model_id


def train_expert_from_manifest(man: SplitManifest, expert_name: str, cfg: dict, device: str = "cpu",
                               variant: str = "clean", freq_input: Optional[str] = None) -> Path:
    """Train one expert head on `train_records(man, variant)`. The OOD centroid is ALWAYS the mean feature of
    CLEAN train_experts images only, so cosine distance has the same meaning for both variants."""
    from src.models import build_expert
    from src.models.cache import FeatureCache
    from src.models.freq_expert import ImagePathDataset, train_freq
    from src.ood.profiler import compute_centroid, update_centroid_file

    recs = train_records(man, variant)
    clean = [r for r in recs if r.is_clean]
    if expert_name == "freq":
        freq_input = freq_input or cfg["experts"]["freq"]["input"]
        cfg = with_freq_input(cfg, freq_input)
    expert = build_expert(expert_name, cfg).to(device)
    if expert_name == "freq":
        ds = ImagePathDataset([str(man.abs_path(r)) for r in recs], [r.label for r in recs])
        train_freq(expert, ds, cfg, device)
        cds = ImagePathDataset([str(man.abs_path(r)) for r in clean], [r.label for r in clean])
        feats = torch.cat([expert.extract_features(cds[i][0][None]) for i in range(len(cds))])
        out = save_head(cfg, "freq", expert.net.state_dict(), expert.feature_dim, expert.model_id,
                        variant, freq_input)
    else:
        cache = FeatureCache.open(cfg, expert_name)
        feats_all = cache.features_for([r.sha256 for r in recs])
        head = train_linear_head(feats_all, torch.tensor([r.label for r in recs]), cfg, device)
        feats = cache.features_for([r.sha256 for r in clean])
        out = save_head(cfg, expert_name, head.state_dict(), feats.shape[1], cache.meta.get("model_id", ""),
                        variant)
    update_centroid_file(cfg, centroid_key(expert_name, freq_input, variant), compute_centroid(feats))
    log.info("trained %s[%s%s] on %d samples (%d clean) -> %s", expert_name, variant,
             f",{freq_input}" if freq_input else "", len(recs), len(clean), out)
    return out


def select_freq_input(man: SplitManifest, cfg: dict, head_variant: str = "clean", device: str = "cpu",
                      experts: Optional[dict] = None, inputs: Sequence[str] = FREQ_INPUTS,
                      out_path: Optional[Path] = None) -> dict:
    """AUROC of each Expert-3 input on CLEAN val_gating_fit; pick the best, write
    <cache_dir>/freq_selection.json (config is NOT modified). `experts` {input: FreqExpert} may be passed
    (already loaded); otherwise the checkpoints of `head_variant` are loaded."""
    from sklearn.metrics import roc_auc_score

    from src.models import build_expert
    from src.models.base import load_image_array

    recs = [r for r in man.by_split(SPLIT_VAL_FIT) if r.is_clean]
    y = [r.label for r in recs]
    aur = {}
    for inp in inputs:
        if experts and inp in experts:
            e = experts[inp]
        else:
            e = build_expert("freq", with_freq_input(cfg, inp)).to(device)
            load_head_into(cfg, e, head_variant)
        sc = []
        for r in recs:
            _, lg = e.features_and_logits(e.preprocess([load_image_array(str(man.abs_path(r)))]))
            sc.append(float(lg[0]))
        aur[inp] = float(roc_auc_score(y, sc)) if len(set(y)) == 2 else float("nan")
    valid = {k: v for k, v in aur.items() if v == v}
    best = max(valid, key=valid.get) if valid else cfg["experts"]["freq"]["input"]
    path = Path(out_path) if out_path else resolve_path(cfg, "cache_dir") / "freq_selection.json"
    blob = json.loads(path.read_text()) if path.exists() else {}
    blob.setdefault("auroc", {})[head_variant] = aur
    blob.setdefault("selected", {})[head_variant] = best
    blob["split"] = "val_gating_fit (clean)"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(blob, indent=2))
    log.info("freq input selection [%s]: %s -> %s", head_variant, aur, best)
    return {"auroc": aur, "selected": best, "path": str(path)}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--expert", required=True, choices=["clip", "dinov2", "freq"])
    ap.add_argument("--variant", default="clean", choices=list(HEAD_VARIANTS))
    ap.add_argument("--freq-input", default=None, choices=list(FREQ_INPUTS),
                    help="expert freq only; default = cfg experts.freq.input")
    ap.add_argument("--select-freq", action="store_true",
                    help="instead of training: compare freq inputs on val_gating_fit and write freq_selection.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    cfg = load_config()
    man = SplitManifest.load(a.manifest)
    if a.select_freq:
        select_freq_input(man, cfg, a.variant, a.device)
    else:
        train_expert_from_manifest(man, a.expert, cfg, a.device, a.variant, a.freq_input)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
