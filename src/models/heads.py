"""Linear heads for frozen CLIP / DINOv2 features + artifact persistence + training CLI.

Heads are trained on cached embeddings of CLEAN train_experts only (never degraded variants).
Artifacts (under cfg.paths.cache_dir):  heads/<expert>_<version>.pt ,  centroids_<version>.pt
CLI: python -m src.models.heads --manifest <path> --expert clip|dinov2|freq
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from src.core.config import load_config, resolve_path
from src.core.contracts import SPLIT_TRAIN, SplitManifest
from src.core.logger import get_logger

log = get_logger(__name__)


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


def head_path(cfg: dict, expert: str) -> Path:
    return resolve_path(cfg, "cache_dir") / "heads" / f"{expert}_{cfg['experts']['extract']['cache_version']}.pt"


def save_head(cfg: dict, expert: str, state: dict, dim: int, model_id: str = "") -> Path:
    p = head_path(cfg, expert)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": state, "dim": dim, "model_id": model_id}, p)
    return p


def load_head_into(cfg: dict, expert) -> None:
    """Load the saved head (clip/dinov2) or CNN weights (freq) into `expert`."""
    blob = torch.load(head_path(cfg, expert.name), map_location="cpu", weights_only=True)
    if expert.name == "freq":
        expert.net.load_state_dict(blob["state"])
        expert.net.eval()
    else:
        head = LinearHead(blob["dim"])
        head.load_state_dict(blob["state"])
        expert.head = head.eval().to(expert.device)
        expert.feature_dim = blob["dim"]
    expert.model_id = blob.get("model_id") or expert.model_id


def train_expert_from_manifest(man: SplitManifest, expert_name: str, cfg: dict, device: str = "cpu") -> Path:
    from src.models import build_expert
    from src.models.cache import FeatureCache
    from src.models.freq_expert import ImagePathDataset, train_freq
    from src.ood.profiler import compute_centroid, update_centroid_file

    recs = man.by_split(SPLIT_TRAIN, clean_only=True)
    labels = torch.tensor([r.label for r in recs])
    expert = build_expert(expert_name, cfg).to(device)
    if expert_name == "freq":
        ds = ImagePathDataset([str(man.abs_path(r)) for r in recs], [r.label for r in recs])
        train_freq(expert, ds, cfg, device)
        feats = torch.cat([expert.extract_features(ds[i][0][None]) for i in range(len(ds))])
        out = save_head(cfg, "freq", expert.net.state_dict(), expert.feature_dim, expert.model_id)
    else:
        cache = FeatureCache.open(cfg, expert_name)
        feats = cache.features_for([r.sha256 for r in recs])
        head = train_linear_head(feats, labels, cfg, device)
        out = save_head(cfg, expert_name, head.state_dict(), feats.shape[1], cache.meta.get("model_id", ""))
    update_centroid_file(cfg, expert_name, compute_centroid(feats))
    log.info("trained %s on %d clean train_experts samples -> %s", expert_name, len(recs), out)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--expert", required=True, choices=["clip", "dinov2", "freq"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    cfg = load_config()
    train_expert_from_manifest(SplitManifest.load(a.manifest), a.expert, cfg, a.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
