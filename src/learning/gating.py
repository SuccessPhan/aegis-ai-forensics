"""Adaptive Gating Controller: Linear(12,32)-ReLU-Dropout-Linear(32,3); w = w_min + scale*softmax(z)."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch import nn

from src.core.config import Config
from src.core.logger import get_logger
from src.learning.features import FeatureSet
from src.learning.fusion import calibrated_logits, fuse, get_fusion

log = get_logger(__name__)


class GatingNet(nn.Module):
    def __init__(self, in_dim: int = 12, hidden: int = 32, n_experts: int = 3, dropout: float = 0.2,
                 w_min: float = 0.05, scale: float = 0.85):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, n_experts))
        self.w_min, self.scale = w_min, scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_min + self.scale * torch.softmax(self.net(x), dim=-1)


class GatingController:
    """Per-sample weights w(x); P_final = fuse(w, experts) (gating.fusion: prob = sum w_i P_i; logit = sigmoid(sum w_i l_i)). Trained on val_gating_fit (see learning.fewshot for refits).

    Loss = BCE(P_final, y) - lambda * mean_x H(w(x)); the entropy bonus discourages collapsing onto one expert.
    """

    def __init__(self, cfg: Config, seed: Optional[int] = None, fusion: Optional[str] = None):
        g = cfg["gating"]
        self.cfg, self.g, self.tr = cfg, g, cfg["gating_train"]
        self.fusion = fusion or get_fusion(cfg)
        self.seed = cfg.get("seed", 0) if seed is None else seed
        torch.manual_seed(self.seed)
        self.net = GatingNet(g["input_dim"], g["hidden_dim"], g["num_experts"], g["dropout"], g["w_min"], g["scale"])
        self.net.eval()
        self.history: List[float] = []

    def fit(self, fs: FeatureSet, y: np.ndarray, epochs: Optional[int] = None) -> "GatingController":
        X = torch.as_tensor(fs.X, dtype=torch.float32)
        P = torch.as_tensor(fs.p, dtype=torch.float32)
        Z = (torch.as_tensor(calibrated_logits(fs.p, fs.z), dtype=torch.float32)
             if self.fusion == "logit" else None)
        Y = torch.as_tensor(y, dtype=torch.float32)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.tr["lr"], weight_decay=self.tr["weight_decay"])
        lam, bs, n = self.g["entropy_reg_lambda"], self.tr["batch_size"], len(X)
        gen = torch.Generator().manual_seed(self.seed)
        self.net.train()
        self.history = []
        for _ in range(epochs or self.tr["epochs"]):
            perm = torch.randperm(n, generator=gen)
            tot = 0.0
            for s in range(0, n, bs):
                i = perm[s:s + bs]
                w = self.net(X[i])
                if self.fusion == "logit":
                    bce = nn.functional.binary_cross_entropy_with_logits((w * Z[i]).sum(1), Y[i])
                else:
                    pf = (w * P[i]).sum(1).clamp(1e-6, 1 - 1e-6)
                    bce = nn.functional.binary_cross_entropy(pf, Y[i])
                ent = -(w * w.clamp_min(1e-8).log()).sum(1).mean()
                loss = bce - lam * ent
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * len(i)
            self.history.append(tot / n)
        self.net.eval()
        log.info("gating fitted: n=%d loss %.4f -> %.4f", n, self.history[0], self.history[-1])
        return self

    @torch.no_grad()
    def weights(self, fs: FeatureSet) -> np.ndarray:
        self.net.eval()
        return self.net(torch.as_tensor(fs.X, dtype=torch.float32)).numpy().astype(np.float64)

    def predict(self, fs: FeatureSet) -> np.ndarray:
        return fuse(self.weights(fs), fs.p, fs.z, self.fusion)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path)

    def load(self, path: str | Path) -> "GatingController":
        self.net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.net.eval()
        return self
