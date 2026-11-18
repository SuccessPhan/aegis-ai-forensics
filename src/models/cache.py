"""Embedding cache: one .pt per (expert, cache_version) -> {sha256: float16 feature}.

Key = sha256 + expert + cache_version (expert & version live in the file name and meta).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Sequence

import torch

from src.core.config import resolve_path
from src.core.logger import get_logger

log = get_logger(__name__)


class FeatureCache:
    def __init__(self, path: Path, expert: str, version: str, dtype: torch.dtype = torch.float16) -> None:
        self.path, self.expert, self.version, self.dtype = Path(path), expert, version, dtype
        self.data: Dict[str, torch.Tensor] = {}
        self.meta: dict = {"expert": expert, "cache_version": version}
        if self.path.exists():
            blob = torch.load(self.path, map_location="cpu", weights_only=True)
            self.data, self.meta = blob["features"], blob["meta"]

    @classmethod
    def open(cls, cfg: dict, expert: str) -> "FeatureCache":
        ex = cfg["experts"]["extract"]
        v = ex["cache_version"]
        return cls(resolve_path(cfg, "cache_dir") / "features" / f"{expert}_{v}.pt", expert, v,
                   getattr(torch, ex.get("dtype", "float16")))

    def __len__(self) -> int:
        return len(self.data)

    def has(self, sha: str) -> bool:
        return sha in self.data

    def put(self, sha: str, feat: torch.Tensor) -> None:
        self.data[sha] = feat.detach().cpu().to(self.dtype)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        torch.save({"features": self.data, "meta": self.meta}, tmp)
        os.replace(tmp, self.path)

    def features_for(self, shas: Sequence[str]) -> torch.Tensor:
        missing = [s for s in shas if s not in self.data]
        if missing:
            raise KeyError(f"{len(missing)} sha256 missing from {self.expert} cache "
                           f"(run `python -m src.models.extract --expert {self.expert}` first)")
        return torch.stack([self.data[s] for s in shas]).float()
