"""Common expert interface.

Every expert is an ``nn.Module`` exposing:
  preprocess(images)        -> float tensor (n, 3, H, W); NEVER resizes
  extract_features(batch)   -> (n, d) float32 features (no grad)
  predict_logits(features)  -> (n,) raw logits for class 1 (Fake); p_fake = sigmoid(logit)
Inputs are already-normalised 224x224 images (PIL, HWC uint8 arrays or an (n,H,W,3) array).
"""
from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import torch
from PIL import Image
from torch import nn

ImageLike = Union[Image.Image, np.ndarray]


def images_to_tensor(images: Union[Sequence[ImageLike], np.ndarray]) -> torch.Tensor:
    """uint8 images -> float32 (n, 3, H, W) in [0, 1]. No resizing."""
    arrs = []
    for im in images:
        a = np.asarray(im.convert("RGB")) if isinstance(im, Image.Image) else np.asarray(im)
        if a.ndim == 2:
            a = np.stack([a] * 3, axis=-1)
        arrs.append(a)
    x = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).contiguous()
    return x.to(torch.float32) / 255.0


def load_image_array(path: str) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


class BaseExpert(nn.Module):
    name: str = "base"
    model_id: str = ""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_dev", torch.zeros(1), persistent=False)

    @property
    def device(self) -> torch.device:
        return self._dev.device

    def preprocess(self, images) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def extract_features(self, batch: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def predict_logits(self, features: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        """Uncalibrated P(fake)."""
        with torch.no_grad():
            return torch.sigmoid(self.predict_logits(features))


class FrozenBackboneExpert(BaseExpert):
    """Expert = frozen backbone (eval, no_grad) + optional trainable LinearHead (set via `head`)."""

    def __init__(self, cfg: dict, backbone: nn.Module | None = None, feature_dim: int | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.head: nn.Module | None = None
        if backbone is not None:
            self._freeze()

    def _freeze(self) -> None:
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def _load_backbone(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbone is not None:
            self.backbone.eval()  # backbone is always in eval mode
        return self

    @torch.no_grad()
    def extract_features(self, batch: torch.Tensor) -> torch.Tensor:
        if self.backbone is None:
            self._load_backbone()
            self.backbone.to(self.device)
        return self.backbone(batch.to(self.device)).float().cpu()

    def predict_logits(self, features: torch.Tensor) -> torch.Tensor:
        if self.head is None:
            raise RuntimeError(f"{self.name}: no head attached (train or load one first)")
        return self.head(features.to(self.device).float()).reshape(-1)
