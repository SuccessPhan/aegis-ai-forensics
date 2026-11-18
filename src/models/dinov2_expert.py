"""Expert 2 - Fine-grained structure: frozen DINOv2 CLS token (768-d) + linear head."""
from __future__ import annotations

import torch
from torch import nn

from src.core.logger import get_logger
from src.models.base import FrozenBackboneExpert, images_to_tensor

log = get_logger(__name__)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class _DinoCls(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).last_hidden_state[:, 0]


class Dinov2Expert(FrozenBackboneExpert):
    name = "dinov2"

    def __init__(self, cfg: dict, backbone: nn.Module | None = None, feature_dim: int | None = None) -> None:
        ecfg = cfg["experts"]["dinov2"]
        super().__init__(cfg, backbone, feature_dim or ecfg["feature_dim"])
        self.model_id = ecfg["model_id"]
        self.register_buffer("_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def preprocess(self, images) -> torch.Tensor:
        x = images_to_tensor(images)  # no resize; 224 is a multiple of patch size 14
        return (x - self._mean.cpu()) / self._std.cpu()

    def _load_backbone(self) -> None:  # pragma: no cover - needs weights
        from transformers import AutoModel
        model = AutoModel.from_pretrained(self.model_id)
        self.backbone = _DinoCls(model)
        self.feature_dim = int(model.config.hidden_size)
        self._freeze()
        log.info("loaded DINOv2 %s (feature_dim=%d)", self.model_id, self.feature_dim)
