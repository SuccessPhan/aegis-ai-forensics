"""Expert 1 - Semantic (UnivFD-style): frozen CLIP image embedding (projection) + linear head."""
from __future__ import annotations

import torch
from torch import nn

from src.core.logger import get_logger
from src.models.base import FrozenBackboneExpert, images_to_tensor

log = get_logger(__name__)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class _ClipImageEncoder(nn.Module):
    """pixel_values -> projected image embedding (vision tower + visual_projection)."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.vision_model = model.vision_model
        self.visual_projection = model.visual_projection

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pooled = self.vision_model(pixel_values=pixel_values).pooler_output
        return self.visual_projection(pooled)


class ClipExpert(FrozenBackboneExpert):
    name = "clip"

    def __init__(self, cfg: dict, backbone: nn.Module | None = None, feature_dim: int | None = None) -> None:
        ecfg = cfg["experts"]["clip"]
        super().__init__(cfg, backbone, feature_dim or ecfg["feature_dim"])
        self.model_id = ecfg["model_id"]
        self.register_buffer("_mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(CLIP_STD).view(1, 3, 1, 1), persistent=False)

    def preprocess(self, images) -> torch.Tensor:
        x = images_to_tensor(images)
        if x.shape[-2:] != (224, 224):
            raise ValueError(f"CLIP expects pre-normalised 224x224 images, got {tuple(x.shape[-2:])}")
        return (x - self._mean.cpu()) / self._std.cpu()

    def _load_backbone(self) -> None:  # pragma: no cover - needs weights
        from transformers import CLIPModel
        ecfg = self.cfg["experts"]["clip"]
        last = None
        model = None
        for mid in (ecfg["model_id"], ecfg.get("fallback_model_id")):
            if not mid:
                continue
            try:
                model = CLIPModel.from_pretrained(mid)
                self.model_id = mid
                break
            except Exception as e:  # noqa: BLE001 - try fallback
                log.warning("could not load %s (%s); trying fallback", mid, e)
                last = e
        if model is None:
            raise RuntimeError("no CLIP model could be loaded") from last
        self.backbone = _ClipImageEncoder(model)
        self.feature_dim = int(model.config.projection_dim)
        self._freeze()
        log.info("loaded CLIP %s (feature_dim=%d)", self.model_id, self.feature_dim)
