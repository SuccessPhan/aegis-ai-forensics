"""In-memory inference engine for the web demo. Nothing is written to disk and no per-image data is logged.

Pipeline (docs/WEB_API_CONTRACT.md): safe decode -> EXIF fix -> RGB -> resize_crop (the SAME functions and config values
as src/data/normalize.py, mode resize_crop, NO jpeg_harmonize) -> one 224x224 array shared by the 3 experts and the
3 quality features -> raw logits + cosine distances -> AegisSystem.predict_one -> JSON.
"""
from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image, ImageOps

from src.core.config import load_config, resolve_path
from src.core.contracts import EXPERT_NAMES, FLAG_CONFLICT, FLAG_OOD, FLAG_UNCERTAIN, QUALITY_FEATURE_NAMES
from src.core.logger import get_logger
from src.data.normalize import center_crop, resize_short_side
from src.data.quality import extract_quality_features
from src.ood.profiler import cosine_distance

log = get_logger(__name__)

MAX_PIXELS = 64_000_000            # decompression-bomb guard (~8000x8000); larger -> bad_image
ALLOWED_FORMATS = ("JPEG", "PNG", "WEBP")
HEAD_VARIANT = "aug"
FREQ_INPUT = "srm"
DEFAULT_SYSTEM_PATH = "serving/aegis_v3_aug.pt"      # relative to paths.cache_dir
DEFAULT_MODEL_VERSION = "aegis-v3-aug-srm"

Image.MAX_IMAGE_PIXELS = MAX_PIXELS


# internal contract constants -> web API short names (docs/WEB_API_CONTRACT.md)
FLAG_API_NAMES = {FLAG_CONFLICT: "conflict", FLAG_UNCERTAIN: "uncertain", FLAG_OOD: "ood"}


def map_flag_reasons(reasons) -> list:
    return [FLAG_API_NAMES.get(r, r) for r in reasons]


class ImageError(Exception):
    """Client-side image problem; `code` is the API error code (bad_image | too_small)."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def decode_image(data: bytes) -> Image.Image:
    """bytes -> RGB PIL image (first frame, EXIF orientation applied). Raises ImageError('bad_image')."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.format not in ALLOWED_FORMATS:
                raise ImageError("bad_image")
            w, h = im.size
            if w < 1 or h < 1 or w * h > MAX_PIXELS:
                raise ImageError("bad_image")
            im.seek(0)                      # first frame if animated (APNG / animated WebP)
            im.load()
            im = ImageOps.exif_transpose(im)
            return im.convert("RGB")
    except ImageError:
        raise
    except Exception as e:  # noqa: BLE001 - any decoding failure is a bad image
        raise ImageError("bad_image") from e


def preprocess_image(img: Image.Image, cfg: dict) -> np.ndarray:
    """resize_crop exactly as normalize_image (mode resize_crop, no harmonize) -> (224,224,3) uint8."""
    ncfg = cfg["data"]["normalize"]
    size = int(ncfg["crop_size"])
    if min(img.size) < size:
        raise ImageError("too_small")
    rgb = resize_short_side(img, int(ncfg.get("resize_short_side", 256)))
    cropped, _ = center_crop(rgb, size, int(ncfg["crop_offset_multiple"]))
    return np.asarray(cropped.convert("RGB"))


class Engine:
    """Holds the loaded experts, centroids and AegisSystem. All components are injectable (tests use fakes)."""

    def __init__(self, cfg: dict, experts: Dict[str, Any], centroids: Dict[str, torch.Tensor], system: Any,
                 model_version: str = DEFAULT_MODEL_VERSION, device: str = "cpu") -> None:
        self.cfg, self.experts, self.centroids, self.system = cfg, experts, centroids, system
        self.model_version, self.device = model_version, device

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, device: Optional[str] = None, system_path: Optional[str | Path] = None) -> "Engine":
        from src.learning.export import AegisSystem
        from src.models import build_expert
        from src.models.heads import centroid_key, load_head_into, with_freq_input
        from src.ood.profiler import load_centroids

        cfg = load_config()
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cfg_f = with_freq_input(cfg, FREQ_INPUT)
        experts: Dict[str, Any] = {}
        for name in EXPERT_NAMES:
            e = build_expert(name, cfg_f).to(device)
            load_head_into(cfg_f, e, HEAD_VARIANT)
            if name != "freq":
                e.extract_features(torch.zeros(1, 3, 224, 224))     # force the backbone to load now
            experts[name] = e.eval()
        cents = dict(load_centroids(cfg))
        cents = {n: (cents[centroid_key("freq", FREQ_INPUT, HEAD_VARIANT)] if n == "freq" else cents[n])
                 for n in EXPERT_NAMES}
        sp = Path(system_path) if system_path else resolve_path(cfg, "cache_dir") / DEFAULT_SYSTEM_PATH
        system = AegisSystem.load(sp)
        version = (getattr(system, "metadata", None) or {}).get("model_version") or DEFAULT_MODEL_VERSION
        eng = cls(cfg, experts, cents, system, version, device)
        eng.warmup()
        log.info("engine ready: version=%s device=%s", version, device)
        return eng

    def warmup(self) -> None:
        self.analyze_array(np.zeros((224, 224, 3), dtype=np.uint8) + 127)

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def analyze_array(self, arr: np.ndarray) -> Dict[str, Any]:
        logits, cosd = [], []
        for name in EXPERT_NAMES:
            e = self.experts[name]
            x = e.preprocess([arr])
            if name == "freq":
                f, lg = e.features_and_logits(x)
            else:
                f = e.extract_features(x)
                lg = e.predict_logits(f)
            logits.append(float(lg.reshape(-1)[0]))
            cosd.append(float(cosine_distance(f.float().cpu().reshape(-1), self.centroids[name])))
        qf = extract_quality_features(arr, self.cfg)
        quality = [float(getattr(qf, n)) for n in QUALITY_FEATURE_NAMES]
        r = self.system.predict_one(np.asarray(logits), np.asarray(cosd), np.asarray(quality))
        return {
            "label": r["label"], "p_ai": r["p_ai"], "confidence": r["confidence"],
            "warning_flag": bool(r["warning_flag"]), "flag_reasons": map_flag_reasons(r["flag_reasons"]),
            "experts": {n: {"p_ai": r["p_cal"][n], "weight": r["weights"][n]} for n in EXPERT_NAMES},
            "quality": dict(zip(QUALITY_FEATURE_NAMES, quality)),
            "ood_score": r["ood_score"], "ood_threshold": r["ood_threshold"],
            "model_version": self.model_version,
        }

    def analyze(self, image_bytes: bytes) -> Dict[str, Any]:
        """bytes -> contract JSON dict (+ latency_ms). Raises ImageError for bad_image / too_small."""
        t0 = time.perf_counter()
        arr = preprocess_image(decode_image(image_bytes), self.cfg)
        out = self.analyze_array(arr)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        out["latency_ms"] = int(round((time.perf_counter() - t0) * 1000))
        return out
