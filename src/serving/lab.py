"""Lab mode for the demo server (docs/WEB_API_CONTRACT.md, "Lab mode").

Picks 8 clean held-out `test_benchmark` images (4 Real + 2 adm + 2 sd_v1_4) with a fixed seed, and builds the 4 degraded
variants of a chosen one with the SAME code as the evaluation (src/data/degradations). Read-only: nothing is written
to disk and nothing per-image is logged.
"""
from __future__ import annotations

import base64
import io
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from src.core.config import resolve_path
from src.core.contracts import SPLIT_TEST, DegradationSpec, ImageRecord, SplitManifest
from src.data.degradations import apply_degradation

DEFAULT_SEED = 7
THUMB_SIDE = 160
QUOTA = (("real", 4), ("adm", 2), ("sd_v1_4", 2))
# (key, Vietnamese title, degradation spec or None for the clean image); fixed order per the contract
VARIANTS: Tuple[Tuple[str, str, Optional[DegradationSpec]], ...] = (
    ("clean", "Ảnh gốc", None),
    ("blur_s1.5", "Mờ nhẹ", DegradationSpec(blur_sigma=1.5, blur_kernel=11, name="blur_k11_s1.5", scenario="S3")),
    ("blur_s2.5", "Mờ nặng", DegradationSpec(blur_sigma=2.5, blur_kernel=17, name="blur_k17_s2.5", scenario="S3")),
    ("jpeg_q40", "Nén JPEG mạnh", DegradationSpec(jpeg_quality=40, name="jpeg_q40", scenario="S2")),
)


class LabUnavailable(Exception):
    """Manifest or images missing / insufficient."""


class UnknownSample(Exception):
    pass


def _data_url(img: Image.Image, fmt: str) -> str:
    b = io.BytesIO()
    if fmt == "JPEG":
        img.convert("RGB").save(b, format="JPEG", quality=75)
        mime = "image/jpeg"
    else:
        img.convert("RGB").save(b, format="PNG")
        mime = "image/png"
    return f"data:{mime};base64," + base64.b64encode(b.getvalue()).decode("ascii")


class Lab:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.seed = int((cfg.get("lab") or {}).get("seed", DEFAULT_SEED))
        self._samples: Optional[List[Dict[str, Any]]] = None     # public list incl. thumbs
        self._paths: Dict[str, Path] = {}

    def _select(self) -> List[Tuple[ImageRecord, Path]]:
        try:
            man = SplitManifest.load(resolve_path(self.cfg, "manifest"))
        except Exception as e:  # noqa: BLE001
            raise LabUnavailable("manifest") from e
        pool = sorted(man.by_split(SPLIT_TEST, clean_only=True), key=lambda r: r.sha256)
        rng = random.Random(self.seed)
        chosen: List[ImageRecord] = []
        for gen, n in QUOTA:
            cand = [r for r in pool if r.generator == gen]
            if len(cand) < n:
                raise LabUnavailable("not enough samples")
            chosen += rng.sample(cand, n)
        return [(r, man.abs_path(r)) for r in chosen]

    def samples(self) -> List[Dict[str, Any]]:
        if self._samples is None:
            out, paths = [], {}
            for i, (r, p) in enumerate(self._select(), 1):
                try:
                    with Image.open(p) as im:
                        im.load()
                        th = im.convert("RGB")
                except Exception as e:  # noqa: BLE001
                    raise LabUnavailable("image") from e
                th.thumbnail((THUMB_SIDE, THUMB_SIDE))
                sid = f"s{i}"
                paths[sid] = p
                out.append({"id": sid, "truth": "AI" if r.label == 1 else "Real", "source": r.generator,
                            "thumb": _data_url(th, "JPEG")})
            self._paths, self._samples = paths, out
        return self._samples

    def load_clean(self, sample_id: str) -> np.ndarray:
        self.samples()
        if sample_id not in self._paths:
            raise UnknownSample(sample_id)
        try:
            with Image.open(self._paths[sample_id]) as im:
                return np.asarray(im.convert("RGB")).copy()
        except Exception as e:  # noqa: BLE001
            raise LabUnavailable("image") from e

    def variant_images(self, clean: np.ndarray) -> List[Tuple[str, str, np.ndarray]]:
        base = Image.fromarray(clean)
        out = []
        for key, title, spec in VARIANTS:
            arr = clean if spec is None else np.asarray(apply_degradation(base, spec))
            out.append((key, title, arr))
        return out

    def run(self, engine: Any, sample_id: str) -> Dict[str, Any]:
        """Blocking; call from an executor while holding the GPU lock."""
        clean = self.load_clean(sample_id)
        meta = next(s for s in self.samples() if s["id"] == sample_id)
        variants = []
        for key, title, arr in self.variant_images(clean):
            res = engine.analyze_array(arr)
            v = {"key": key, "title": title, "thumb": _data_url(Image.fromarray(arr), "PNG")}
            for k in ("label", "p_ai", "confidence", "warning_flag", "flag_reasons", "experts", "quality"):
                v[k] = res[k]
            variants.append(v)
        return {"sample_id": sample_id, "truth": meta["truth"], "source": meta["source"],
                "model_version": engine.model_version, "variants": variants}
