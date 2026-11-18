"""Single source of truth for inter-module data structures in AEGIS.

Any change here is a BREAKING change and must be reported to the other agents.

Conventions
-----------
* label: 0 = Real, 1 = AI-generated (Fake).  p_fake / P_i = probability of class 1.
* Expert order is fixed everywhere: (clip, dinov2, freq) == EXPERT_NAMES.
* Split names are the constants in SPLITS (+ S4 pools in S4_SPLITS).

Gating input vector (12 dims, FIXED column order == GATING_FEATURE_NAMES)
------------------------------------------------------------------------
 0  jpeg_quality_est          blind blockiness-based quality estimate (pixels only)
 1  log_laplacian_var         log variance of the Laplacian (blur indicator)
 2  high_freq_energy_ratio    FFT energy fraction above cutoff
 3  entropy_clip              predictive entropy of expert 1
 4  entropy_dinov2            predictive entropy of expert 2
 5  entropy_freq              predictive entropy of expert 3
 6  cosdist_clip              cosine distance to train_experts centroid, expert 1
 7  cosdist_dinov2            ... expert 2
 8  cosdist_freq              ... expert 3
 9  p_clip                    calibrated P(fake), expert 1
10  p_dinov2                  calibrated P(fake), expert 2
11  p_freq                    calibrated P(fake), expert 3
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

LABEL_REAL = 0
LABEL_FAKE = 1

EXPERT_NAMES: Tuple[str, str, str] = ("clip", "dinov2", "freq")

SPLIT_TRAIN = "train_experts"
SPLIT_VAL_FIT = "val_gating_fit"
SPLIT_VAL_VERIFY = "val_gating_verify"
SPLIT_TEST = "test_benchmark"
SPLITS: Tuple[str, ...] = (SPLIT_TRAIN, SPLIT_VAL_FIT, SPLIT_VAL_VERIFY, SPLIT_TEST)
# S4 (OOD / few-shot) pools: unseen-generator AI + dedicated Real store. Disjoint from SPLITS.
SPLIT_S4_SUPPORT = "s4_support"
SPLIT_S4_EVAL = "s4_eval"
S4_SPLITS: Tuple[str, ...] = (SPLIT_S4_SUPPORT, SPLIT_S4_EVAL)
ALL_SPLITS: Tuple[str, ...] = SPLITS + S4_SPLITS

QUALITY_FEATURE_NAMES: Tuple[str, ...] = ("jpeg_quality_est", "log_laplacian_var", "high_freq_energy_ratio")
GATING_FEATURE_NAMES: Tuple[str, ...] = (
    QUALITY_FEATURE_NAMES
    + tuple(f"entropy_{e}" for e in EXPERT_NAMES)
    + tuple(f"cosdist_{e}" for e in EXPERT_NAMES)
    + tuple(f"p_{e}" for e in EXPERT_NAMES)
)
GATING_DIM = len(GATING_FEATURE_NAMES)  # 12

# Warning-flag reason codes (Prediction.flag_reasons)
FLAG_CONFLICT = "expert_conflict"
FLAG_UNCERTAIN = "uncertain_p_final"
FLAG_OOD = "high_ood_score"


# --------------------------------------------------------------------------- data
@dataclass
class DegradationSpec:
    """One degradation applied to a clean 224x224 image. All params None => clean.

    Order of application when several are set: blur -> resize(down,up) -> jpeg.
    """
    jpeg_quality: Optional[int] = None
    blur_sigma: Optional[float] = None
    blur_kernel: Optional[int] = None          # exact odd kernel size used
    resize_factor: Optional[float] = None      # downscale factor, then upscale back (bilinear)
    name: str = "clean"                        # e.g. "jpeg_q85", "blur_k11_s1.5", "resize_0.5", "val_jpeg_q47"
    scenario: Optional[str] = None             # "S2" / "S3" for fixed test levels, "val" for random val ones

    def is_clean(self) -> bool:
        return self.jpeg_quality is None and self.blur_sigma is None and self.resize_factor is None


@dataclass
class ImageRecord:
    sha256: str                                # SHA-256 of the stored PNG file bytes
    path: str                                  # relative to SplitManifest.root when set
    label: int                                 # 0 real / 1 fake
    generator: str                             # "real" for real images
    source_group_id: str                       # shared by a clean image and all its degraded variants
    split: Optional[str] = None
    degradation: Optional[DegradationSpec] = None   # None => clean
    parent_sha256: Optional[str] = None        # clean parent sha for degraded variants
    is_unseen_generator: bool = False
    width: int = 224
    height: int = 224
    provenance: Dict[str, Any] = field(default_factory=dict)  # source path, original size/format, crop offset, harmonize Q...

    @property
    def is_clean(self) -> bool:
        return self.degradation is None or self.degradation.is_clean()


@dataclass
class SplitManifest:
    records: List[ImageRecord] = field(default_factory=list)
    seed: int = 0
    root: str = ""                             # directory that record.path is relative to
    meta: Dict[str, Any] = field(default_factory=dict)

    def by_split(self, split: str, clean_only: bool = False) -> List[ImageRecord]:
        return [r for r in self.records if r.split == split and (r.is_clean or not clean_only)]

    def abs_path(self, rec: ImageRecord) -> Path:
        p = Path(rec.path)
        return p if p.is_absolute() or not self.root else Path(self.root) / p

    def to_dict(self) -> Dict[str, Any]:
        return {"seed": self.seed, "root": self.root, "meta": self.meta,
                "records": [asdict(r) for r in self.records]}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SplitManifest":
        recs = []
        for rd in d["records"]:
            rd = dict(rd)
            deg = rd.pop("degradation", None)
            recs.append(ImageRecord(degradation=DegradationSpec(**deg) if deg else None, **rd))
        return SplitManifest(records=recs, seed=d.get("seed", 0), root=d.get("root", ""), meta=d.get("meta", {}))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=1)

    @staticmethod
    def load(path: str | Path) -> "SplitManifest":
        with open(path, "r", encoding="utf-8") as f:
            m = SplitManifest.from_dict(json.load(f))
        if not m.root:
            m.root = str(Path(path).parent)
        return m


@dataclass
class SupportSet:
    """Few-shot support: ceil(N/2) AI + floor(N/2) Real, disjoint from the S4 eval pool."""
    n: int
    generator: Optional[str]
    seed: int
    ai_records: List[ImageRecord]
    real_records: List[ImageRecord]

    @property
    def records(self) -> List[ImageRecord]:
        return list(self.ai_records) + list(self.real_records)


# ------------------------------------------------------------------ model-side
@dataclass
class QualityFeatures:
    jpeg_quality_est: float
    log_laplacian_var: float
    high_freq_energy_ratio: float

    def to_array(self) -> np.ndarray:
        return np.array([self.jpeg_quality_est, self.log_laplacian_var, self.high_freq_energy_ratio], dtype=np.float32)


@dataclass
class ExpertOutput:
    """Output of one expert for one image."""
    expert: str                                # one of EXPERT_NAMES
    logit: float
    p_raw: float                               # uncalibrated P(fake)
    p_cal: Optional[float] = None              # calibrated P(fake) (Temperature/Platt on val_gating_fit)
    entropy: Optional[float] = None            # binary predictive entropy of p_cal (nats)
    cosine_distance: Optional[float] = None    # to train_experts centroid in this expert's feature space
    energy: Optional[float] = None


@dataclass
class OODProfile:
    quality: QualityFeatures
    entropy: Tuple[float, float, float]        # EXPERT_NAMES order
    cosine_distance: Tuple[float, float, float]
    energy: Optional[Tuple[float, float, float]] = None
    mean_ood_score: Optional[float] = None     # aggregate used for the 95th-percentile flag


@dataclass
class GatingInput:
    """12-d gating input; column order == GATING_FEATURE_NAMES (see module docstring)."""
    quality: QualityFeatures
    entropy: Tuple[float, float, float]
    cosine_distance: Tuple[float, float, float]
    p_cal: Tuple[float, float, float]

    def to_vector(self) -> np.ndarray:
        return np.array([*self.quality.to_array(), *self.entropy, *self.cosine_distance, *self.p_cal],
                        dtype=np.float32)

    @staticmethod
    def from_vector(v: Sequence[float]) -> "GatingInput":
        v = [float(x) for x in v]
        if len(v) != GATING_DIM:
            raise ValueError(f"expected {GATING_DIM} features, got {len(v)}")
        return GatingInput(QualityFeatures(*v[0:3]), tuple(v[3:6]), tuple(v[6:9]), tuple(v[9:12]))

    @staticmethod
    def stack(items: Sequence["GatingInput"]) -> np.ndarray:
        """(n, 12) float32 matrix."""
        return np.stack([i.to_vector() for i in items]) if items else np.zeros((0, GATING_DIM), np.float32)


@dataclass
class GatingWeights:
    """w = w_min + scale * softmax(z); sum==1, each in [w_min, w_min+scale]."""
    w: Tuple[float, float, float]
    source: str = "gating"                     # "gating" | "base" | "adapted" | "refit"
    alpha: float = 0.0                         # anchor coefficient used when source == "adapted"

    def validate(self, w_min: float = 0.05, scale: float = 0.85, tol: float = 1e-5) -> bool:
        s = sum(self.w)
        return abs(s - 1.0) <= tol and all(w_min - tol <= x <= w_min + scale + tol for x in self.w)


@dataclass
class Prediction:
    sample_id: str                             # record sha256 (or any id)
    p_final: float
    label: int                                 # 1 if p_final >= 0.5 else 0
    weights: GatingWeights
    expert_probs: Tuple[float, float, float]
    warning_flag: bool = False
    flag_reasons: List[str] = field(default_factory=list)   # FLAG_* codes
    confidence: Optional[float] = None         # e.g. max(p_final, 1-p_final)
    ood_score: Optional[float] = None


@dataclass
class EvalReport:
    scenario: str                              # "S1" | "S2" | "S3" | "S4"
    method: str                                # "clip" | "dinov2" | "freq" | "naive_avg" | "static_lr" | "aegis"
    n: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    auroc: float
    ece: float
    selective_accuracy: Optional[float] = None  # accuracy over non-flagged samples
    flagged_rate: Optional[float] = None
    extras: Dict[str, Any] = field(default_factory=dict)   # e.g. {"N": 20, "level": "jpeg_q65"}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
