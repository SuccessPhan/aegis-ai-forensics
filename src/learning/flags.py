"""Warning flags (brief 5.6). Thresholds come from config `flags`."""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.core.config import Config
from src.core.contracts import (EXPERT_NAMES, FLAG_CONFLICT, FLAG_OOD, FLAG_UNCERTAIN, GatingWeights, Prediction)


def fit_ood_threshold(mean_ood_val: np.ndarray, cfg: Config) -> float:
    """Percentile (config flags.ood_percentile) of the mean OOD score on val_gating (we use val_gating_fit)."""
    return float(np.percentile(mean_ood_val, cfg["flags"]["ood_percentile"]))


def compute_flags(p: np.ndarray, p_final: np.ndarray, mean_ood: np.ndarray, ood_threshold: float,
                  cfg: Config) -> Tuple[np.ndarray, List[List[str]]]:
    """p: (n,3) expert probs, p_final: (n,), mean_ood: (n,). Returns (bool[n], reasons per sample).

    conflict : max_i P_i > conflict_high AND min_j P_j < conflict_low AND std(P1,P2,P3) > conflict_std (population std)
    uncertain: P_final in [uncertain_low, uncertain_high]
    ood      : mean OOD score > ood_threshold
    """
    f = cfg["flags"]
    conflict = (p.max(1) > f["conflict_high"]) & (p.min(1) < f["conflict_low"]) & (p.std(1) > f["conflict_std"])
    uncertain = (p_final >= f["uncertain_low"]) & (p_final <= f["uncertain_high"])
    ood = mean_ood > ood_threshold
    reasons = [[r for r, m in ((FLAG_CONFLICT, c), (FLAG_UNCERTAIN, u), (FLAG_OOD, o)) if m]
               for c, u, o in zip(conflict, uncertain, ood)]
    return conflict | uncertain | ood, reasons


def build_predictions(ids: Sequence[str], p: np.ndarray, p_final: np.ndarray, w: np.ndarray, mean_ood: np.ndarray,
                      ood_threshold: float, cfg: Config, source: str = "gating", alpha: float = 0.0
                      ) -> List[Prediction]:
    flags, reasons = compute_flags(p, p_final, mean_ood, ood_threshold, cfg)
    out = []
    for i, sid in enumerate(ids):
        pf = float(p_final[i])
        out.append(Prediction(sample_id=sid, p_final=pf, label=int(pf >= 0.5),
                              weights=GatingWeights(tuple(float(x) for x in w[i]), source, alpha),
                              expert_probs=tuple(float(x) for x in p[i]), warning_flag=bool(flags[i]),
                              flag_reasons=reasons[i], confidence=max(pf, 1 - pf), ood_score=float(mean_ood[i])))
    return out
