"""Champion-Challenger on val_gating_verify (NEVER test_benchmark) + on-disk version registry + auto rollback.

predict_fn(model, data) -> p_final array; the caller decides what `model` / `data` are (e.g. a (pipeline, controller)
tuple and a Bundle). `y` must be the ground-truth labels of `data`.
"""
from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from src.core.config import Config
from src.core.contracts import SPLIT_VAL_VERIFY
from src.core.logger import get_logger
from src.evaluation.metrics import compute_metrics

log = get_logger(__name__)


class SplitGuardError(ValueError):
    """Raised when a promotion/rollback decision is attempted on a split other than val_gating_verify."""


def _guard(split: str, cfg: Config) -> None:
    allowed = cfg["evaluation"]["champion_challenger"]["split"]
    if split != allowed or split == "test_benchmark":
        raise SplitGuardError(f"champion/challenger decisions are only allowed on {allowed!r}, got {split!r}")


class ModelRegistry:
    """Versioned model store: <root>/v0001.pkl + registry.json (history, champion pointer, promotion stack)."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "registry.json"
        self.index: Dict[str, Any] = {"versions": [], "champion": None, "stack": []}
        if self.index_path.exists():
            self.index = json.loads(self.index_path.read_text(encoding="utf-8"))

    def _save(self) -> None:
        self.index_path.write_text(json.dumps(self.index, indent=1), encoding="utf-8")

    @property
    def champion_id(self) -> Optional[str]:
        return self.index["champion"]

    @property
    def history(self) -> List[Dict[str, Any]]:
        return list(self.index["versions"])

    def add(self, model: Any, metrics: Dict[str, float], note: str = "") -> str:
        vid = f"v{len(self.index['versions']) + 1:04d}"
        with open(self.root / f"{vid}.pkl", "wb") as f:
            pickle.dump(model, f)
        self.index["versions"].append({"id": vid, "time": time.time(), "metrics": metrics, "note": note,
                                       "status": "candidate"})
        self._save()
        return vid

    def get(self, vid: Optional[str] = None) -> Any:
        vid = vid or self.champion_id
        if vid is None:
            raise KeyError("registry has no champion")
        with open(self.root / f"{vid}.pkl", "rb") as f:
            return pickle.load(f)

    def record(self, vid: str) -> Dict[str, Any]:
        return next(v for v in self.index["versions"] if v["id"] == vid)

    def promote(self, vid: str) -> None:
        old = self.champion_id
        if old is not None:
            self.index["stack"].append(old)
            self.record(old)["status"] = "retired"
        self.index["champion"] = vid
        self.record(vid)["status"] = "champion"
        self._save()

    def rollback(self, reason: str = "") -> Optional[str]:
        """Restore the previous champion; the demoted version is marked 'rolled_back'. None if nothing to restore."""
        if not self.index["stack"]:
            return None
        bad = self.champion_id
        prev = self.index["stack"].pop()
        self.record(bad)["status"] = "rolled_back"
        self.record(bad)["note"] += f" | rollback: {reason}"
        self.record(prev)["status"] = "champion"
        self.index["champion"] = prev
        self._save()
        log.warning("ROLLBACK %s -> %s (%s)", bad, prev, reason)
        return prev


@dataclass
class Decision:
    promoted: bool
    reason: str
    version: Optional[str] = None
    champion_metrics: Dict[str, float] = field(default_factory=dict)
    challenger_metrics: Dict[str, float] = field(default_factory=dict)


class ChampionChallenger:
    def __init__(self, registry: ModelRegistry, cfg: Config, predict_fn: Callable[[Any, Any], np.ndarray]):
        self.reg, self.cfg, self.predict_fn = registry, cfg, predict_fn
        cc = cfg["evaluation"]["champion_challenger"]
        self.primary, self.min_imp, self.max_ece_up = cc["primary_metric"], cc["min_improvement"], cc["max_ece_increase"]
        self.primary_tol = cc.get("rollback_primary_tolerance", 0.01)

    def _metrics(self, model: Any, data: Any, y: np.ndarray) -> Dict[str, float]:
        r = compute_metrics(y, self.predict_fn(model, data), self.cfg)
        return {self.primary: float(getattr(r, self.primary)), "ece": r.ece}

    def submit(self, challenger: Any, data: Any, y: np.ndarray, split: str, note: str = "") -> Decision:
        """Evaluate challenger vs champion on val_gating_verify; promote only if strictly better by the config rule."""
        _guard(split, self.cfg)
        ch = self._metrics(challenger, data, y)
        if self.reg.champion_id is None:
            vid = self.reg.add(challenger, ch, note or "initial")
            self.reg.promote(vid)
            log.info("first model %s becomes champion %s", vid, ch)
            return Decision(True, "no champion yet", vid, {}, ch)
        cm = self._metrics(self.reg.get(), data, y)
        better = ch[self.primary] > cm[self.primary] + self.min_imp
        ece_ok = ch["ece"] <= cm["ece"] + self.max_ece_up
        vid = self.reg.add(challenger, ch, note)
        if better and ece_ok:
            self.reg.promote(vid)
            log.info("PROMOTE %s: %s %.4f > %.4f", vid, self.primary, ch[self.primary], cm[self.primary])
            return Decision(True, "improved", vid, cm, ch)
        why = (f"{self.primary} {ch[self.primary]:.4f} not > {cm[self.primary]:.4f}+{self.min_imp}" if not better
               else f"ece {ch['ece']:.4f} > {cm['ece']:.4f}+{self.max_ece_up}")
        self.reg.record(vid)["status"] = "rejected"
        self.reg._save()
        log.info("REJECT %s: %s", vid, why)
        return Decision(False, why, vid, cm, ch)

    def monitor(self, data: Any, y: np.ndarray, split: str) -> Decision:
        """Re-check the current champion on (fresh) val_gating_verify data; auto-rollback on regression versus the
        metrics it was promoted with (primary drop > tolerance or ECE up by > max_ece_increase)."""
        _guard(split, self.cfg)
        vid = self.reg.champion_id
        base = self.reg.record(vid)["metrics"]
        now = self._metrics(self.reg.get(), data, y)
        drop = base[self.primary] - now[self.primary] > self.primary_tol
        ece_up = now["ece"] - base["ece"] > self.max_ece_up
        if not (drop or ece_up):
            return Decision(False, "healthy", vid, base, now)
        prev = self.reg.rollback(f"{self.primary} {base[self.primary]:.4f}->{now[self.primary]:.4f}, "
                                 f"ece {base['ece']:.4f}->{now['ece']:.4f}")
        return Decision(prev is not None, "regression: rolled back" if prev else "regression: nothing to roll back",
                        prev, base, now)
