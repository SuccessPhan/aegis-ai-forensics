"""Load configs/default.yaml into a plain nested dict (single source of hyperparameters)."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"

Config = Dict[str, Any]


def load_config(path: Optional[str | Path] = None, overrides: Optional[Config] = None) -> Config:
    """Load YAML config; `overrides` is deep-merged on top."""
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def deep_merge(base: Config, over: Config) -> Config:
    """Return a new dict = base with `over` recursively merged in."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def get(cfg: Config, dotted: str, default: Any = None) -> Any:
    """cfg['a']['b'] via get(cfg, 'a.b')."""
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def resolve_path(cfg: Config, key: str) -> Path:
    """Resolve cfg['paths'][key] against the repo root (absolute paths unchanged)."""
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else REPO_ROOT / p
