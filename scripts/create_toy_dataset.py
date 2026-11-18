"""Generate data/toy/: 100 clean 224x224 PNGs (50 real / 50 fake, incl. unseen generators), fixed seed.

Also writes degraded variants for val_gating / test_benchmark (disable with --no-variants) and manifest.json.
Usage: python scripts/create_toy_dataset.py [--out data/toy] [--seed 42] [--no-variants]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.core.config import load_config, resolve_path  # noqa: E402
from src.data.toy import build_toy  # noqa: E402


def main(argv: Optional[List[str]] = None) -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(resolve_path(cfg, "toy_dir")))
    ap.add_argument("--seed", type=int, default=int(cfg["seed"]))
    ap.add_argument("--no-variants", action="store_true")
    a = ap.parse_args(argv)
    build_toy(a.out, cfg, seed=a.seed, with_variants=not a.no_variants)
    return 0


if __name__ == "__main__":
    sys.exit(main())
