import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import load_config  # noqa: E402
from src.data.toy import load_or_build_toy  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def toy_manifest(cfg):
    """Toy dataset (100 clean + degraded variants); reused across pytest runs from data/test_toy_cache (gitignored)
    while the config/code key matches, regenerated otherwise."""
    return load_or_build_toy(ROOT / "data" / "test_toy_cache", cfg)
