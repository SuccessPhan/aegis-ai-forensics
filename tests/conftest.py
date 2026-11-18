import copy
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


def no_variants(cfg: dict) -> dict:
    """Config copy with val/train_aug variants off and a single fixed test level (for tests that do not inspect variants)."""
    c = copy.deepcopy(cfg)
    c["degradation"]["val"]["variants_per_source"] = 0
    c["degradation"]["train_aug"]["variants_per_source"] = 0
    c["degradation"]["test"]["s2_jpeg_quality"] = []
    c["degradation"]["test"]["s3_blur"] = []
    return c


@pytest.fixture(scope="session")
def toy_manifest_resize_crop(cfg):
    """Toy built in resize_crop mode; disk-cached like toy_manifest (cache key includes data.normalize.mode)."""
    c = copy.deepcopy(cfg)
    c["data"]["normalize"]["mode"] = "resize_crop"
    return load_or_build_toy(ROOT / "data" / "test_toy_cache_resize_crop", c)
