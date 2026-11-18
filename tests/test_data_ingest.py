import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from src.core.config import deep_merge
from src.core.contracts import LABEL_FAKE, LABEL_REAL, SplitManifest
from src.data.ingest import IngestError, main, scan_genimage

N = 6
GENS = ["adm", "sd_v1_4", "midjourney", "glide"]


def _img(path: Path, seed: int, fmt: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.default_rng(seed).integers(0, 256, (240, 240, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path, format=fmt)


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "genimage"
    for gi, gen in enumerate(GENS + ["biggan"]):
        for i in range(N):
            _img(root / gen / "ai" / f"{i}.png", 1000 * gi + i, "PNG")
            _img(root / gen / "nature" / f"{i}.JPEG", 100 * gi + i + 50000, "JPEG")
    # the same Real image under two different generators
    shutil.copy(root / "adm" / "nature" / "0.JPEG", root / "glide" / "nature" / "dup.JPEG")
    return root


def test_scan_filters_and_extensions(tree, cfg):
    items = scan_genimage(tree, cfg)
    assert {i.generator for i in items} == set(GENS) | {"real"}
    assert len(items) == 4 * 2 * N + 1
    assert all(i.label == (LABEL_REAL if i.generator == "real" else LABEL_FAKE) for i in items)
    assert any(i.path.suffix == ".JPEG" for i in items)
    assert not any("biggan" in i.path.parts for i in items)


def test_max_per_class_balanced_and_seeded(tree, cfg):
    a = scan_genimage(tree, cfg, max_per_class=3)
    b = scan_genimage(tree, cfg, max_per_class=3)
    assert [i.path for i in a] == [i.path for i in b]
    for gen in GENS:
        fake = [i for i in a if i.generator == gen]
        real = [i for i in a if i.label == LABEL_REAL and gen in i.path.parts]
        assert len(fake) == len(real) == 3


def test_missing_and_empty_root(tmp_path, cfg):
    with pytest.raises(IngestError, match="adm"):
        scan_genimage(tmp_path / "nope", cfg)
    (tmp_path / "empty").mkdir()
    with pytest.raises(IngestError, match="no images"):
        scan_genimage(tmp_path / "empty", cfg)
    assert main(["--root", str(tmp_path / "nope"), "--out", str(tmp_path / "o")]) == 2


def test_cli_end_to_end(tree, tmp_path, cfg):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump(deep_merge(cfg, {"split": cfg["toy"]["split_overrides"]})))
    out = tmp_path / "norm"
    assert main(["--root", str(tree), "--out", str(out), "--config", str(cfg_path)]) == 0
    man = SplitManifest.load(out / "manifest.json")
    clean = [r for r in man.records if r.is_clean]
    assert len(clean) == 4 * 2 * N          # the duplicated Real was dropped
    assert not any(r.generator == "biggan" for r in man.records)
    assert all(r.generator == "real" for r in clean if r.label == LABEL_REAL)
    # unseen generators only in S4 pools; Real from unseen dirs is ordinary Real (main splits possible)
    assert {r.split for r in clean if r.generator in ("midjourney", "glide")} <= {"s4_support", "s4_eval"}
    assert any(r.label == LABEL_REAL and r.split in ("train_experts", "val_gating_fit", "test_benchmark")
               for r in clean)
    rep = json.loads((out / "shortcut_audit.json").read_text())
    assert rep["shortcut_audit"]["status"] in ("pass", "fail", "skipped")
    assert rep["stats"]
