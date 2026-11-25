"""External test sets (v4): folder ingestion (src.data.external), evaluation (src.evaluation.external_eval) and the
Hugging Face download helper (scripts/fetch_external.py)."""
import io
import json
import shutil

import numpy as np
import pytest
from PIL import Image

from src.core.contracts import SplitManifest
from src.data.external import ingest_external, scan_folder, split_name
from src.evaluation.external_eval import (DEEP_METHODS, RESULT_COLUMNS, check_ready, evaluate_external,
                                          evaluation_items, external_split, summary_lines)
from src.learning.classic_models import DEFAULT_MODELS

DEMO_DIR = "test_demo_external"


@pytest.fixture(scope="module")
def demo(cfg):
    from tests.conftest import ROOT
    from src.models.demo_features import build_demo
    return build_demo(ROOT / "data" / DEMO_DIR / "demo", scale=2, cfg=cfg)


@pytest.fixture(scope="module")
def raw_dir(tmp_path_factory):
    from src.models.demo_features import build_demo_external_raw
    return build_demo_external_raw(tmp_path_factory.mktemp("raw") / "newgen", n_per_generator=12, n_real=16)


@pytest.fixture(scope="module")
def ext(demo, raw_dir, tmp_path_factory):
    cfg, man = demo
    from src.models.demo_features import build_demo_external
    e = ingest_external(raw_dir, tmp_path_factory.mktemp("ext") / "newgen", "newgen", cfg)
    build_demo_external(cfg, man, e)
    return e


def _img(path, size=240, fmt="PNG", seed=0):
    arr = np.random.default_rng(seed).integers(0, 256, (size, size, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, format=fmt)


# --------------------------------------------------------------------------------------------- folder ingestion
def test_scan_folder_layout(tmp_path):
    for i in range(3):
        _img(tmp_path / "Nature" / f"{i}.jpg", fmt="JPEG", seed=i)
        _img(tmp_path / "FLUX 1" / "sub" / f"{i}.png", seed=10 + i)
    (tmp_path / "notes.txt").write_text("ignored")
    (tmp_path / "empty").mkdir()
    g = scan_folder(tmp_path)
    assert set(g) == {"real", "flux_1"} and len(g["real"]) == 3 and len(g["flux_1"]) == 3
    assert len(scan_folder(tmp_path, max_per_generator=2)["flux_1"]) == 2
    with pytest.raises(FileNotFoundError):
        scan_folder(tmp_path / "missing")


def test_ingest_normalises_like_genimage(raw_dir, tmp_path, cfg):
    man = ingest_external(raw_dir, tmp_path / "h", "t1", cfg)
    assert external_split(man) == split_name("t1") == "ext_t1"
    assert {r.generator for r in man.records} == {"real", "newgen_a", "newgen_b"}
    assert {r.label for r in man.records if r.generator == "real"} == {0}
    assert all(r.is_unseen_generator == (r.label == 1) for r in man.records)
    for r in man.records[:5]:
        with Image.open(man.abs_path(r)) as im:
            assert im.size == (224, 224) and im.format == "PNG"
    ai = [r for r in man.records if r.label == 1]
    assert all(r.provenance["harmonize_quality"] is not None for r in ai)       # JPEG-harmonised like GenImage
    loaded = SplitManifest.load(tmp_path / "h" / "manifest.json")
    assert len(loaded.records) == len(man.records) and loaded.meta["harmonize"] is True
    raw = ingest_external(raw_dir, tmp_path / "r", "t1", cfg, harmonize=False)
    assert all(r.provenance["harmonize_quality"] is None for r in raw.records)   # "web" setting
    assert raw.meta["harmonize"] is False


def test_ingest_skips_small_and_duplicate_images(tmp_path, cfg):
    _img(tmp_path / "in" / "real" / "a.jpg", fmt="JPEG", seed=1)
    _img(tmp_path / "in" / "gen" / "a.png", seed=2)
    shutil.copy(tmp_path / "in" / "gen" / "a.png", tmp_path / "in" / "gen" / "dup.png")
    _img(tmp_path / "in" / "gen" / "tiny.png", size=64, seed=3)
    man = ingest_external(tmp_path / "in", tmp_path / "out", "t2", cfg)
    assert len(man.records) == 2


# ---------------------------------------------------------------------------------------------------- evaluation
def test_check_ready_lists_missing_steps(demo, raw_dir, tmp_path):
    cfg, _ = demo
    fresh = ingest_external(raw_dir, tmp_path / "fresh", "notprepared", cfg)
    todo = check_ready(cfg, fresh, ["clean", "aug"], ["handcrafted", "clip", "dinov2"])
    assert sum("src.models.extract" in t for t in todo) == 2
    assert sum("src.models.bundle" in t for t in todo) == 2
    assert all("ext_notprepared" in t for t in todo)
    assert sum("src.models.extract" in t for t in check_ready(cfg, fresh, ["clean"], ["handcrafted"])) == 0
    with pytest.raises(FileNotFoundError):
        evaluate_external(cfg, demo[1], fresh, ["clean"], ["handcrafted"], B=0)


def test_evaluate_external_end_to_end(demo, ext, tmp_path):
    cfg, man = demo
    assert check_ready(cfg, ext, ["clean", "aug"], ["handcrafted", "clip", "dinov2"]) == []
    out = evaluate_external(cfg, man, ext, ["clean"], ["handcrafted", "clip"], B=50, out_dir=tmp_path)
    res = out["results"]
    assert all(set(RESULT_COLUMNS) <= set(r) for r in res)
    assert {r["generator"] for r in res} == {"newgen_a", "newgen_b", "all"}
    assert {r["model"] for r in res if r["group"] == "deep"} == set(DEEP_METHODS)
    assert {(r["feature_set"], r["model"]) for r in res if r["group"] == "ml"} == \
        {(f, m) for f in ("handcrafted", "clip") for m in DEFAULT_MODELS}
    a = [r for r in res if r["generator"] == "newgen_a"][0]
    assert a["n_real"] == 16 and a["n_ai"] == 12 and a["n"] == 28
    assert all(0 <= r["auroc"] <= 1 for r in res)
    assert out["significance"] and "aegis" not in {r["model"] for r in out["significance"]}
    for f in ("results.csv", "significance.csv", "summary.md"):
        assert (tmp_path / f).exists()
    assert "newgen_a" in (tmp_path / "summary.md").read_text()
    assert summary_lines(res)[0].startswith("| head")


def test_external_results_deterministic(demo, ext):
    cfg, man = demo
    a = evaluate_external(cfg, man, ext, ["aug"], ["handcrafted"], ["random_forest"], B=0)["results"]
    b = evaluate_external(cfg, man, ext, ["aug"], ["handcrafted"], ["random_forest"], B=0)["results"]
    assert [r["auroc"] for r in a] == [r["auroc"] for r in b]


def test_real_from_main_for_sets_without_real(demo, ext):
    cfg, man = demo
    ai_only = SplitManifest(records=[r for r in ext.records if r.label == 1], root=ext.root, meta=dict(ext.meta))
    shas, y, gens, origin = evaluation_items(ai_only, man, real_from_main=False)    # no real -> taken from main
    assert (y == 0).sum() > 0 and set(np.array(origin)[y == 0]) == {"main_test"}
    with pytest.raises(ValueError):
        evaluation_items(ai_only, None, real_from_main=False)
    out = evaluate_external(cfg, man, ai_only, ["clean"], ["handcrafted"], ["xgboost"], B=0)
    assert {r["generator"] for r in out["results"]} == {"newgen_a", "newgen_b", "all"}


def test_external_split_must_be_single(ext):
    mixed = SplitManifest(records=list(ext.records), root=ext.root)
    mixed.records[0] = type(mixed.records[0])(**{**mixed.records[0].__dict__, "split": "test_benchmark"})
    with pytest.raises(ValueError):
        external_split(mixed)


# ----------------------------------------------------------------------------------------- Hugging Face helper
def _fetch_module():
    import importlib.util
    from tests.conftest import ROOT
    spec = importlib.util.spec_from_file_location("fetch_external", ROOT / "scripts" / "fetch_external.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fetch_external_writes_original_bytes(tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    mod = _fetch_module()

    def enc(fmt, seed):
        b = io.BytesIO()
        Image.fromarray(np.random.default_rng(seed).integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(b, fmt)
        return b.getvalue()
    imgs = [enc("JPEG", 0), enc("PNG", 1), enc("PNG", 2), enc("PNG", 3), b"not an image"]
    table = pa.table({"image": [{"bytes": b, "path": None} for b in imgs],
                      "label": ["real", "ai", "ai", "ai", "ai"], "generator": ["real", "FLUX", "flux", "sdxl", "x"]})
    pq.write_table(table, tmp_path / "train-00000.parquet", row_group_size=2)
    monkeypatch.setattr(mod, "parquet_files", lambda *a, **k: ["data/train-00000.parquet"])
    monkeypatch.setattr(mod, "download", lambda ds, f: tmp_path / "train-00000.parquet")
    out = tmp_path / "out"
    assert mod.main(["--dataset", "x", "--split", "train", "--out", str(out), "--generator-column", "generator",
                     "--max-per-generator", "1"]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["flux", "real", "sdxl"]          # unreadable image skipped
    real = next((out / "real").iterdir())
    assert real.suffix == ".jpg" and real.read_bytes() == imgs[0]                      # not re-encoded
    assert len(list((out / "flux").iterdir())) == 1                                    # max-per-generator


def test_fetch_external_selects_split_files(monkeypatch):
    hub = pytest.importorskip("huggingface_hub")
    mod = _fetch_module()
    files = ["README.md", "data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet",
             "data/test-00000-of-00001.parquet", "cfgA/test/0000.parquet", "cfgB/test/0000.parquet"]

    class FakeApi:
        def list_repo_files(self, *a, **k):
            return files
    monkeypatch.setattr(hub, "HfApi", FakeApi)
    assert mod.parquet_files("x", "train") == files[1:3]
    assert mod.parquet_files("x", "test", "cfgA") == ["cfgA/test/0000.parquet"]
    with pytest.raises(FileNotFoundError):
        mod.parquet_files("x", "validation")
