"""Lab mode tests: fake engine, fake manifest + images in tmp_path."""
import base64
import io
import os

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from src.core.config import load_config
from src.core.contracts import EXPERT_NAMES, ImageRecord, SplitManifest, SPLIT_TEST, SPLIT_TRAIN
from src.data.degradations import apply_degradation, fixed_test_specs
from src.serving.app import create_app
from src.serving.engine import Engine
from src.serving.lab import VARIANTS, Lab
from tests.test_serving_api import FakeExpert, FakeSystem


class RecordingEngine(Engine):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.seen = []

    def analyze_array(self, arr):
        self.seen.append(arr.copy())
        return super().analyze_array(arr)


def _build_manifest(root, with_images=True, n_each=6):
    recs = []
    rng = np.random.default_rng(0)
    for gen, label in (("real", 0), ("adm", 1), ("sd_v1_4", 1)):
        for i in range(n_each):
            for split in (SPLIT_TEST, SPLIT_TRAIN):
                rel = f"{split}_{gen}_{i}.png"
                if with_images:
                    a = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
                    Image.fromarray(a).save(root / rel)
                recs.append(ImageRecord(sha256=f"{gen}{i:02d}{split[:2]}", path=rel, label=label, generator=gen,
                                        source_group_id=f"{gen}{i}{split}", split=split))
    SplitManifest(records=recs, root=str(root)).save(root / "manifest.json")
    return root / "manifest.json"


@pytest.fixture()
def lab_cfg(tmp_path):
    cfg = dict(load_config())
    cfg["paths"] = dict(cfg["paths"], manifest=str(_build_manifest(tmp_path)))
    return cfg


def _engine(cfg):
    ex = {n: FakeExpert(n) for n in EXPERT_NAMES}
    return RecordingEngine(cfg, ex, {n: torch.ones(4) for n in EXPERT_NAMES}, FakeSystem(), "aegis-v3-aug-srm", "cpu")


@pytest.fixture()
def client(lab_cfg):
    return TestClient(create_app(engine=_engine(lab_cfg)))


def test_samples_schema_and_stable(client):
    r1 = client.get("/api/lab/samples")
    assert r1.status_code == 200
    s = r1.json()["samples"]
    assert [x["id"] for x in s] == [f"s{i}" for i in range(1, 9)]
    assert sorted(x["truth"] for x in s) == ["AI"] * 4 + ["Real"] * 4
    assert sorted(x["source"] for x in s) == ["adm"] * 2 + ["real"] * 4 + ["sd_v1_4"] * 2
    for x in s:
        assert set(x) == {"id", "truth", "source", "thumb"} and x["thumb"].startswith("data:image/jpeg;base64,")
        im = Image.open(io.BytesIO(base64.b64decode(x["thumb"].split(",", 1)[1])))
        assert max(im.size) <= 160
    assert client.get("/api/lab/samples").json() == r1.json()


def test_samples_only_clean_test_split(lab_cfg):
    man = SplitManifest.load(lab_cfg["paths"]["manifest"])
    test_paths = {str(man.abs_path(r)) for r in man.by_split(SPLIT_TEST, clean_only=True)}
    lab = Lab(lab_cfg)
    lab.samples()
    assert {str(p) for p in lab._paths.values()} <= test_paths


def test_run_schema_and_order(client):
    r = client.post("/api/lab/run", json={"sample_id": "s1"})
    assert r.status_code == 200
    j = r.json()
    assert j["sample_id"] == "s1" and j["truth"] in ("Real", "AI") and j["model_version"] == "aegis-v3-aug-srm"
    assert isinstance(j["latency_ms"], int)
    assert [v["key"] for v in j["variants"]] == ["clean", "blur_s1.5", "blur_s2.5", "jpeg_q40"]
    assert [v["title"] for v in j["variants"]] == ["Ảnh gốc", "Mờ nhẹ", "Mờ nặng", "Nén JPEG mạnh"]
    for v in j["variants"]:
        assert set(v) == {"key", "title", "thumb", "label", "p_ai", "confidence", "warning_flag", "flag_reasons",
                          "experts", "quality"}
        assert set(v["experts"]) == set(EXPERT_NAMES)
        assert set(v["experts"]["clip"]) == {"p_ai", "weight"}
        assert v["thumb"].startswith("data:image/")


def test_variants_match_apply_degradation(lab_cfg):
    eng = _engine(lab_cfg)
    c = TestClient(create_app(engine=eng))
    assert c.post("/api/lab/run", json={"sample_id": "s5"}).status_code == 200
    clean, blur1, blur2, jpg = eng.seen
    assert clean.shape == (224, 224, 3)
    specs = {s.name: s for s in fixed_test_specs(lab_cfg)["S3"] + fixed_test_specs(lab_cfg)["S2"]}
    base = Image.fromarray(clean)
    assert np.array_equal(blur1, np.asarray(apply_degradation(base, specs["blur_k11_s1.5"])))
    assert np.array_equal(blur2, np.asarray(apply_degradation(base, specs["blur_k17_s2.5"])))
    assert np.array_equal(jpg, np.asarray(apply_degradation(base, specs["jpeg_q40"])))
    # first variant is the stored normalized image, untouched (no resize / crop)
    lab = Lab(lab_cfg)
    lab.samples()
    assert np.array_equal(clean, np.asarray(Image.open(lab._paths["s5"]).convert("RGB")))
    assert len(VARIANTS) == 4


def test_unknown_sample_404(client):
    for body in ({"sample_id": "s99"}, {"sample_id": 3}, {}):
        r = client.post("/api/lab/run", json=body)
        assert r.status_code == 404 and r.json()["error"] == "unknown_sample"
    r = client.post("/api/lab/run", content=b"not json", headers={"content-type": "application/json"})
    assert r.status_code == 404 and r.json()["error"] == "unknown_sample"


def test_lab_unavailable_503(lab_cfg, tmp_path):
    cfg = dict(lab_cfg)
    cfg["paths"] = dict(cfg["paths"], manifest=str(tmp_path / "nope" / "manifest.json"))
    c = TestClient(create_app(engine=_engine(cfg)))
    for r in (c.get("/api/lab/samples"), c.post("/api/lab/run", json={"sample_id": "s1"})):
        assert r.status_code == 503 and r.json()["error"] == "lab_unavailable"
        assert "Lab" in r.json()["message"]


def test_lab_unavailable_missing_images(tmp_path):
    d = tmp_path / "noimg"
    d.mkdir()
    cfg = dict(load_config())
    cfg["paths"] = dict(cfg["paths"], manifest=str(_build_manifest(d, with_images=False)))
    r = TestClient(create_app(engine=_engine(cfg))).get("/api/lab/samples")
    assert r.status_code == 503 and r.json()["error"] == "lab_unavailable"


def test_shared_rate_limit(lab_cfg):
    c = TestClient(create_app(engine=_engine(lab_cfg), rate_limit=3))
    assert c.get("/api/lab/samples").status_code == 200
    assert c.post("/api/lab/run", json={"sample_id": "s1"}).status_code == 200
    assert c.get("/api/health").status_code == 200                      # health is not limited
    r = c.post("/api/predict", files={"file": ("a.png", b"x", "application/octet-stream")})
    assert r.status_code == 400
    r = c.post("/api/lab/run", json={"sample_id": "s1"})
    assert r.status_code == 429 and r.json()["error"] == "rate_limited"


def test_not_ready_without_engine():
    c = TestClient(create_app(engine_factory=None))
    r = c.post("/api/lab/run", json={"sample_id": "s1"})
    assert r.status_code == 503 and r.json()["error"] == "not_ready"


def test_no_new_files(client, lab_cfg, tmp_path):
    def snap():
        out = set()
        for root, _, fs in os.walk(tmp_path):
            out |= {os.path.join(root, f) for f in fs}
        return out
    before = snap()
    client.get("/api/lab/samples")
    client.post("/api/lab/run", json={"sample_id": "s2"})
    assert snap() == before


def test_no_per_image_data_in_logs(client, caplog):
    import logging
    with caplog.at_level(logging.DEBUG):
        client.get("/api/lab/samples")
        client.post("/api/lab/run", json={"sample_id": "s2"})
    text = " ".join(r.getMessage() for r in caplog.records)
    assert ".png" not in text and "p_ai" not in text and "adm" not in text
