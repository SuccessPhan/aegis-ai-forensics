"""Serving API tests with fake experts / fake AegisSystem (no weights, no GPU)."""
import io
import logging
import tempfile

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from src.core.config import load_config
from src.core.contracts import EXPERT_NAMES, FLAG_CONFLICT, FLAG_OOD, FLAG_UNCERTAIN, QUALITY_FEATURE_NAMES
from src.data.normalize import normalize_image
from src.serving.app import MAX_BYTES, create_app
from src.serving.engine import Engine, decode_image, map_flag_reasons, preprocess_image


class FakeExpert:
    def __init__(self, name):
        self.name = name
        self.shapes = []

    def preprocess(self, images):
        x = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float() / 255.0
        self.shapes.append(tuple(x.shape))
        return x

    def extract_features(self, x):
        return x.mean(dim=(2, 3)).repeat(1, 2)[:, :4] + 0.1

    def predict_logits(self, f):
        return f.sum(dim=1) - 1.0

    def features_and_logits(self, x):
        f = self.extract_features(x)
        return f, self.predict_logits(f)


class FakeSystem:
    metadata = {}

    def predict_one(self, logits, cosdist, quality):
        assert np.asarray(logits).shape == (3,) and np.asarray(cosdist).shape == (3,)
        assert np.asarray(quality).shape == (3,)
        return {"p_ai": 0.87, "label": "AI", "confidence": 0.87, "warning_flag": True,
                "flag_reasons": [FLAG_CONFLICT, FLAG_UNCERTAIN, FLAG_OOD],
                "weights": {"clip": 0.55, "dinov2": 0.25, "freq": 0.20},
                "p_cal": {"clip": 0.91, "dinov2": 0.72, "freq": 0.95},
                "ood_score": 0.21, "ood_threshold": 0.34}


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture()
def engine(cfg):
    ex = {n: FakeExpert(n) for n in EXPERT_NAMES}
    cents = {n: torch.ones(4) for n in EXPERT_NAMES}
    return Engine(cfg, ex, cents, FakeSystem(), "aegis-v3-aug-srm", "cpu")


@pytest.fixture()
def client(engine):
    return TestClient(create_app(engine=engine))


def _img(w=300, h=260, fmt="PNG", seed=0):
    a = np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)
    b = io.BytesIO()
    Image.fromarray(a).save(b, format=fmt)
    return b.getvalue()


def _post(client, data, name="secret_name_photo.png", headers=None):
    return client.post("/api/predict", files={"file": (name, data, "application/octet-stream")}, headers=headers)


def test_health_ok(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_version": "aegis-v3-aug-srm", "device": "cpu", "queue": 0}


def test_health_loading_and_predict_not_ready():
    app = create_app(engine=None)
    c = TestClient(app)
    assert c.get("/api/health").json()["status"] == "loading"
    assert c.get("/api/health").status_code == 503
    r = _post(c, _img())
    assert r.status_code == 503 and r.json()["error"] == "not_ready"


def test_predict_schema(client):
    r = _post(client, _img())
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["label"] == "AI" and j["p_ai"] == 0.87 and j["confidence"] == 0.87
    assert j["warning_flag"] is True and j["flag_reasons"] == ["conflict", "uncertain", "ood"]
    assert set(j["experts"]) == {"clip", "dinov2", "freq"}
    for v in j["experts"].values():
        assert set(v) == {"p_ai", "weight"}
    assert set(j["quality"]) == set(QUALITY_FEATURE_NAMES)
    assert all(isinstance(v, float) for v in j["quality"].values())
    assert j["ood_score"] == 0.21 and j["ood_threshold"] == 0.34
    assert j["model_version"] == "aegis-v3-aug-srm" and isinstance(j["latency_ms"], int)


def test_flag_mapping_uses_constants():
    assert map_flag_reasons([FLAG_CONFLICT, FLAG_UNCERTAIN, FLAG_OOD]) == ["conflict", "uncertain", "ood"]
    assert map_flag_reasons([]) == []


def test_bad_image(client):
    r = _post(client, b"this is not an image")
    assert r.status_code == 400 and r.json()["error"] == "bad_image"
    assert "message" in r.json()
    r = client.post("/api/predict", data={"x": "1"})            # no file field
    assert r.status_code == 400 and r.json()["error"] == "bad_image"
    r = _post(client, _img(fmt="BMP"))                            # unsupported format
    assert r.status_code == 400 and r.json()["error"] == "bad_image"


def test_too_small(client):
    r = _post(client, _img(300, 200))
    assert r.status_code == 400 and r.json()["error"] == "too_small"


def test_too_large(client):
    r = _post(client, b"\0" * (MAX_BYTES + 1))
    assert r.status_code == 413 and r.json()["error"] == "too_large"


def test_rate_limit_and_cf_header(engine):
    c = TestClient(create_app(engine=engine))
    img = _img(240, 240)
    codes = [_post(c, img).status_code for _ in range(20)]
    assert codes == [200] * 20
    r = _post(c, img)
    assert r.status_code == 429 and r.json()["error"] == "rate_limited"
    assert _post(c, img, headers={"CF-Connecting-IP": "1.2.3.4"}).status_code == 200   # other client IP


def test_queue_full_503(engine):
    c = TestClient(create_app(engine=engine, max_waiting=0))
    c.app.state.pending = 1
    r = _post(c, _img())
    assert r.status_code == 503 and r.json()["error"] == "not_ready"


def test_no_files_and_no_filename_in_logs(client, tmp_path, monkeypatch):
    tmp, cwd = tmp_path / "tmp", tmp_path / "cwd"
    tmp.mkdir()
    cwd.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp))
    monkeypatch.chdir(cwd)
    stream = io.StringIO()
    h = logging.StreamHandler(stream)
    lg = logging.getLogger("aegis")
    lg.addHandler(h)
    try:
        assert _post(client, _img(), name="secret_name_photo.png").status_code == 200
        assert _post(client, b"junk", name="secret_name_photo.png").status_code == 400
    finally:
        lg.removeHandler(h)
    assert list(tmp.iterdir()) == [] and list(cwd.iterdir()) == []
    out = stream.getvalue()
    assert "secret_name_photo" not in out and "status=200" in out and "0.87" not in out


def test_preprocess_matches_normalize_image(cfg, tmp_path):
    ccfg = load_config()
    assert ccfg["data"]["normalize"]["mode"] == "resize_crop"
    for (w, h, fmt) in [(300, 260, "PNG"), (500, 333, "JPEG"), (230, 640, "PNG")]:
        data = _img(w, h, fmt)
        src = tmp_path / f"s_{w}x{h}.{fmt.lower()}"
        src.write_bytes(data)
        dst = tmp_path / f"d_{w}x{h}.png"
        normalize_image(src, dst, is_real=True, cfg={**cfg, "data": {**cfg["data"], "normalize": {
            **cfg["data"]["normalize"], "jpeg_harmonize": False}}})
        ref = np.asarray(Image.open(dst).convert("RGB"))
        got = preprocess_image(decode_image(data), cfg)
        assert got.shape == (224, 224, 3) and got.dtype == np.uint8
        assert np.array_equal(got, ref)


def test_exif_orientation_and_first_frame():
    a = np.random.default_rng(1).integers(0, 256, (240, 400, 3), dtype=np.uint8)
    im = Image.fromarray(a)
    ex = Image.Exif()
    ex[0x0112] = 6                                   # rotate 270 -> displayed 400 tall x 240 wide
    b = io.BytesIO()
    im.save(b, format="JPEG", exif=ex)
    assert decode_image(b.getvalue()).size == (240, 400)
    frames = [Image.fromarray(a), Image.fromarray(255 - a)]
    b = io.BytesIO()
    frames[0].save(b, format="WEBP", save_all=True, append_images=frames[1:], lossless=True)
    assert decode_image(b.getvalue()).size == (400, 240)
