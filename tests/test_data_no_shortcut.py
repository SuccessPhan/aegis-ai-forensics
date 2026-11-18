"""test_no_dataset_shortcut: negative control (label independent of image stats -> chance) and positive control
(fake has different JPEG history -> audit must detect). Synthetic images, 96x96 for speed."""
import numpy as np
from PIL import Image

from src.data.normalize import jpeg_roundtrip
from src.data.shortcut_audit import audit_images, audit_manifest, main
from src.core.contracts import SplitManifest


def _synthetic(rng, size=96):
    small = rng.random((8, 8, 3)) * 255
    a = np.asarray(Image.fromarray(small.astype(np.uint8)).resize((size, size), Image.BICUBIC), dtype=np.float64)
    a += rng.normal(0, rng.uniform(2, 8), a.shape)
    return np.clip(a, 0, 255).astype(np.uint8)


def _balanced_labels(rng, n):
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    return rng.permutation(y)


def test_no_dataset_shortcut_negative_control(cfg):
    rng = np.random.default_rng(123)
    n = 1200
    arrays = [_synthetic(rng) for _ in range(n)]
    labels = _balanced_labels(rng, n)                 # independent of content
    res = audit_images(arrays, labels, cfg)
    assert abs(res.accuracy - 0.5) <= 0.05, res
    assert abs(res.auroc - 0.5) <= 0.05, res
    assert res.passed


def test_no_dataset_shortcut_positive_control_detects_leak(cfg):
    rng = np.random.default_rng(7)
    n = 300
    labels = _balanced_labels(rng, n)
    arrays = []
    for y in labels:
        img = Image.fromarray(_synthetic(rng))
        # "real" carries mild JPEG history (Q=90), "fake" a different one (Q=50)
        arrays.append(np.asarray(jpeg_roundtrip(img, 50 if y == 1 else 90)))
    res = audit_images(arrays, labels, cfg)
    assert res.auroc > 0.8, res
    assert not res.passed


def test_audit_runs_on_toy_manifest_and_cli(toy_manifest, tmp_path, cfg):
    res = audit_manifest(toy_manifest, cfg)
    assert 0 <= res.accuracy <= 1 and 0 <= res.auroc <= 1 and res.n > 0
    toy_manifest.save(tmp_path / "manifest.json")
    assert main(["--manifest", str(tmp_path / "manifest.json")]) in (0, 1)
    assert SplitManifest.load(tmp_path / "manifest.json").root == toy_manifest.root


def test_audit_full_reported_and_has_unseen(toy_manifest, cfg):
    import copy
    from src.data.shortcut_audit import audit_full
    full = audit_full(toy_manifest, cfg)
    assert full["status"] == "reported" and isinstance(full["passed"], bool)
    pg = full["per_generator"]
    assert {g for g, r in pg.items() if r["seen"]} == {"adm", "sd_v1_4"}
    assert {g for g, r in pg.items() if r["seen"] is False} == {"midjourney", "glide"}
    c = copy.deepcopy(cfg)
    c["shortcut_audit"]["enforce"] = True
    assert audit_full(toy_manifest, c)["status"] in ("pass", "fail")
