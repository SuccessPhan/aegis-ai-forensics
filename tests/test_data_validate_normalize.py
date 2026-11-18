import numpy as np
from PIL import Image

from src.data.normalize import (center_crop, crop_offset, estimate_quality_from_qtable, jpeg_roundtrip,
                                normalize_image, real_quality_pool)
from src.data.pipeline import RawItem, prepare_records
from src.data.quality import blockiness, to_gray
from src.data.validate import dedup_paths, dedup_records, sha256_file, validate_image


def _noise_img(rng, w, h):
    small = rng.random((9, 9, 3)) * 255
    return Image.fromarray(small.astype(np.uint8)).resize((w, h), Image.BICUBIC)


def test_validate_and_dedup(tmp_path):
    rng = np.random.default_rng(0)
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    img = _noise_img(rng, 240, 240)
    img.save(a)
    img.save(b)                                   # byte-identical duplicate
    _noise_img(rng, 240, 240).save(tmp_path / "c.png")
    (tmp_path / "bad.png").write_bytes(b"not an image")
    small = tmp_path / "small.png"
    _noise_img(rng, 100, 100).save(small)
    assert validate_image(a, 224)[0]
    assert not validate_image(tmp_path / "bad.png")[0]
    assert not validate_image(small, 224)[0]
    kept, dropped = dedup_paths([a, b, tmp_path / "c.png"])
    assert len(kept) == 2 and len(dropped) == 1 and sha256_file(a) == sha256_file(b)


def test_crop_offset_multiple_of_8_and_in_range():
    for length in range(224, 700, 7):
        o = crop_offset(length, 224, 8)
        assert o % 8 == 0 and 0 <= o <= length - 224
    assert crop_offset(301, 224) == 32            # (301-224)//2 = 38 -> 32
    img = Image.new("RGB", (301, 277))
    out, (l, t) = center_crop(img, 224, 8)
    assert out.size == (224, 224) and (l, t) == (32, 24)


def test_normalize_png_224_and_no_resize(tmp_path, cfg):
    rng = np.random.default_rng(1)
    src = tmp_path / "raw.png"
    raw = _noise_img(rng, 301, 277)
    raw.save(src)
    cfg2 = {**cfg, "data": {**cfg["data"], "normalize": {**cfg["data"]["normalize"], "jpeg_harmonize": False}}}
    dst = tmp_path / "out.png"
    prov = normalize_image(src, dst, is_real=False, cfg=cfg2)
    with Image.open(dst) as im:
        assert im.format == "PNG" and im.size == (224, 224)
        out = np.asarray(im)
    l, t = prov["crop_left"], prov["crop_top"]
    assert l % 8 == 0 and t % 8 == 0
    np.testing.assert_array_equal(out, np.asarray(raw)[t:t + 224, l:l + 224])   # pure crop, no resampling


def test_jpeg_harmonize_adds_jpeg_history_to_fake_only(tmp_path, cfg):
    rng = np.random.default_rng(2)
    src = tmp_path / "raw.png"
    _noise_img(rng, 256, 256).save(src)
    on = {**cfg, "data": {**cfg["data"], "normalize": {**cfg["data"]["normalize"], "jpeg_harmonize": True}}}
    off = {**cfg, "data": {**cfg["data"], "normalize": {**cfg["data"]["normalize"], "jpeg_harmonize": False}}}
    p_on = normalize_image(src, tmp_path / "on.png", is_real=False, cfg=on, quality_pool=[50])
    p_off = normalize_image(src, tmp_path / "off.png", is_real=False, cfg=off, quality_pool=[50])
    p_real = normalize_image(src, tmp_path / "real.png", is_real=True, cfg=on, quality_pool=[50])
    assert p_on["harmonize_quality"] == 50 and p_off["harmonize_quality"] is None and p_real["harmonize_quality"] is None
    g = lambda n: to_gray(np.asarray(Image.open(tmp_path / n)))  # noqa: E731
    assert blockiness(g("on.png")) > blockiness(g("off.png")) + 0.1


def test_qtable_quality_estimate_roundtrip(tmp_path):
    img = _noise_img(np.random.default_rng(3), 64, 64)
    for q in (30, 65, 85, 95):
        p = tmp_path / f"q{q}.jpg"
        img.save(p, format="JPEG", quality=q)
        with Image.open(p) as im:
            assert estimate_quality_from_qtable(im) == q
    assert real_quality_pool([tmp_path / "q30.jpg", tmp_path / "q85.jpg"]) == [30, 85]


def test_prepare_records_dedups_and_records_provenance(tmp_path, cfg):
    rng = np.random.default_rng(4)
    raw = tmp_path / "raw"
    raw.mkdir()
    img = _noise_img(rng, 260, 250)
    img.save(raw / "r1.jpg", quality=80)
    img.save(raw / "r1_copy.jpg", quality=80)       # duplicate bytes
    _noise_img(rng, 270, 260).save(raw / "f1.png")
    items = [RawItem(raw / "r1.jpg", "real", 0), RawItem(raw / "r1_copy.jpg", "real", 0), RawItem(raw / "f1.png", "adm", 1)]
    recs = prepare_records(items, tmp_path / "out", cfg)
    assert len(recs) == 2
    assert len({r.sha256 for r in recs}) == 2
    fake = next(r for r in recs if r.label == 1)
    assert fake.provenance["harmonize_quality"] is not None and "raw_sha256" in fake.provenance
    assert dedup_records(recs + recs)[1].__len__() == 2
