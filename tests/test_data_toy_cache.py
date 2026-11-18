import json

from src.data.toy import load_or_build_toy, toy_cache_key


def _small(cfg):
    c = json.loads(json.dumps(cfg))
    c["toy"]["counts"] = {"adm": 1, "sd_v1_4": 1, "midjourney": 2, "glide": 2, "real": 4}
    c["degradation"]["val"]["variants_per_source"] = 0
    c["degradation"]["train_aug"]["variants_per_source"] = 0
    c["degradation"]["test"]["s2_jpeg_quality"] = []
    c["degradation"]["test"]["s3_blur"] = []
    return c


def test_cache_reused_then_rebuilt_on_key_change_or_corruption(tmp_path, cfg):
    c = _small(cfg)
    m1 = load_or_build_toy(tmp_path, c)
    stamp = (tmp_path / "manifest.json").stat().st_mtime_ns
    m2 = load_or_build_toy(tmp_path, c)                       # same key -> reuse, no rewrite
    assert (tmp_path / "manifest.json").stat().st_mtime_ns == stamp
    assert [r.sha256 for r in m1.records] == [r.sha256 for r in m2.records]
    c2 = json.loads(json.dumps(c))
    c2["seed"] = c["seed"] + 1                                # key change -> rebuild
    assert toy_cache_key(c) != toy_cache_key(c2)
    m3 = load_or_build_toy(tmp_path, c2)
    assert [r.sha256 for r in m3.records] != [r.sha256 for r in m1.records]
    m3.abs_path(m3.records[0]).unlink()                       # missing file -> rebuild, no failure
    (tmp_path / 'manifest.json').write_text('{broken')        # ...and corrupt manifest too
    m4 = load_or_build_toy(tmp_path, c2)
    assert m4.abs_path(m4.records[0]).exists()
