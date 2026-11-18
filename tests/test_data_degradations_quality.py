import numpy as np
from PIL import Image

from src.data.degradations import (apply_degradation, auto_kernel, fixed_test_specs, gaussian_blur,
                                   gaussian_kernel_1d, sample_val_spec)
from src.data.normalize import estimate_quality_from_qtable, jpeg_roundtrip
from src.data.quality import (blockiness, extract_quality_features, high_freq_energy_ratio, jpeg_quality_est,
                              log_laplacian_var, to_gray)


def _img(seed=0, size=224):
    rng = np.random.default_rng(seed)
    small = rng.random((14, 14, 3)) * 255
    a = np.asarray(Image.fromarray(small.astype(np.uint8)).resize((size, size), Image.BICUBIC), dtype=np.float64)
    return np.clip(a + rng.normal(0, 6, a.shape), 0, 255).astype(np.uint8)


def test_fixed_test_levels_match_brief(cfg):
    s = fixed_test_specs(cfg)
    assert [x.jpeg_quality for x in s["S2"]] == [85, 65, 40]
    assert s["S3"][0].resize_factor == 0.5
    assert [(x.blur_kernel, x.blur_sigma) for x in s["S3"][1:]] == [(11, 1.5), (17, 2.5)]


def test_blur_uses_exact_kernel_size():
    for k, sg in ((11, 1.5), (17, 2.5)):
        ker = gaussian_kernel_1d(k, sg)
        assert ker.shape == (k,) and abs(ker.sum() - 1) < 1e-12
    arr = _img()
    # impulse response width check: a delta spreads over exactly k pixels (nonzero taps)
    delta = np.zeros((40, 40, 3), np.uint8)
    delta[20, 20] = 255
    out = gaussian_blur(delta, 11, 5.0)[20, :, 0]
    assert (np.nonzero(out)[0].min(), np.nonzero(out)[0].max()) == (15, 25)     # 11 taps
    spec = fixed_test_specs({"degradation": {"test": {"s2_jpeg_quality": [], "s3_resize": {"factor": .5}, "s3_blur": [{"kernel": 17, "sigma": 2.5}]}}})["S3"][1]
    a = np.asarray(apply_degradation(Image.fromarray(arr), spec))
    np.testing.assert_array_equal(a, gaussian_blur(arr, 17, 2.5))


def test_jpeg_degradation_has_requested_quality(tmp_path):
    img = Image.fromarray(_img())
    for q in (85, 65, 40):
        p = tmp_path / f"q{q}.jpg"
        # same encoder call path as degradations.jpeg_roundtrip
        img.save(p, format="JPEG", quality=q)
        with Image.open(p) as im:
            assert estimate_quality_from_qtable(im) == q
        assert jpeg_roundtrip(img, q).size == (224, 224)


def test_resize_down_up_keeps_size_and_blurs():
    arr = _img()
    out = np.asarray(apply_degradation(Image.fromarray(arr), fixed_test_specs({"degradation": {"test": {
        "s2_jpeg_quality": [], "s3_resize": {"factor": 0.5}, "s3_blur": []}}})["S3"][0]))
    assert out.shape == arr.shape
    assert log_laplacian_var(to_gray(out)) < log_laplacian_var(to_gray(arr))


def test_val_specs_within_ranges_and_continuous(cfg):
    rng = np.random.default_rng(0)
    qs, ss = [], []
    for i in range(300):
        s = sample_val_spec(rng, cfg, kind="jpeg" if i % 2 else "blur")
        if s.jpeg_quality is not None:
            assert 30 <= s.jpeg_quality <= 95
            qs.append(s.jpeg_quality)
        else:
            assert 0.5 <= s.blur_sigma <= 3.0 and s.blur_kernel % 2 == 1 and s.blur_kernel >= auto_kernel(s.blur_sigma) - 2
            ss.append(s.blur_sigma)
    assert len(set(qs)) > 30 and len(set(ss)) > 100


def test_quality_features_respond_to_degradations():
    arr = _img()
    img = Image.fromarray(arr)
    g0 = to_gray(arr)
    gq = {q: to_gray(np.asarray(jpeg_roundtrip(img, q))) for q in (95, 65, 30)}
    assert blockiness(g0) < 0.02
    assert jpeg_quality_est(gq[95]) > jpeg_quality_est(gq[65]) > jpeg_quality_est(gq[30])
    blurred = to_gray(gaussian_blur(arr, 11, 1.5))
    assert log_laplacian_var(blurred) < log_laplacian_var(g0)
    assert high_freq_energy_ratio(blurred) < high_freq_energy_ratio(g0)
    f = extract_quality_features(arr)
    assert f.to_array().shape == (3,) and 0 <= f.high_freq_energy_ratio <= 1
