import numpy as np
import pytest

from src.core.config import get, load_config, resolve_path
from src.core.contracts import (EXPERT_NAMES, GATING_DIM, GATING_FEATURE_NAMES, DegradationSpec, GatingInput,
                                GatingWeights, ImageRecord, QualityFeatures, SplitManifest)
from src.core.logger import get_logger


def test_gating_feature_order_is_fixed_12d():
    assert GATING_DIM == 12
    assert GATING_FEATURE_NAMES[:3] == ("jpeg_quality_est", "log_laplacian_var", "high_freq_energy_ratio")
    assert GATING_FEATURE_NAMES[3:6] == tuple(f"entropy_{e}" for e in EXPERT_NAMES)
    assert GATING_FEATURE_NAMES[6:9] == tuple(f"cosdist_{e}" for e in EXPERT_NAMES)
    assert GATING_FEATURE_NAMES[9:] == tuple(f"p_{e}" for e in EXPERT_NAMES)


def test_gating_input_roundtrip():
    gi = GatingInput(QualityFeatures(50.0, 2.0, 0.1), (0.1, 0.2, 0.3), (0.4, 0.5, 0.6), (0.7, 0.8, 0.9))
    v = gi.to_vector()
    assert v.shape == (12,) and v.dtype == np.float32
    np.testing.assert_allclose(v, [50, 2, 0.1, .1, .2, .3, .4, .5, .6, .7, .8, .9], rtol=1e-6)
    assert GatingInput.stack([gi, gi]).shape == (2, 12)
    with pytest.raises(ValueError):
        GatingInput.from_vector([0.0] * 11)


def test_gating_weights_validate():
    assert GatingWeights((0.05, 0.05, 0.90)).validate()
    assert not GatingWeights((0.0, 0.1, 0.9)).validate()
    assert not GatingWeights((0.5, 0.5, 0.5)).validate()


def test_manifest_json_roundtrip(tmp_path):
    rec = ImageRecord("ab" * 32, "images/a.png", 1, "adm", "adm:1", split="train_experts",
                      degradation=DegradationSpec(jpeg_quality=65, name="jpeg_q65", scenario="S2"))
    m = SplitManifest([rec], seed=3, root=str(tmp_path))
    m.save(tmp_path / "m.json")
    m2 = SplitManifest.load(tmp_path / "m.json")
    assert m2.records[0] == rec and not m2.records[0].is_clean


def test_config_values_from_brief(cfg):
    assert get(cfg, "data.generators.unseen") == ["midjourney", "glide"]
    assert get(cfg, "data.normalize.crop_size") == 224
    assert get(cfg, "data.normalize.jpeg_harmonize") is True
    assert get(cfg, "degradation.val.jpeg_quality_range") == [30, 95]
    assert get(cfg, "degradation.val.blur_sigma_range") == [0.5, 3.0]
    assert get(cfg, "degradation.test.s2_jpeg_quality") == [85, 65, 40]
    assert get(cfg, "gating.hidden_dim") == 32 and get(cfg, "gating.w_min") == 0.05 and get(cfg, "gating.scale") == 0.85
    assert get(cfg, "fewshot.n_values") == [5, 20, 100] and get(cfg, "fewshot.alpha_cap") == 0.6
    assert get(cfg, "shortcut_audit.tolerance") == 0.05
    assert get(cfg, "nope.missing", "d") == "d"
    assert resolve_path(cfg, "toy_dir").is_absolute()
    assert load_config(overrides={"seed": 7})["seed"] == 7


def test_logger_is_namespaced():
    assert get_logger("x").name == "aegis.x"
