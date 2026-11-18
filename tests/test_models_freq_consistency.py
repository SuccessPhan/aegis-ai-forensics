"""Regression: Expert 3 train path and inference path (preprocess -> features_and_logits) must agree for both inputs."""
import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset

from src.core.config import deep_merge
from src.models.freq_expert import FreqExpert, train_freq


class _Synth(Dataset):
    """Real = smooth random image; fake = same + checkerboard/high-frequency noise (separable by frequency)."""

    def __init__(self, n=64, size=32, seed=0):
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[:size, :size]
        checker = ((yy + xx) % 2) * 2.0 - 1.0
        self.x, self.y = [], []
        for i in range(n):
            base = rng.uniform(0.2, 0.8, (3, 1, 1)) + 0.1 * np.sin(xx / rng.uniform(3, 8))[None]
            lab = i % 2
            img = base + 0.01 * rng.standard_normal((3, size, size))
            if lab:
                img = img + 0.03 * checker[None] + 0.02 * rng.standard_normal((3, size, size))
            self.x.append(torch.from_numpy(np.clip(img, 0, 1)).float())
            self.y.append(float(lab))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], torch.tensor(self.y[i])


@pytest.mark.parametrize("inp", ["log_magnitude_fft", "srm"])
def test_freq_train_infer_consistent(cfg, inp):
    c = deep_merge(cfg, {"experts": {"freq": {"input": inp, "channels": [4, 8, 8, 8],
                                              "train": {"epochs": 12, "batch_size": 16, "lr": 0.003}}}})
    ds = _Synth()
    e = FreqExpert(c)
    train_freq(e, ds, c)
    e.train(True)  # inference must not depend on module mode
    scores = []
    for x, _ in ds:
        _, lg = e.features_and_logits(e.preprocess((x.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)[None]))
        scores.append(float(lg[0]))
    assert roc_auc_score(ds.y, scores) > 0.8


def test_freq_srm_bn_stats_match_final_weights(cfg):
    """After train_freq(srm) the BN running stats must be the exact cumulative stats of the final weights."""
    import copy

    from src.models.freq_expert import recalibrate_bn

    c = deep_merge(cfg, {"experts": {"freq": {"input": "srm", "channels": [4, 8, 8, 8],
                                              "train": {"epochs": 3, "batch_size": 16, "lr": 0.003}}}})
    ds = _Synth()
    e = FreqExpert(c)
    train_freq(e, ds, c)
    before = copy.deepcopy(e.net.state_dict())
    recalibrate_bn(e, ds, c)
    for k, v in e.net.state_dict().items():
        if "running_" in k:
            assert torch.allclose(v, before[k], rtol=1e-4, atol=1e-6), k
