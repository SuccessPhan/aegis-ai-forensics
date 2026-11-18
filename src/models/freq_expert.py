"""Expert 3 - Frequency: log-magnitude FFT (or SRM residuals) -> 4-block CNN trained from scratch.

The only expert trained from scratch (~1.17M params with channels [32,64,128,256]).
Images are NEVER resized: the network is fully convolutional + global average pooling, so it accepts any
H x W and keeps high-frequency artifacts intact.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.core.logger import get_logger
from src.models.base import BaseExpert, images_to_tensor, load_image_array

log = get_logger(__name__)

# SRM high-pass kernels (3-kernel subset: Fridrich & Kodovsky style residual filters).
_SRM = [
    np.array([[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]]) / 4.0,
    np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2],
              [-1, 2, -2, 2, -1]]) / 12.0,
    np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]) / 2.0,
]
SRM_SCALE = 10.0  # fixed gain so residuals (~0.01-0.1 on [0,1] images) reach O(1); not data-fitted


def log_magnitude_fft(x: torch.Tensor) -> torch.Tensor:
    """(n,3,H,W) in [0,1] -> per-channel centred log1p|FFT|, standardised per sample/channel."""
    spec = torch.fft.fftshift(torch.fft.fft2(x - x.mean(dim=(-2, -1), keepdim=True)), dim=(-2, -1))
    m = torch.log1p(spec.abs())
    mu = m.mean(dim=(-2, -1), keepdim=True)
    sd = m.std(dim=(-2, -1), keepdim=True) + 1e-6
    return (m - mu) / sd


def srm_residual(x: torch.Tensor) -> torch.Tensor:
    """(n,3,H,W) -> (n,3,H,W) residuals of the grayscale image under 3 fixed SRM kernels."""
    gray = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
    k = torch.tensor(np.stack(_SRM), dtype=x.dtype, device=x.device).unsqueeze(1)  # (3,1,5,5)
    return F.conv2d(gray, k, padding=2) * SRM_SCALE


def _block(cin: int, cout: int, pool: bool) -> nn.Sequential:
    layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
              nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class FreqCNN(nn.Module):
    def __init__(self, channels: Sequence[int], in_ch: int = 3) -> None:
        super().__init__()
        chans = [in_ch, *channels]
        self.blocks = nn.Sequential(*[_block(chans[i], chans[i + 1], pool=i < len(channels) - 1)
                                      for i in range(len(channels))])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[-1], 1)

    def features(self, z: torch.Tensor) -> torch.Tensor:
        return self.pool(self.blocks(z)).flatten(1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(z)).squeeze(-1)


class FreqExpert(BaseExpert):
    name = "freq"

    def __init__(self, cfg: dict, input_mode: str | None = None) -> None:
        super().__init__()
        fc = cfg["experts"]["freq"]
        self.cfg = cfg
        self.input_mode = input_mode or fc["input"]
        if self.input_mode not in ("log_magnitude_fft", "srm"):
            raise ValueError(f"unknown freq input '{self.input_mode}'")
        self.net = FreqCNN(fc["channels"])
        self.feature_dim = int(fc["channels"][-1])
        self.model_id = f"freq_cnn_{self.input_mode}"

    def n_params(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    def preprocess(self, images) -> torch.Tensor:
        return images_to_tensor(images)  # [0,1] floats, original size kept

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        return log_magnitude_fft(x) if self.input_mode == "log_magnitude_fft" else srm_residual(x)

    @torch.no_grad()
    def extract_features(self, batch: torch.Tensor) -> torch.Tensor:
        """Global-average-pooled vector (n, 256) before the classifier (used for cosine distance)."""
        self.net.eval()
        return self.net.features(self._transform(batch.to(self.device))).float().cpu()

    def predict_logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.net.fc(features.to(self.device).float()).reshape(-1)

    @torch.no_grad()
    def features_and_logits(self, batch: torch.Tensor):
        self.net.eval()
        f = self.net.features(self._transform(batch.to(self.device)))
        return f.float().cpu(), self.net.fc(f).reshape(-1).float().cpu()


class ImagePathDataset(Dataset):
    def __init__(self, paths: Sequence[str], labels: Sequence[int]) -> None:
        self.paths, self.labels = list(paths), list(labels)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        x = images_to_tensor([load_image_array(self.paths[i])])[0]
        return x, torch.tensor(float(self.labels[i]))


def train_freq(expert: FreqExpert, dataset: Dataset, cfg: dict, device: str | torch.device = "cpu") -> list:
    """Train the Frequency CNN from scratch on (clean) train_experts. Returns per-epoch mean loss."""
    tc = cfg["experts"]["freq"]["train"]
    torch.manual_seed(int(cfg.get("seed", 0)))
    expert.to(device)
    expert.net.train()
    opt = torch.optim.Adam(expert.net.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    loader = DataLoader(dataset, batch_size=tc["batch_size"], shuffle=True)
    history = []
    for ep in range(int(tc["epochs"])):
        tot, cnt = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = F.binary_cross_entropy_with_logits(expert.net(expert._transform(x)), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(y)
            cnt += len(y)
        history.append(tot / max(cnt, 1))
        log.info("freq epoch %d/%d loss=%.4f", ep + 1, tc["epochs"], history[-1])
    expert.net.eval()
    return history
