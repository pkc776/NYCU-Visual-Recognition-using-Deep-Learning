from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mse = F.mse_loss(pred.clamp(0, 1), target.clamp(0, 1), reduction="none")
    mse = mse.flatten(1).mean(dim=1)
    return 10.0 * torch.log10(1.0 / (mse + eps))


def _gaussian_window(size: int, sigma: float, channels: int, device: torch.device) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window = (g[:, None] @ g[None, :]).expand(channels, 1, size, size).contiguous()
    return window


def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    channels = pred.shape[1]
    window = _gaussian_window(window_size, 1.5, channels, pred.device)
    padding = window_size // 2
    mu1 = F.conv2d(pred, window, padding=padding, groups=channels)
    mu2 = F.conv2d(target, window, padding=padding, groups=channels)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu12 = mu1 * mu2
    sigma1_sq = F.conv2d(pred * pred, window, padding=padding, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=padding, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=padding, groups=channels) - mu12
    c1 = 0.01**2
    c2 = 0.03**2
    score = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return score.flatten(1).mean(dim=1)


class AverageMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        if math.isfinite(value):
            self.total += value * n
            self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)
