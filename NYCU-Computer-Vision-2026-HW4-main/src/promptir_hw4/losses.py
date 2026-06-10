from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import ssim


def gradient_map(x: torch.Tensor) -> torch.Tensor:
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return torch.abs(dx) + torch.abs(dy)


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, x, y):
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps2)
        return torch.mean(loss)


def charbonnier_per_sample(x: torch.Tensor, y: torch.Tensor, eps2: float = 1e-6) -> torch.Tensor:
    loss = torch.sqrt((x - y).pow(2) + eps2)
    return loss.flatten(1).mean(dim=1)


def gradient_loss_per_sample(pred: torch.Tensor, target: torch.Tensor, eps2: float = 1e-6) -> torch.Tensor:
    return charbonnier_per_sample(gradient_map(pred), gradient_map(target), eps2)


def fft_residual_loss_per_sample(pred_res: torch.Tensor, target_res: torch.Tensor) -> torch.Tensor:
    pred_fft = torch.fft.rfft2(pred_res.float(), norm="ortho")
    target_fft = torch.fft.rfft2(target_res.float(), norm="ortho")
    h, w = pred_res.shape[-2:]
    fy = torch.fft.fftfreq(h, device=pred_res.device).abs()[:, None]
    fx = torch.fft.rfftfreq(w, device=pred_res.device).abs()[None, :]
    radius = torch.sqrt(fy.pow(2) + fx.pow(2))
    mask = (radius >= 0.18).to(pred_fft.real.dtype)
    diff = (pred_fft - target_fft).abs() * mask
    return torch.log1p(diff).flatten(1).mean(dim=1).to(pred_res.dtype)


class RestorationLoss(nn.Module):
    def __init__(
        self,
        l1_weight: float = 1.0,
        ssim_weight: float = 0.2,
        edge_weight: float = 0.05,
        residual_weight: float = 0.35,
        freq_residual_weight: float = 0.08,
        snow_weight: float = 1.4,
        hard_weight: float = 0.3,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight
        self.residual_weight = residual_weight
        self.freq_residual_weight = freq_residual_weight
        self.snow_weight = snow_weight
        self.hard_weight = hard_weight
        self.charbonnier = CharbonnierLoss()

    def _sample_weights(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        inp: torch.Tensor | None,
        kind: list[str] | tuple[str, ...] | None,
    ) -> torch.Tensor:
        weights = torch.ones(pred.size(0), device=pred.device, dtype=pred.dtype)
        if kind is not None and self.snow_weight != 1.0:
            snow = torch.tensor([1.0 if item == "snow" else 0.0 for item in kind], device=pred.device, dtype=pred.dtype)
            weights = weights * (1.0 + snow * (self.snow_weight - 1.0))
        if inp is not None and self.hard_weight > 0:
            with torch.no_grad():
                mse = (inp - target).pow(2).flatten(1).mean(dim=1)
                relative = mse / mse.mean().clamp_min(1e-8)
                hard = (1.0 + self.hard_weight * (relative - 1.0)).clamp(0.75, 1.75)
            weights = weights * hard.to(weights.dtype)
        return weights / weights.mean().clamp_min(1e-8)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        inp: torch.Tensor | None = None,
        kind: list[str] | tuple[str, ...] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        weights = self._sample_weights(pred, target, inp, kind)
        l1_each = charbonnier_per_sample(pred, target)
        ssim_each = 1.0 - ssim(pred, target)
        edge_each = gradient_loss_per_sample(pred, target)

        residual_each = pred.new_zeros(pred.size(0))
        freq_each = pred.new_zeros(pred.size(0))
        if inp is not None:
            pred_res = pred - inp
            target_res = target - inp
            residual_each = charbonnier_per_sample(pred_res, target_res)
            freq_each = fft_residual_loss_per_sample(pred_res, target_res)

        l1 = (l1_each * weights).mean()
        ssim_loss = (ssim_each * weights).mean()
        edge = (edge_each * weights).mean()
        residual = (residual_each * weights).mean()
        freq_residual = (freq_each * weights).mean()
        loss = (
            self.l1_weight * l1
            + self.ssim_weight * ssim_loss
            + self.edge_weight * edge
            + self.residual_weight * residual
            + self.freq_residual_weight * freq_residual
        )
        return loss, {
            "l1": l1.detach(),
            "ssim_loss": ssim_loss.detach(),
            "edge": edge.detach(),
            "residual": residual.detach(),
            "freq_residual": freq_residual.detach(),
            "sample_weight": weights.detach().mean(),
        }
