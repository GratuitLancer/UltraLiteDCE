from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _gaussian_window(
    channels: int,
    window_size: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype)
    coords = coords - (window_size - 1) / 2.0
    gaussian = torch.exp(-(coords**2) / (2.0 * sigma**2))
    gaussian = gaussian / gaussian.sum()
    window_2d = gaussian[:, None] * gaussian[None, :]
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim_index(
    prediction: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("SSIM inputs must have the same shape")
    if prediction.ndim != 4:
        raise ValueError("SSIM inputs must have shape BxCxHxW")
    channels = prediction.shape[1]
    window = _gaussian_window(
        channels, window_size, sigma, prediction.device, prediction.dtype
    )
    padding = window_size // 2
    mu_x = F.conv2d(prediction, window, padding=padding, groups=channels)
    mu_y = F.conv2d(target, window, padding=padding, groups=channels)
    mu_x_sq = mu_x.square()
    mu_y_sq = mu_y.square()
    mu_xy = mu_x * mu_y
    sigma_x = F.conv2d(prediction.square(), window, padding=padding, groups=channels) - mu_x_sq
    sigma_y = F.conv2d(target.square(), window, padding=padding, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(prediction * target, window, padding=padding, groups=channels) - mu_xy
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x + sigma_y + c2)
    # C1*C2 is 9e-8 for data_range=1, smaller than float32 eps but still valid.
    # Using eps here would incorrectly reduce SSIM for identical black images.
    return (numerator / denominator.clamp_min(torch.finfo(prediction.dtype).tiny)).mean()


class SSIMLoss(nn.Module):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - ssim_index(prediction, target)


class EdgePreservationLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(-1, -2).contiguous()
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _gradient(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        luminance = (
            image[:, 0:1] * 0.299 + image[:, 1:2] * 0.587 + image[:, 2:3] * 0.114
        )
        return (
            F.conv2d(luminance, self.sobel_x, padding=1),
            F.conv2d(luminance, self.sobel_y, padding=1),
        )

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_x, pred_y = self._gradient(prediction)
        target_x, target_y = self._gradient(target)
        return F.l1_loss(pred_x, target_x) + F.l1_loss(pred_y, target_y)
