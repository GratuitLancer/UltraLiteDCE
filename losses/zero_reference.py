from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SpatialConsistencyLoss(nn.Module):
    def __init__(self, pool_size: int = 4) -> None:
        super().__init__()
        kernels = torch.tensor(
            [
                [[0, 0, 0], [-1, 1, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 1, -1], [0, 0, 0]],
                [[0, -1, 0], [0, 1, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 1, 0], [0, -1, 0]],
            ],
            dtype=torch.float32,
        ).unsqueeze(1)
        self.register_buffer("kernels", kernels)
        self.pool_size = pool_size

    def forward(self, original: torch.Tensor, enhanced: torch.Tensor) -> torch.Tensor:
        original_gray = original.mean(dim=1, keepdim=True)
        enhanced_gray = enhanced.mean(dim=1, keepdim=True)
        kernel = min(self.pool_size, original.shape[-2], original.shape[-1])
        original_pool = F.avg_pool2d(original_gray, kernel_size=kernel, stride=kernel)
        enhanced_pool = F.avg_pool2d(enhanced_gray, kernel_size=kernel, stride=kernel)
        original_grad = F.conv2d(original_pool, self.kernels, padding=1)
        enhanced_grad = F.conv2d(enhanced_pool, self.kernels, padding=1)
        return F.mse_loss(enhanced_grad, original_grad)


class ExposureControlLoss(nn.Module):
    def __init__(self, target: float = 0.6, patch_size: int = 16) -> None:
        super().__init__()
        self.target = target
        self.patch_size = patch_size

    def forward(self, enhanced: torch.Tensor) -> torch.Tensor:
        luminance = enhanced.mean(dim=1, keepdim=True)
        kernel = min(self.patch_size, enhanced.shape[-2], enhanced.shape[-1])
        patch_mean = F.avg_pool2d(luminance, kernel_size=kernel, stride=kernel)
        return torch.mean((patch_mean - self.target) ** 2)


class ColorConstancyLoss(nn.Module):
    def forward(self, enhanced: torch.Tensor) -> torch.Tensor:
        channel_means = enhanced.mean(dim=(2, 3))
        red, green, blue = channel_means.unbind(dim=1)
        return ((red - green) ** 2 + (red - blue) ** 2 + (green - blue) ** 2).mean()


class TotalVariationLoss(nn.Module):
    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        loss = curve.new_zeros(())
        if curve.shape[-2] > 1:
            loss = loss + torch.mean((curve[:, :, 1:, :] - curve[:, :, :-1, :]) ** 2)
        if curve.shape[-1] > 1:
            loss = loss + torch.mean((curve[:, :, :, 1:] - curve[:, :, :, :-1]) ** 2)
        return loss

