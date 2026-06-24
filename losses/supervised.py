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
    # Variance as E[x^2] - E[x]^2 is unstable in float16 under CUDA AMP.
    # Keep this loss in float32 even when the model forward uses mixed precision.
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        prediction = prediction.float()
        target = target.float()
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
        sigma_x = (
            F.conv2d(prediction.square(), window, padding=padding, groups=channels)
            - mu_x_sq
        ).clamp_min(0.0)
        sigma_y = (
            F.conv2d(target.square(), window, padding=padding, groups=channels)
            - mu_y_sq
        ).clamp_min(0.0)
        sigma_xy = (
            F.conv2d(prediction * target, window, padding=padding, groups=channels)
            - mu_xy
        )
        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2
        numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
        denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x + sigma_y + c2)
        ssim_map = numerator / denominator.clamp_min(torch.finfo(torch.float32).tiny)
        return ssim_map.clamp(-1.0, 1.0).mean()


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


class DarkRegionSmoothnessLoss(nn.Module):
    """Suppress high-frequency enhanced-image variation only in dark input areas."""

    def __init__(self, threshold: float = 0.25) -> None:
        super().__init__()
        self.threshold = float(threshold)

    def forward(self, low: torch.Tensor, enhanced: torch.Tensor) -> torch.Tensor:
        if low.shape != enhanced.shape:
            raise ValueError("DarkRegionSmoothnessLoss expects low/enhanced with same shape")
        gray = low.mean(dim=1, keepdim=True)
        dark_mask = (gray < self.threshold).to(dtype=enhanced.dtype).detach()
        loss = enhanced.new_zeros(())
        if enhanced.shape[-1] > 1:
            dx = torch.abs(enhanced[:, :, :, 1:] - enhanced[:, :, :, :-1])
            mask_x = dark_mask[:, :, :, 1:] * dark_mask[:, :, :, :-1]
            loss = loss + torch.mean(dx * mask_x)
        if enhanced.shape[-2] > 1:
            dy = torch.abs(enhanced[:, :, 1:, :] - enhanced[:, :, :-1, :])
            mask_y = dark_mask[:, :, 1:, :] * dark_mask[:, :, :-1, :]
            loss = loss + torch.mean(dy * mask_y)
        return loss


class ChromaticityLoss(nn.Module):
    """Match RGB proportions while largely ignoring brightness differences."""

    def __init__(self, eps: float = 1e-4) -> None:
        super().__init__()
        self.eps = eps

    def _chromaticity(self, image: torch.Tensor) -> torch.Tensor:
        return image / image.sum(dim=1, keepdim=True).clamp_min(self.eps)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction_chroma = self._chromaticity(prediction)
        target_chroma = self._chromaticity(target)

        # Very dark pixels have unstable RGB ratios, so weight them by target luminance.
        weight = target.mean(dim=1, keepdim=True).detach()
        pixel_error = torch.abs(prediction_chroma - target_chroma)
        weighted_pixel = (pixel_error * weight).sum() / (
            weight.sum() * prediction.shape[1] + self.eps
        )

        prediction_mean = prediction.mean(dim=(2, 3))
        target_mean = target.mean(dim=(2, 3))
        prediction_global = prediction_mean / prediction_mean.sum(
            dim=1, keepdim=True
        ).clamp_min(self.eps)
        target_global = target_mean / target_mean.sum(
            dim=1, keepdim=True
        ).clamp_min(self.eps)
        global_error = F.l1_loss(prediction_global, target_global)
        return weighted_pixel + global_error
