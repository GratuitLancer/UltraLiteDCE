from __future__ import annotations

import torch

from losses import ChromaticityLoss, CombinedLoss, DarkRegionSmoothnessLoss, ssim_index
from models import UltraLiteDCE


def _config() -> dict:
    return {
        "loss": {
            "spatial": 1.0,
            "exposure": 1.0,
            "color": 0.5,
            "curve_tv": 20.0,
            "l1": 1.0,
            "ssim": 0.2,
            "edge": 0.1,
            "chromaticity": 1.0,
            "dark_smooth": 0.2,
            "saturation": 0.1,
            "exposure_target": 0.6,
            "dark_threshold": 0.25,
            "saturation_threshold": 0.95,
        }
    }


def test_all_losses_are_finite_and_backward_works() -> None:
    model = UltraLiteDCE()
    criterion = CombinedLoss(_config())
    low = torch.rand(2, 3, 32, 32)
    high = torch.rand_like(low)
    enhanced, curve = model.enhance(low)
    total, components = criterion(low, enhanced, high, curve)
    assert set(components) == {
        "total",
        "spatial",
        "exposure",
        "color",
        "curve_tv",
        "l1",
        "ssim",
        "edge",
        "chromaticity",
        "dark_smooth",
        "saturation",
    }
    assert all(torch.isfinite(value) for value in components.values())
    total.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_ssim_numerical_stability() -> None:
    for value in (0.0, 0.5, 1.0):
        image = torch.full((1, 3, 9, 13), value)
        score = ssim_index(image, image)
        assert torch.isfinite(score)
        assert 0.999 <= score.item() <= 1.001
    first = torch.rand(1, 3, 16, 16)
    second = torch.rand_like(first)
    score = ssim_index(first, second)
    assert torch.isfinite(score)
    assert -1.0 <= score.item() <= 1.0


def test_ssim_half_precision_input_is_stable() -> None:
    first = torch.rand(2, 3, 32, 32, dtype=torch.float16)
    second = torch.rand_like(first)
    score = ssim_index(first, second)
    assert score.dtype == torch.float32
    assert torch.isfinite(score)
    assert -1.0 <= score.item() <= 1.0


def test_chromaticity_loss_detects_color_cast_not_brightness() -> None:
    loss = ChromaticityLoss()
    target = torch.rand(1, 3, 16, 16) * 0.8 + 0.1
    brightness_changed = target * 0.5
    purple_cast = target.clone()
    purple_cast[:, 0] *= 1.5
    purple_cast[:, 1] *= 0.3
    purple_cast[:, 2] *= 1.5
    assert loss(brightness_changed, target).item() < 1e-5
    assert loss(purple_cast, target).item() > 0.05


def test_dark_region_smoothness_targets_dark_inputs() -> None:
    loss = DarkRegionSmoothnessLoss(threshold=0.25)
    enhanced = torch.zeros(1, 3, 8, 8)
    enhanced[:, :, :, ::2] = 1.0
    dark_low = torch.zeros_like(enhanced) + 0.05
    bright_low = torch.zeros_like(enhanced) + 0.8
    assert loss(dark_low, enhanced).item() > 0.0
    assert loss(bright_low, enhanced).item() == 0.0
