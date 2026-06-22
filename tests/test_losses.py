from __future__ import annotations

import torch

from losses import CombinedLoss, ssim_index
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
            "saturation": 0.1,
            "exposure_target": 0.6,
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

