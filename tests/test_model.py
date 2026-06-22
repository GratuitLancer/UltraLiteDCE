from __future__ import annotations

import pytest
import torch

from models import UltraLiteDCE


@pytest.mark.parametrize("curve_mode", ["shared", "per_step"])
def test_model_shape_range_and_curve_modes(curve_mode: str) -> None:
    model = UltraLiteDCE(
        width=8,
        num_blocks=3,
        num_iterations=4,
        curve_mode=curve_mode,
        prediction_scale=0.5,
    )
    image = torch.rand(2, 3, 32, 40)
    enhanced, curve = model.enhance(image)
    assert enhanced.shape == image.shape
    expected_channels = 3 if curve_mode == "shared" else 12
    assert curve.shape == (2, expected_channels, 32, 40)
    assert torch.all(enhanced >= 0.0)
    assert torch.all(enhanced <= 1.0)


@pytest.mark.parametrize("height,width", [(31, 47), (255, 257), (17, 19)])
def test_arbitrary_odd_size_inference(height: int, width: int) -> None:
    model = UltraLiteDCE(prediction_scale=0.25).eval()
    image = torch.rand(1, 3, height, width)
    with torch.inference_mode():
        output = model(image)
    assert output.shape == image.shape


def test_forward_backward_has_finite_gradients() -> None:
    model = UltraLiteDCE()
    image = torch.rand(1, 3, 24, 24)
    target = torch.rand_like(image)
    output = model(image)
    loss = torch.nn.functional.l1_loss(output, target)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)
    assert any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients if gradient is not None)

