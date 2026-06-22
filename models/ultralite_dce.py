from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from .blocks import make_conv_block
from .curve import apply_curve


class UltraLiteDCE(nn.Module):
    """A deployment-friendly Zero-DCE-style curve estimation network."""

    def __init__(
        self,
        width: int = 8,
        num_blocks: int = 3,
        num_iterations: int = 4,
        curve_mode: str = "shared",
        prediction_scale: float = 0.5,
        convolution: str = "depthwise_separable",
    ) -> None:
        super().__init__()
        if width < 1 or num_blocks < 0:
            raise ValueError("width must be positive and num_blocks non-negative")
        if num_iterations < 1:
            raise ValueError("num_iterations must be positive")
        if curve_mode not in {"shared", "per_step"}:
            raise ValueError("curve_mode must be 'shared' or 'per_step'")
        if not 0.0 < prediction_scale <= 1.0:
            raise ValueError("prediction_scale must be in (0, 1]")

        self.width = int(width)
        self.num_blocks = int(num_blocks)
        self.num_iterations = int(num_iterations)
        self.curve_mode = curve_mode
        self.prediction_scale = float(prediction_scale)
        self.convolution = convolution

        self.stem = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[make_conv_block(convolution, width, width) for _ in range(num_blocks)]
        )
        curve_channels = 3 if curve_mode == "shared" else 3 * num_iterations
        self.curve_head = nn.Conv2d(width, curve_channels, kernel_size=1, bias=True)

    def predict_curve(self, image: torch.Tensor) -> torch.Tensor:
        if self.prediction_scale == 1.0:
            prediction_input = image
        else:
            prediction_input = F.interpolate(
                image,
                scale_factor=self.prediction_scale,
                mode="bilinear",
                align_corners=False,
                recompute_scale_factor=False,
            )
        features = self.blocks(self.stem(prediction_input))
        curve = torch.tanh(self.curve_head(features))
        if curve.shape[-2:] != image.shape[-2:]:
            curve = F.interpolate(
                curve,
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return curve

    def enhance(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        curve = self.predict_curve(image)
        enhanced = apply_curve(
            image,
            curve,
            num_iterations=self.num_iterations,
            curve_mode=self.curve_mode,
        )
        return enhanced, curve

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.enhance(image)[0]

    def extra_repr(self) -> str:
        return (
            f"width={self.width}, num_blocks={self.num_blocks}, "
            f"num_iterations={self.num_iterations}, curve_mode={self.curve_mode}, "
            f"prediction_scale={self.prediction_scale}, convolution={self.convolution}"
        )


def build_model(config: Mapping[str, Any]) -> UltraLiteDCE:
    model_cfg = config.get("model", config)
    name = str(model_cfg.get("name", "ultralite_dce")).lower()
    if name not in {"ultralite_dce", "ultralitedce"}:
        raise ValueError(f"Unsupported model name: {name}")
    return UltraLiteDCE(
        width=int(model_cfg.get("width", 8)),
        num_blocks=int(model_cfg.get("num_blocks", 3)),
        num_iterations=int(model_cfg.get("num_iterations", 4)),
        curve_mode=str(model_cfg.get("curve_mode", "shared")),
        prediction_scale=float(model_cfg.get("prediction_scale", 0.5)),
        convolution=str(model_cfg.get("convolution", "depthwise_separable")),
    )

