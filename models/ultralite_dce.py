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
        curve_color_mode: str = "coupled",
        curve_chroma_scale: float = 0.05,
        identity_init: bool = True,
        use_dark_denoise_head: bool = False,
        dark_denoise_threshold: float = 0.25,
        dark_denoise_strength: float = 0.05,
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
        if curve_color_mode not in {"coupled", "independent"}:
            raise ValueError("curve_color_mode must be 'coupled' or 'independent'")
        if not 0.0 <= curve_chroma_scale <= 1.0:
            raise ValueError("curve_chroma_scale must be in [0, 1]")

        self.width = int(width)
        self.num_blocks = int(num_blocks)
        self.num_iterations = int(num_iterations)
        self.curve_mode = curve_mode
        self.prediction_scale = float(prediction_scale)
        self.convolution = convolution
        self.curve_color_mode = curve_color_mode
        self.curve_chroma_scale = float(curve_chroma_scale)
        self.identity_init = bool(identity_init)
        self.use_dark_denoise_head = bool(use_dark_denoise_head)
        self.dark_denoise_threshold = float(dark_denoise_threshold)
        self.dark_denoise_strength = float(dark_denoise_strength)
        if not 0.0 <= self.dark_denoise_strength <= 1.0:
            raise ValueError("dark_denoise_strength must be in [0, 1]")

        self.stem = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[make_conv_block(convolution, width, width) for _ in range(num_blocks)]
        )
        curve_channels = 3 if curve_mode == "shared" else 3 * num_iterations
        self.curve_head = nn.Conv2d(width, curve_channels, kernel_size=1, bias=True)
        self.dark_denoise_head: nn.Module | None = None
        if self.use_dark_denoise_head:
            self.dark_denoise_head = nn.Sequential(
                make_conv_block(convolution, 3, width),
                make_conv_block(convolution, width, width),
                nn.Conv2d(width, 3, kernel_size=1, bias=True),
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        if self.identity_init:
            nn.init.zeros_(self.curve_head.weight)
            nn.init.zeros_(self.curve_head.bias)
            if self.dark_denoise_head is not None:
                final = self.dark_denoise_head[-1]
                if isinstance(final, nn.Conv2d):
                    nn.init.zeros_(final.weight)
                    nn.init.zeros_(final.bias)

    def _parameterize_curve(self, logits: torch.Tensor) -> torch.Tensor:
        if self.curve_color_mode == "independent":
            return torch.tanh(logits)

        steps = 1 if self.curve_mode == "shared" else self.num_iterations
        batch, _, height, width = logits.shape
        grouped = logits.reshape(batch, steps, 3, height, width)
        luminance_logits = grouped.mean(dim=2, keepdim=True)
        luminance_curve = torch.tanh(luminance_logits)
        chroma_curve = torch.tanh(grouped - luminance_logits)
        curve = torch.clamp(
            luminance_curve + self.curve_chroma_scale * chroma_curve,
            -1.0,
            1.0,
        )
        return curve.reshape(batch, steps * 3, height, width)

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
        curve = self._parameterize_curve(self.curve_head(features))
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
        if self.dark_denoise_head is not None:
            gray = image.mean(dim=1, keepdim=True)
            dark_mask = (gray < self.dark_denoise_threshold).to(dtype=enhanced.dtype)
            noise_residual = torch.tanh(self.dark_denoise_head(enhanced))
            enhanced = torch.clamp(
                enhanced - noise_residual * dark_mask * self.dark_denoise_strength,
                0.0,
                1.0,
            )
        return enhanced, curve

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.enhance(image)[0]

    def extra_repr(self) -> str:
        return (
            f"width={self.width}, num_blocks={self.num_blocks}, "
            f"num_iterations={self.num_iterations}, curve_mode={self.curve_mode}, "
            f"prediction_scale={self.prediction_scale}, convolution={self.convolution}, "
            f"curve_color_mode={self.curve_color_mode}, "
            f"curve_chroma_scale={self.curve_chroma_scale}, "
            f"identity_init={self.identity_init}, "
            f"use_dark_denoise_head={self.use_dark_denoise_head}, "
            f"dark_denoise_threshold={self.dark_denoise_threshold}, "
            f"dark_denoise_strength={self.dark_denoise_strength}"
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
        curve_color_mode=str(model_cfg.get("curve_color_mode", "coupled")),
        curve_chroma_scale=float(model_cfg.get("curve_chroma_scale", 0.05)),
        identity_init=bool(model_cfg.get("identity_init", True)),
        use_dark_denoise_head=bool(model_cfg.get("use_dark_denoise_head", False)),
        dark_denoise_threshold=float(model_cfg.get("dark_denoise_threshold", 0.25)),
        dark_denoise_strength=float(model_cfg.get("dark_denoise_strength", 0.05)),
    )
