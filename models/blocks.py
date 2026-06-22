from __future__ import annotations

import torch
from torch import nn


class StandardConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=True,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.depthwise(x))
        return self.activation(self.pointwise(x))


def make_conv_block(kind: str, in_channels: int, out_channels: int) -> nn.Module:
    normalized = kind.lower().replace("-", "_")
    if normalized in {"depthwise", "depthwise_separable", "dsconv"}:
        return DepthwiseSeparableConv(in_channels, out_channels)
    if normalized in {"standard", "conv"}:
        return StandardConvBlock(in_channels, out_channels)
    raise ValueError(
        f"Unsupported convolution type '{kind}'. "
        "Expected 'standard' or 'depthwise_separable'."
    )

