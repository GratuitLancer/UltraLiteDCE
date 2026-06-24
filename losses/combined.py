from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from .supervised import (
    ChromaticityLoss,
    DarkRegionSmoothnessLoss,
    EdgePreservationLoss,
    SSIMLoss,
)
from .zero_reference import (
    ColorConstancyLoss,
    ExposureControlLoss,
    SpatialConsistencyLoss,
    TotalVariationLoss,
)


class CombinedLoss(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        cfg = config.get("loss", config)
        self.weights = {
            "spatial": float(cfg.get("spatial", 1.0)),
            "exposure": float(cfg.get("exposure", 1.0)),
            "color": float(cfg.get("color", 0.5)),
            "curve_tv": float(cfg.get("curve_tv", 20.0)),
            "l1": float(cfg.get("l1", 1.0)),
            "ssim": float(cfg.get("ssim", 0.2)),
            "edge": float(cfg.get("edge", 0.1)),
            "chromaticity": float(cfg.get("chromaticity", 0.0)),
            "dark_smooth": float(cfg.get("dark_smooth", 0.0)),
            "saturation": float(cfg.get("saturation", 0.1)),
        }
        self.spatial = SpatialConsistencyLoss()
        self.exposure = ExposureControlLoss(float(cfg.get("exposure_target", 0.6)))
        self.color = ColorConstancyLoss()
        self.curve_tv = TotalVariationLoss()
        self.ssim = SSIMLoss()
        self.edge = EdgePreservationLoss()
        self.chromaticity = ChromaticityLoss()
        self.dark_smooth = DarkRegionSmoothnessLoss(float(cfg.get("dark_threshold", 0.25)))
        self.saturation_threshold = float(cfg.get("saturation_threshold", 0.95))

    def forward(
        self,
        low: torch.Tensor,
        enhanced: torch.Tensor,
        high: torch.Tensor,
        curve: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Losses are inexpensive relative to the model and are numerically safer
        # in float32. Gradients still propagate through these casts to AMP outputs.
        low = low.float()
        enhanced = enhanced.float()
        high = high.float()
        curve = curve.float()
        losses = {
            "spatial": self.spatial(low, enhanced),
            "exposure": self.exposure(enhanced),
            "color": self.color(enhanced),
            "curve_tv": self.curve_tv(curve),
            "l1": F.l1_loss(enhanced, high),
            "ssim": self.ssim(enhanced, high),
            "edge": self.edge(enhanced, high),
            "chromaticity": self.chromaticity(enhanced, high),
            "dark_smooth": self.dark_smooth(low, enhanced),
            "saturation": F.relu(enhanced - self.saturation_threshold).mean(),
        }
        total = sum(self.weights[name] * value for name, value in losses.items())
        if not torch.isfinite(total):
            values = {name: float(value.detach().item()) for name, value in losses.items()}
            raise FloatingPointError(f"Non-finite total loss; components={values}")
        return total, {"total": total, **losses}
