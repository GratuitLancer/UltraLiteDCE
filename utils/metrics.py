from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from losses.supervised import ssim_index


def psnr(prediction: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    mse = F.mse_loss(prediction, target).item()
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10((data_range**2) / mse)


def mae(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return F.l1_loss(prediction, target).item()


@torch.inference_mode()
def paired_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    return {
        "psnr": psnr(prediction, target),
        "ssim": float(ssim_index(prediction, target).item()),
        "mae": mae(prediction, target),
    }


class OptionalMetrics:
    def __init__(self, use_lpips: bool, use_niqe: bool, device: torch.device) -> None:
        self.lpips_model: Any = None
        self.niqe_model: Any = None
        if use_lpips:
            try:
                import lpips
            except ImportError as exc:
                raise ImportError("LPIPS requested; install with `pip install lpips`") from exc
            self.lpips_model = lpips.LPIPS(net="alex").to(device).eval()
        if use_niqe:
            try:
                import pyiqa
            except ImportError as exc:
                raise ImportError("NIQE requested; install with `pip install pyiqa`") from exc
            self.niqe_model = pyiqa.create_metric("niqe", device=device)

    @torch.inference_mode()
    def compute(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, float]:
        values: dict[str, float] = {}
        if self.lpips_model is not None:
            values["lpips"] = float(
                self.lpips_model(prediction * 2.0 - 1.0, target * 2.0 - 1.0).mean().item()
            )
        if self.niqe_model is not None:
            values["niqe"] = float(self.niqe_model(prediction).mean().item())
        return values

