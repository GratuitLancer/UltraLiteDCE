from __future__ import annotations

import torch


def apply_curve(
    image: torch.Tensor,
    curve_map: torch.Tensor,
    num_iterations: int,
    curve_mode: str,
) -> torch.Tensor:
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("image must have shape Bx3xHxW")
    if num_iterations < 1:
        raise ValueError("num_iterations must be at least 1")

    mode = curve_mode.lower()
    if mode == "shared":
        if curve_map.shape[1] != 3:
            raise ValueError("shared curve mode requires a 3-channel curve map")
        curves = [curve_map] * num_iterations
    elif mode == "per_step":
        expected = 3 * num_iterations
        if curve_map.shape[1] != expected:
            raise ValueError(
                f"per_step curve mode requires {expected} channels, "
                f"got {curve_map.shape[1]}"
            )
        curves = torch.chunk(curve_map, num_iterations, dim=1)
    else:
        raise ValueError("curve_mode must be 'shared' or 'per_step'")

    enhanced = image
    for curve in curves:
        enhanced = enhanced + curve * enhanced * (1.0 - enhanced)
    return torch.clamp(enhanced, 0.0, 1.0)

