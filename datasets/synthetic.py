from __future__ import annotations

import torch
from torch.utils.data import Dataset


class SyntheticPairedDataset(Dataset[dict[str, object]]):
    """Deterministic paired data for pipeline smoke tests, not for reporting quality."""

    def __init__(self, length: int = 16, image_size: int = 64, seed: int = 42) -> None:
        if length < 1 or image_size < 8:
            raise ValueError("length must be positive and image_size at least 8")
        self.length = length
        self.image_size = image_size
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, object]:
        generator = torch.Generator().manual_seed(self.seed + index)
        high = torch.rand(
            3,
            self.image_size,
            self.image_size,
            generator=generator,
        )
        # Add smooth structure so edge and reconstruction losses exercise useful paths.
        high = torch.nn.functional.avg_pool2d(
            high.unsqueeze(0), kernel_size=5, stride=1, padding=2
        ).squeeze(0)
        low = torch.clamp(high.pow(1.6) * 0.45, 0.0, 1.0)
        return {"low": low, "high": high, "name": f"synthetic_{index:04d}.png"}

