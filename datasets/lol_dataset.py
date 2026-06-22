from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class PairedImage:
    name: str
    low: Path
    high: Path


def _image_map(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    files = sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(f"No supported images found in: {directory}")

    mapping: dict[str, Path] = {}
    for path in files:
        key = path.stem.casefold()
        if key in mapping:
            raise ValueError(
                f"Ambiguous image stem '{path.stem}' in {directory}: "
                f"{mapping[key].name} and {path.name}"
            )
        mapping[key] = path
    return mapping


def discover_paired_images(low_dir: str | Path, high_dir: str | Path) -> list[PairedImage]:
    low_dir = Path(low_dir)
    high_dir = Path(high_dir)
    low_map = _image_map(low_dir)
    high_map = _image_map(high_dir)

    missing_high = sorted(set(low_map) - set(high_map))
    missing_low = sorted(set(high_map) - set(low_map))
    if missing_high or missing_low:
        details = []
        if missing_high:
            details.append(
                "missing high images for: " + ", ".join(low_map[k].name for k in missing_high[:20])
            )
        if missing_low:
            details.append(
                "missing low images for: " + ", ".join(high_map[k].name for k in missing_low[:20])
            )
        raise ValueError(
            f"Low/high pairing failed between '{low_dir}' and '{high_dir}': "
            + "; ".join(details)
        )

    return [
        PairedImage(name=low_map[key].name, low=low_map[key], high=high_map[key])
        for key in sorted(low_map)
    ]


def split_pairs(
    pairs: Sequence[PairedImage],
    val_ratio: float,
    seed: int,
) -> tuple[list[PairedImage], list[PairedImage]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1)")
    indices = list(range(len(pairs)))
    random.Random(seed).shuffle(indices)
    if val_ratio == 0.0 or len(pairs) < 2:
        return [pairs[i] for i in indices], []
    val_count = max(1, int(round(len(pairs) * val_ratio)))
    val_count = min(val_count, len(pairs) - 1)
    val_indices = set(indices[:val_count])
    train = [pair for i, pair in enumerate(pairs) if i not in val_indices]
    val = [pair for i, pair in enumerate(pairs) if i in val_indices]
    return train, val


def _load_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def paired_random_crop(
    low: torch.Tensor,
    high: torch.Tensor,
    crop_size: int,
    *,
    top: int | None = None,
    left: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if low.shape != high.shape:
        raise ValueError(f"Paired images must have equal shape, got {low.shape} and {high.shape}")
    if crop_size <= 0:
        raise ValueError("crop_size must be positive")

    _, height, width = low.shape
    pad_h = max(0, crop_size - height)
    pad_w = max(0, crop_size - width)
    if pad_h or pad_w:
        pad = (0, pad_w, 0, pad_h)
        mode = (
            "reflect"
            if height > 1
            and width > 1
            and pad_h < height
            and pad_w < width
            else "replicate"
        )
        low = F.pad(low, pad, mode=mode)
        high = F.pad(high, pad, mode=mode)
        height, width = low.shape[-2:]

    top = random.randint(0, height - crop_size) if top is None else top
    left = random.randint(0, width - crop_size) if left is None else left
    if not (0 <= top <= height - crop_size and 0 <= left <= width - crop_size):
        raise ValueError("Requested crop lies outside the image")
    slices = (..., slice(top, top + crop_size), slice(left, left + crop_size))
    return low[slices], high[slices]


def paired_geometric_augment(
    low: torch.Tensor,
    high: torch.Tensor,
    horizontal_flip: bool,
    vertical_flip: bool,
    rotate_90: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if horizontal_flip and random.random() < 0.5:
        low, high = torch.flip(low, (-1,)), torch.flip(high, (-1,))
    if vertical_flip and random.random() < 0.5:
        low, high = torch.flip(low, (-2,)), torch.flip(high, (-2,))
    if rotate_90:
        k = random.randint(0, 3)
        if k:
            low, high = torch.rot90(low, k, (-2, -1)), torch.rot90(high, k, (-2, -1))
    return low.contiguous(), high.contiguous()


class LOLPairedDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        pairs: Sequence[PairedImage] | None = None,
        *,
        low_dir: str | Path | None = None,
        high_dir: str | Path | None = None,
        training: bool = False,
        crop_size: int | None = None,
        horizontal_flip: bool = True,
        vertical_flip: bool = False,
        rotate_90: bool = True,
    ) -> None:
        if pairs is None:
            if low_dir is None or high_dir is None:
                raise ValueError("Provide either pairs or both low_dir and high_dir")
            pairs = discover_paired_images(low_dir, high_dir)
        self.pairs = list(pairs)
        if not self.pairs:
            raise ValueError("Dataset contains no image pairs")
        self.training = training
        self.crop_size = crop_size
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotate_90 = rotate_90

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, object]:
        pair = self.pairs[index]
        low = _load_rgb(pair.low)
        high = _load_rgb(pair.high)
        if low.shape != high.shape:
            raise ValueError(
                f"Pair '{pair.name}' has mismatched dimensions: "
                f"low={tuple(low.shape)}, high={tuple(high.shape)}"
            )
        if self.training:
            if self.crop_size is not None:
                low, high = paired_random_crop(low, high, self.crop_size)
            low, high = paired_geometric_augment(
                low,
                high,
                self.horizontal_flip,
                self.vertical_flip,
                self.rotate_90,
            )
        return {"low": low, "high": high, "name": pair.name}
