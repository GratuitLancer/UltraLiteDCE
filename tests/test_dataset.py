from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from datasets.lol_dataset import (
    LOLPairedDataset,
    discover_paired_images,
    paired_random_crop,
)


def _write_rgb(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(path)


def test_paired_crop_keeps_identical_position() -> None:
    base = torch.arange(3 * 12 * 14, dtype=torch.float32).reshape(3, 12, 14)
    low = base.clone()
    high = base + 1000.0
    low_crop, high_crop = paired_random_crop(low, high, 6, top=3, left=5)
    assert low_crop.shape == high_crop.shape == (3, 6, 6)
    assert torch.equal(high_crop - low_crop, torch.full_like(low_crop, 1000.0))
    assert torch.equal(low_crop, low[:, 3:9, 5:11])


def test_dataset_pairs_by_stem_and_preserves_shape(tmp_path: Path) -> None:
    low_dir = tmp_path / "low"
    high_dir = tmp_path / "high"
    image = np.full((13, 15, 3), 50, dtype=np.uint8)
    _write_rgb(low_dir / "scene01.png", image)
    _write_rgb(high_dir / "scene01.jpg", image + 100)
    pairs = discover_paired_images(low_dir, high_dir)
    sample = LOLPairedDataset(
        pairs,
        training=True,
        crop_size=8,
        horizontal_flip=True,
        vertical_flip=True,
        rotate_90=True,
    )[0]
    assert sample["low"].shape == sample["high"].shape == (3, 8, 8)


def test_missing_pair_has_clear_error(tmp_path: Path) -> None:
    low_dir = tmp_path / "low"
    high_dir = tmp_path / "high"
    _write_rgb(low_dir / "only_low.png", np.zeros((8, 8, 3), dtype=np.uint8))
    _write_rgb(high_dir / "different.png", np.zeros((8, 8, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="pairing failed"):
        discover_paired_images(low_dir, high_dir)

