from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageOps, ImageDraw

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_image(path: str | Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().float().cpu().clamp(0.0, 1.0)
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError("Only batch size 1 can be converted directly to PIL")
        tensor = tensor[0]
    array = (
        tensor.permute(1, 2, 0).numpy() * 255.0
    ).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def save_image(tensor: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(tensor).save(path)


def list_images(path: str | Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {path.suffix}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    images = sorted(
        p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No supported images found in: {path}")
    return images


def save_comparison(
    low: torch.Tensor,
    enhanced: torch.Tensor,
    high: torch.Tensor,
    path: str | Path,
) -> None:
    images = [tensor_to_pil(item) for item in (low, enhanced, high)]
    labels = ["Low", "Enhanced", "High"]
    label_height = 24
    width = sum(image.width for image in images)
    height = max(image.height for image in images) + label_height
    canvas = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(canvas)
    x = 0
    for image, label in zip(images, labels):
        canvas.paste(image, (x, label_height))
        draw.text((x + 5, 5), label, fill="white")
        x += image.width
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def save_training_preview(
    low: torch.Tensor,
    enhanced: torch.Tensor,
    high: torch.Tensor,
    path: str | Path,
    max_items: int = 4,
) -> None:
    rows = []
    for index in range(min(low.shape[0], max_items)):
        row = [tensor_to_pil(batch[index]) for batch in (low, enhanced, high)]
        canvas = Image.new("RGB", (sum(im.width for im in row), row[0].height))
        x = 0
        for image in row:
            canvas.paste(image, (x, 0))
            x += image.width
        rows.append(canvas)
    output = Image.new("RGB", (rows[0].width, sum(row.height for row in rows)))
    y = 0
    for row in rows:
        output.paste(row, (0, y))
        y += row.height
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.save(path)


def save_curve_preview(
    curve: torch.Tensor,
    path: str | Path,
    max_items: int = 4,
) -> None:
    """Save curve maps as RGB previews, mapping A from [-1, 1] to [0, 1]."""
    curve = curve.detach().float().cpu()
    if curve.ndim != 4:
        raise ValueError("curve must have shape BxCxHxW")
    if curve.shape[1] % 3 == 0:
        curve_rgb = curve.reshape(curve.shape[0], -1, 3, curve.shape[-2], curve.shape[-1])
        curve_rgb = curve_rgb.mean(dim=1)
    elif curve.shape[1] >= 3:
        curve_rgb = curve[:, :3]
    else:
        curve_rgb = curve.mean(dim=1, keepdim=True).expand(-1, 3, -1, -1)
    curve_rgb = (curve_rgb + 1.0) * 0.5

    rows = [tensor_to_pil(curve_rgb[index]) for index in range(min(curve_rgb.shape[0], max_items))]
    width = max(image.width for image in rows)
    height = sum(image.height for image in rows)
    output = Image.new("RGB", (width, height), "black")
    y = 0
    for image in rows:
        output.paste(image, (0, y))
        y += image.height
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.save(path)
