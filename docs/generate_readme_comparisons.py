"""Regenerate README triptychs from real LOL-v1 pairs and a trained checkpoint.

Run from the repository root: python docs/generate_readme_comparisons.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import build_model
from utils.checkpoint import load_checkpoint
from utils.image import load_image, tensor_to_pil
from utils.metrics import paired_metrics


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = (
        ["arialbd.ttf", "DejaVuSans-Bold.ttf"]
        if bold
        else ["arial.ttf", "DejaVuSans.ttf"]
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    raise RuntimeError("Install Arial or DejaVu Sans to render readable figure labels")


def comparison_panel(
    images: list[Image.Image], name: str, epoch: int, metrics: dict[str, float]
) -> Image.Image:
    width, height = images[0].size
    if any(image.size != (width, height) for image in images):
        raise ValueError(f"Unaligned input / enhanced / GT sizes for {name}")
    padding, gap, image_y = 24, 16, 180
    canvas = Image.new("RGB", (width * 3 + gap * 2 + padding * 2, height + 242), "white")
    draw = ImageDraw.Draw(canvas)
    ink, muted, accent = "#142D46", "#53677C", "#087E8B"
    draw.text((padding, 20), f"LOL-v1 / eval15  |  {name}", font=font(34, True), fill=ink)
    draw.text(
        (padding, 67),
        f"Real paired capture  |  Native {width} x {height} pixels  |  Best-PSNR checkpoint: epoch {epoch}",
        font=font(22), fill=muted,
    )
    columns = [
        ("Input", "Original low-light image"),
        ("Enhanced (UltraLiteDCE)", "Model output from input only"),
        ("GT", "Paired normal-light reference"),
    ]
    for index, (image, (label, detail)) in enumerate(zip(images, columns)):
        x = padding + index * (width + gap)
        draw.rectangle((x, 109, x + width, 112), fill=accent if index == 1 else "#CBD5DF")
        draw.text((x, 119), label, font=font(29, True), fill=ink)
        draw.text((x, 154), detail, font=font(20), fill=muted)
        canvas.paste(image, (x, image_y))
    draw.text(
        (padding, image_y + height + 18),
        f"Enhanced vs. GT: PSNR {metrics['psnr']:.2f} dB  |  SSIM {metrics['ssim']:.4f}"
        "  |  No crops, resizing, or post-processing",
        font=font(22), fill=muted,
    )
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/LOL/eval15")
    parser.add_argument(
        "--checkpoint", type=Path,
        default=ROOT / "outputs/color_safe_gpu_denoise/checkpoints/best_psnr.pt",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "docs/assets/comparisons")
    args = parser.parse_args()
    torch.set_num_threads(1)
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    if config.get("data", {}).get("synthetic", False):
        raise ValueError("README comparisons require a checkpoint trained on real data")
    model = build_model(config).cpu().eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    # Match evaluation.py's filename order, independent of image quality scores.
    names = [path.name for path in sorted((args.dataset / "low").glob("*.png"))][:4]
    if len(names) != 4:
        raise ValueError("Expected at least four real LOL-v1 evaluation pairs")
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "dataset": "LOL-v1",
        "split": "eval15",
        "dataset_source": "https://daooshee.github.io/BMVC2018website/",
        "selection": "First four PNG filenames in lexicographic order; no metric-based selection",
        "checkpoint": relative_path(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_epoch": checkpoint["epoch"],
        "configured_training_epochs": config["training"]["epochs"],
        "model": config["model"],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "inference": {"device": "cpu", "dtype": "float32", "torch": torch.__version__},
        "display": "Native RGB pixels; output clamped and rounded to 8-bit by tensor_to_pil; no other processing",
        "metrics": "Repository paired_metrics on float32 output before 8-bit conversion",
        "samples": [],
    }
    with torch.inference_mode():
        for name in names:
            low_path, high_path = args.dataset / "low" / name, args.dataset / "high" / name
            low, high = load_image(low_path).unsqueeze(0), load_image(high_path).unsqueeze(0)
            if low.shape != high.shape:
                raise ValueError(f"Unaligned paired images: {name}")
            enhanced = model(low)
            metrics = paired_metrics(enhanced, high)
            with Image.open(low_path) as low_image, Image.open(high_path) as high_image:
                images = [low_image.convert("RGB"), tensor_to_pil(enhanced), high_image.convert("RGB")]
            destination = args.output / f"lol_eval15_{Path(name).stem}.png"
            panel = comparison_panel(images, name, checkpoint["epoch"], metrics)
            panel.save(destination, optimize=True)
            provenance["samples"].append({
                "name": name,
                "input": relative_path(low_path), "input_sha256": sha256(low_path),
                "gt": relative_path(high_path), "gt_sha256": sha256(high_path),
                "native_size": list(images[0].size),
                "enhanced_rgb_sha256": hashlib.sha256(images[1].tobytes()).hexdigest(),
                "figure": relative_path(destination), "figure_sha256": sha256(destination),
                **metrics,
            })
            print(f"{name}: PSNR {metrics['psnr']:.4f}, SSIM {metrics['ssim']:.4f} -> {relative_path(destination)}")
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
