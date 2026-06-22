from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from models import build_model
from train import resolve_device
from utils.checkpoint import load_checkpoint
from utils.config import apply_overrides, load_project_config
from utils.image import list_images, load_image, save_image


def _checkpoint_config(path: str | Path) -> dict[str, Any]:
    checkpoint = load_checkpoint(path, map_location="cpu")
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError(
            "Checkpoint does not embed its configuration; provide --config explicitly."
        )
    return config


def run_inference(
    checkpoint_path: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    *,
    config: dict[str, Any] | None = None,
    device_name: str = "auto",
) -> list[Path]:
    config = config or _checkpoint_config(checkpoint_path)
    device = resolve_device(device_name)
    model = build_model(config).to(device).eval()
    load_checkpoint(checkpoint_path, model=model, map_location=device)
    inputs = list_images(input_path)
    input_path = Path(input_path)
    output_path = Path(output_path)
    single_file = input_path.is_file()
    if single_file and output_path.suffix:
        destinations = [output_path]
    else:
        output_path.mkdir(parents=True, exist_ok=True)
        destinations = [output_path / path.name for path in inputs]

    with torch.inference_mode():
        for source, destination in zip(inputs, destinations):
            image = load_image(source).unsqueeze(0).to(device)
            enhanced = model(image)
            save_image(enhanced[0], destination)
            print(f"{source} -> {destination}")
    return destinations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enhance one image or a directory")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = (
        load_project_config(args.config, args.overrides)
        if args.config
        else apply_overrides(_checkpoint_config(args.checkpoint), args.overrides)
    )
    run_inference(
        args.checkpoint,
        args.input,
        args.output,
        config=config,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
