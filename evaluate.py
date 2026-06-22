from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from datasets import LOLPairedDataset
from models import build_model
from train import resolve_device
from utils.benchmark import checkpoint_size_bytes, count_parameters, estimate_macs
from utils.checkpoint import load_checkpoint
from utils.config import load_project_config
from utils.image import save_comparison, save_image
from utils.metrics import OptionalMetrics, paired_metrics


def evaluate(config: dict[str, Any], checkpoint_path: str, device_name: str | None = None) -> dict[str, Any]:
    device = resolve_device(device_name or str(config["training"].get("device", "auto")))
    model = build_model(config).to(device).eval()
    load_checkpoint(checkpoint_path, model=model, map_location=device)
    data_cfg = config["data"]
    root = Path(data_cfg["root"])
    dataset = LOLPairedDataset(
        low_dir=root / data_cfg["test_low"],
        high_dir=root / data_cfg["test_high"],
        training=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    output_dir = Path(config.get("evaluation", {}).get("output_dir", "outputs/evaluation"))
    enhanced_dir = output_dir / "enhanced"
    comparison_dir = output_dir / "comparisons"
    output_dir.mkdir(parents=True, exist_ok=True)
    optional = OptionalMetrics(
        bool(config.get("evaluation", {}).get("lpips", False)),
        bool(config.get("evaluation", {}).get("niqe", False)),
        device,
    )

    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in loader:
            low = batch["low"].to(device)
            high = batch["high"].to(device)
            enhanced = model(low)
            name = str(batch["name"][0])
            values = paired_metrics(enhanced, high)
            values.update(optional.compute(enhanced, high))
            rows.append({"name": name, **values})
            save_image(enhanced[0], enhanced_dir / name)
            save_comparison(low[0], enhanced[0], high[0], comparison_dir / name)

    metric_names = [key for key in rows[0] if key != "name"]
    with (output_dir / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", *metric_names])
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, Any] = {
        "num_images": len(rows),
        "parameters": count_parameters(model),
        "checkpoint_bytes": checkpoint_size_bytes(checkpoint_path),
        "macs_256": estimate_macs(model, (1, 3, 256, 256)),
    }
    for name in metric_names:
        summary[name] = statistics.fmean(float(row[name]) for row in rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate UltraLiteDCE on LOL eval pairs")
    parser.add_argument("--config", default="configs/ultralite_dce.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device")
    parser.add_argument("--output-dir")
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--niqe", action="store_true")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = list(args.overrides)
    if args.output_dir:
        overrides.append(f"evaluation.output_dir={args.output_dir}")
    if args.lpips:
        overrides.append("evaluation.lpips=true")
    if args.niqe:
        overrides.append("evaluation.niqe=true")
    config = load_project_config(args.config, overrides)
    evaluate(config, args.checkpoint, args.device)


if __name__ == "__main__":
    main()

