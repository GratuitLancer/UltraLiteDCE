from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from evaluate import evaluate
from export_onnx import export_and_validate
from models import build_model
from utils.benchmark import checkpoint_size_bytes, count_parameters, estimate_macs
from utils.config import load_project_config


def _slug(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


def _training_command(config_path: str) -> str:
    return (
        "python train.py "
        f"--config {config_path} "
        "--epochs 60 "
        "--num-workers 2 "
        "--device cuda"
    )


def _static_model_fields(config: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
    model_cfg = config.get("model", {})
    model = build_model(config).cpu().eval()
    curve_mode = str(model_cfg.get("curve_mode", "shared"))
    iterations = int(model_cfg.get("num_iterations", 4))
    curve_maps = 3 if curve_mode == "shared" else 3 * iterations
    row: dict[str, Any] = {
        "width": int(model_cfg.get("width", 8)),
        "num_blocks": int(model_cfg.get("num_blocks", 3)),
        "curve_mode": curve_mode,
        "curve_maps": curve_maps,
        "iterations": iterations,
        "prediction_scale": float(model_cfg.get("prediction_scale", 0.5)),
        "convolution": str(model_cfg.get("convolution", "depthwise_separable")),
        "epochs": int(config.get("training", {}).get("epochs", 0)),
        "params": count_parameters(model),
        "macs_256": estimate_macs(model, (1, 3, 256, 256)),
        "checkpoint_bytes": checkpoint_size_bytes(checkpoint) if checkpoint.is_file() else "",
    }
    return row


def _add_evaluation_fields(row: dict[str, Any], summary: dict[str, Any]) -> None:
    for key in (
        "psnr",
        "ssim",
        "mae",
        "cpu_ms_256",
        "cpu_ms_512",
        "cpu_fps_256",
        "cpu_fps_512",
    ):
        row[key] = summary.get(key, "")
    row["evaluation_summary"] = summary.get("_summary_path", "")


def _add_onnx_fields(row: dict[str, Any], report: dict[str, Any]) -> None:
    benchmark = report.get("benchmark", {})
    row["onnx_path"] = report.get("path", "")
    row["onnx_max_abs_error"] = report.get("max_abs_error", "")
    row["onnx_mean_abs_error"] = report.get("mean_abs_error", "")
    row["onnx_dynamic_max_abs_error"] = report.get("dynamic_max_abs_error", "")
    row["onnx_dynamic_mean_abs_error"] = report.get("dynamic_mean_abs_error", "")
    row["onnx_cpu_ms_256"] = benchmark.get("mean_ms", "")
    row["onnx_cpu_fps_256"] = benchmark.get("fps", "")
    row["onnx_report"] = report.get("_report_path", "")


def _format_float(value: Any) -> str:
    if value == "" or value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isfinite(number):
        return f"{number:.4f}"
    return str(value)


def _write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    columns = [
        "model",
        "curve_mode",
        "iterations",
        "prediction_scale",
        "params",
        "macs_256",
        "psnr",
        "ssim",
        "mae",
        "cpu_ms_256",
        "onnx_cpu_ms_256",
        "status",
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            "| Model | Curve Mode | Iterations | Scale | Params | MACs 256 | "
            "PSNR | SSIM | MAE | CPU ms | ONNX ms | Status |\n"
        )
        handle.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for row in rows:
            values = [
                row.get("model", ""),
                row.get("curve_mode", ""),
                row.get("iterations", ""),
                row.get("prediction_scale", ""),
                row.get("params", ""),
                row.get("macs_256", ""),
                row.get("psnr", ""),
                row.get("ssim", ""),
                row.get("mae", ""),
                row.get("cpu_ms_256", ""),
                row.get("onnx_cpu_ms_256", ""),
                row.get("status", ""),
            ]
            handle.write("| " + " | ".join(_format_float(item) for item in values) + " |\n")


def run_one_model(
    *,
    name: str,
    config_path: str,
    checkpoint_path: str,
    output_dir: Path,
    device: str,
    quick: bool,
    skip_evaluation: bool,
    skip_onnx: bool,
    require_checkpoint: bool,
) -> dict[str, Any]:
    model_dir = output_dir / _slug(name)
    eval_dir = model_dir / "evaluation"
    onnx_dir = model_dir / "onnx"
    overrides = [f"evaluation.output_dir={eval_dir.as_posix()}"]
    if quick:
        overrides.extend(
            [
                "benchmark.sizes=[256]",
                "benchmark.warmup=1",
                "benchmark.iterations=2",
                "onnx.warmup=1",
                "onnx.iterations=2",
            ]
        )
    config = load_project_config(config_path, overrides)
    checkpoint = Path(checkpoint_path)
    row: dict[str, Any] = {
        "model": name,
        "status": "ok",
        "config": config_path,
        "checkpoint": str(checkpoint),
        "training_command": _training_command(config_path),
    }
    row.update(_static_model_fields(config, checkpoint))

    if not checkpoint.is_file():
        row["status"] = "missing_checkpoint"
        message = (
            f"Checkpoint not found for {name}: {checkpoint}. "
            f"Train it with: {_training_command(config_path)}"
        )
        if require_checkpoint:
            raise FileNotFoundError(message)
        row["note"] = message
        return row

    if not skip_evaluation:
        summary = evaluate(config, str(checkpoint), device_name=device)
        summary_path = eval_dir / "summary.json"
        summary["_summary_path"] = str(summary_path)
        _add_evaluation_fields(row, summary)
    else:
        row["status"] = "evaluation_skipped"

    if not skip_onnx:
        try:
            onnx_path = onnx_dir / f"{_slug(name)}.onnx"
            report = export_and_validate(config, str(checkpoint), onnx_path)
            report["_report_path"] = str(onnx_path.with_suffix(".json"))
            _add_onnx_fields(row, report)
        except ImportError as exc:
            row["status"] = "onnx_dependency_missing"
            row["note"] = str(exc)
            if require_checkpoint:
                raise
    return row


def compare(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        run_one_model(
            name=args.ultralite_name,
            config_path=args.ultralite_config,
            checkpoint_path=args.ultralite_checkpoint,
            output_dir=output_dir,
            device=args.device,
            quick=args.quick,
            skip_evaluation=args.skip_evaluation,
            skip_onnx=args.skip_onnx,
            require_checkpoint=args.require_all,
        ),
        run_one_model(
            name=args.baseline_name,
            config_path=args.baseline_config,
            checkpoint_path=args.baseline_checkpoint,
            output_dir=output_dir,
            device=args.device,
            quick=args.quick,
            skip_evaluation=args.skip_evaluation,
            skip_onnx=args.skip_onnx,
            require_checkpoint=args.require_all,
        ),
    ]

    fieldnames = sorted({key for row in rows for key in row})
    csv_path = output_dir / "comparison_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_dir / "comparison_summary.json"
    summary = {"rows": rows, "csv": str(csv_path)}
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    markdown_path = output_dir / "comparison_table.md"
    _write_markdown(rows, markdown_path)
    print(json.dumps(summary, indent=2))
    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {markdown_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare UltraLiteDCE against a ZeroDCE-style baseline."
    )
    parser.add_argument(
        "--ultralite-name",
        default="UltraLiteDCE color_safe_gpu_denoise",
    )
    parser.add_argument(
        "--ultralite-config",
        default="configs/comparison/ultralite_color_safe_gpu_denoise_60.yaml",
    )
    parser.add_argument(
        "--ultralite-checkpoint",
        default="outputs/color_safe_gpu_denoise/checkpoints/best_psnr.pt",
    )
    parser.add_argument("--baseline-name", default="ZeroDCE-style baseline")
    parser.add_argument(
        "--baseline-config",
        default="configs/comparison/zerodce_baseline_60.yaml",
    )
    parser.add_argument(
        "--baseline-checkpoint",
        default="outputs/zerodce_baseline_60/checkpoints/best_psnr.pt",
    )
    parser.add_argument("--output-dir", default="outputs/comparison_60")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use 256x256 only and tiny warmup/iteration counts for smoke tests.",
    )
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--skip-onnx", action="store_true")
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Fail if any checkpoint is missing.",
    )
    return parser.parse_args()


def main() -> None:
    compare(parse_args())


if __name__ == "__main__":
    main()
