from __future__ import annotations

import argparse
import inspect
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models import build_model
from utils.checkpoint import load_checkpoint
from utils.config import load_project_config


def benchmark_onnx_session(
    session: Any,
    sample: np.ndarray,
    warmup: int,
    iterations: int,
) -> dict[str, float | int]:
    for _ in range(warmup):
        session.run(["enhanced"], {"input": sample})
    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        session.run(["enhanced"], {"input": sample})
        timings.append((time.perf_counter() - start) * 1000.0)
    mean_ms = statistics.fmean(timings)
    return {
        "warmup": warmup,
        "iterations": iterations,
        "mean_ms": mean_ms,
        "std_ms": statistics.pstdev(timings),
        "fps": 1000.0 / mean_ms,
    }


def export_and_validate(
    config: dict[str, Any],
    checkpoint_path: str,
    output_path: str | Path,
    *,
    height: int = 256,
    width: int = 256,
) -> dict[str, Any]:
    try:
        import onnx
    except ImportError as exc:
        raise ImportError("ONNX export validation requires `pip install onnx`") from exc
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError("ONNX comparison requires `pip install onnxruntime`") from exc

    model = build_model(config).cpu().eval()
    load_checkpoint(checkpoint_path, model=model, map_location="cpu")
    sample = torch.rand(1, 3, height, width)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    opset = int(config.get("onnx", {}).get("opset", 17))
    with torch.inference_mode():
        torch_output = model(sample).numpy()
    export_kwargs: dict[str, Any] = {
        "export_params": True,
        "opset_version": opset,
        "do_constant_folding": True,
        "input_names": ["input"],
        "output_names": ["enhanced"],
        "dynamic_axes": {
            "input": {0: "batch", 2: "height", 3: "width"},
            "enhanced": {0: "batch", 2: "height", 3: "width"},
        },
        "verbose": False,
    }
    # The classic exporter plus dynamic_axes is stable across the PyTorch
    # versions commonly used for this project. Newer PyTorch versions expose a
    # dynamo flag; forcing False avoids version-specific dynamic_shapes behavior.
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False
    torch.onnx.export(model, sample, output_path, **export_kwargs)
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    ort_output = session.run(["enhanced"], {"input": sample.numpy()})[0]
    difference = np.abs(torch_output - ort_output)
    max_error = float(difference.max())
    mean_error = float(difference.mean())
    dynamic_sample = torch.rand(2, 3, height + 2, width + 4)
    with torch.inference_mode():
        dynamic_torch_output = model(dynamic_sample).numpy()
    dynamic_ort_output = session.run(
        ["enhanced"], {"input": dynamic_sample.numpy()}
    )[0]
    dynamic_difference = np.abs(dynamic_torch_output - dynamic_ort_output)
    dynamic_max_error = float(dynamic_difference.max())
    dynamic_mean_error = float(dynamic_difference.mean())
    threshold = float(config.get("onnx", {}).get("error_threshold", 1e-4))
    if max(max_error, dynamic_max_error) > threshold:
        raise RuntimeError(
            "ONNX output mismatch: "
            f"max error {max(max_error, dynamic_max_error):.6g} exceeds {threshold:.6g}"
        )
    onnx_cfg = config.get("onnx", {})
    benchmark = benchmark_onnx_session(
        session,
        sample.numpy(),
        warmup=int(onnx_cfg.get("warmup", 20)),
        iterations=int(onnx_cfg.get("iterations", 100)),
    )
    result = {
        "path": str(output_path),
        "opset": opset,
        "max_abs_error": max_error,
        "mean_abs_error": mean_error,
        "dynamic_shape": list(dynamic_sample.shape),
        "dynamic_max_abs_error": dynamic_max_error,
        "dynamic_mean_abs_error": dynamic_mean_error,
        "threshold": threshold,
        "benchmark": benchmark,
    }
    report_path = output_path.with_suffix(".json")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export and validate UltraLiteDCE ONNX")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="outputs/onnx/ultralite_dce.onnx")
    parser.add_argument("--config", default="configs/ultralite_dce.yaml")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--opset", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = list(args.overrides)
    for key in ("opset", "iterations", "warmup"):
        value = getattr(args, key)
        if value is not None:
            overrides.append(f"onnx.{key}={value}")
    config = load_project_config(args.config, overrides)
    export_and_validate(
        config,
        args.checkpoint,
        args.output,
        height=args.height,
        width=args.width,
    )


if __name__ == "__main__":
    main()
