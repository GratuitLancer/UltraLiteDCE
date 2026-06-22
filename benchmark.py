from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from models import build_model
from utils.benchmark import (
    benchmark_pytorch_cpu,
    checkpoint_size_bytes,
    count_parameters,
    estimate_macs,
)
from utils.checkpoint import load_checkpoint
from utils.config import load_project_config


def benchmark_model(
    config: dict[str, Any],
    checkpoint: str | None = None,
    output: str | Path | None = None,
) -> dict[str, Any]:
    cfg = config.get("benchmark", {})
    model = build_model(config).cpu().eval()
    if checkpoint:
        load_checkpoint(checkpoint, model=model, map_location="cpu")
    results: dict[str, Any] = {
        "parameters": count_parameters(model),
        "checkpoint_bytes": checkpoint_size_bytes(checkpoint) if checkpoint else None,
        "runs": [],
    }
    for size in cfg.get("sizes", [256, 512]):
        size = int(size)
        run = benchmark_pytorch_cpu(
            model,
            size=size,
            warmup=int(cfg.get("warmup", 20)),
            iterations=int(cfg.get("iterations", 100)),
            threads=int(cfg.get("threads", 1)),
        )
        run["macs"] = estimate_macs(model, (1, 3, size, size))
        run["approx_flops"] = 2 * run["macs"]
        results["runs"].append(run)
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
    print(json.dumps(results, indent=2))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU benchmark for UltraLiteDCE")
    parser.add_argument("--config", default="configs/ultralite_dce.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", default="outputs/benchmark.json")
    parser.add_argument("--sizes", nargs="+", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = list(args.overrides)
    for key in ("warmup", "iterations", "threads"):
        value = getattr(args, key)
        if value is not None:
            overrides.append(f"benchmark.{key}={value}")
    if args.sizes:
        overrides.append(f"benchmark.sizes={args.sizes}")
    config = load_project_config(args.config, overrides)
    benchmark_model(config, args.checkpoint, args.output)


if __name__ == "__main__":
    main()

