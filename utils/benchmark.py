from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


@torch.inference_mode()
def estimate_macs(model: nn.Module, input_shape: tuple[int, int, int, int]) -> int:
    macs = 0
    hooks = []

    def conv_hook(module: nn.Conv2d, inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
        nonlocal macs
        batch, out_channels, out_height, out_width = output.shape
        kernel_ops = (
            module.kernel_size[0]
            * module.kernel_size[1]
            * (module.in_channels // module.groups)
        )
        macs += batch * out_channels * out_height * out_width * kernel_ops

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
    try:
        model(torch.zeros(input_shape, device=next(model.parameters()).device))
    finally:
        for hook in hooks:
            hook.remove()
    return int(macs)


@torch.inference_mode()
def benchmark_pytorch_cpu(
    model: nn.Module,
    size: int,
    warmup: int = 20,
    iterations: int = 100,
    threads: int = 1,
) -> dict[str, float | int]:
    if iterations < 1 or warmup < 0:
        raise ValueError("iterations must be positive and warmup non-negative")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    model = model.cpu().eval()
    sample = torch.rand(1, 3, size, size)
    try:
        for _ in range(warmup):
            model(sample)
        timings = []
        for _ in range(iterations):
            start = time.perf_counter()
            model(sample)
            timings.append((time.perf_counter() - start) * 1000.0)
    finally:
        torch.set_num_threads(previous_threads)
    mean_ms = statistics.fmean(timings)
    return {
        "size": size,
        "threads": threads,
        "warmup": warmup,
        "iterations": iterations,
        "mean_ms": mean_ms,
        "std_ms": statistics.pstdev(timings),
        "fps": 1000.0 / mean_ms,
    }


def checkpoint_size_bytes(path: str | Path) -> int:
    return Path(path).stat().st_size

