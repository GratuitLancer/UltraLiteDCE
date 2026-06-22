from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch


def save_checkpoint(state: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(state), temporary)
    temporary.replace(path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dictionary")
    state_dict = checkpoint.get("model", checkpoint.get("state_dict"))
    if model is not None:
        if state_dict is None:
            raise KeyError("Checkpoint does not contain a model/state_dict entry")
        model.load_state_dict(state_dict, strict=strict)
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint
