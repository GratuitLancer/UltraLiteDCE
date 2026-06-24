from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from datasets import (
    LOLPairedDataset,
    SyntheticPairedDataset,
    discover_paired_images,
    split_pairs,
)
from losses import CombinedLoss
from models import build_model
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.config import load_project_config, save_config
from utils.image import save_training_preview
from utils.metrics import paired_metrics
from utils.seed import seed_everything, seed_worker

LOSS_NAMES = [
    "total",
    "spatial",
    "exposure",
    "color",
    "curve_tv",
    "l1",
    "ssim",
    "edge",
    "chromaticity",
    "saturation",
]


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def build_datasets(config: dict[str, Any]) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    data_cfg = config["data"]
    train_cfg = config["training"]
    seed = int(train_cfg.get("seed", 42))
    if bool(data_cfg.get("synthetic", False)):
        image_size = int(train_cfg.get("crop_size", 256))
        train_dataset = SyntheticPairedDataset(
            length=int(data_cfg.get("synthetic_train_size", 16)),
            image_size=image_size,
            seed=seed,
        )
        val_dataset = SyntheticPairedDataset(
            length=int(data_cfg.get("synthetic_val_size", 4)),
            image_size=image_size,
            seed=seed + 10000,
        )
        return train_dataset, val_dataset

    root = Path(data_cfg["root"])
    pairs = discover_paired_images(root / data_cfg["train_low"], root / data_cfg["train_high"])
    train_pairs, val_pairs = split_pairs(
        pairs,
        val_ratio=float(data_cfg.get("val_ratio", 0.1)),
        seed=seed,
    )
    if not val_pairs:
        raise ValueError(
            "Validation split is empty. Provide at least two training pairs or set a valid val_ratio."
        )
    train_dataset = LOLPairedDataset(
        train_pairs,
        training=True,
        crop_size=int(train_cfg.get("crop_size", 256)),
        horizontal_flip=bool(data_cfg.get("horizontal_flip", True)),
        vertical_flip=bool(data_cfg.get("vertical_flip", False)),
        rotate_90=bool(data_cfg.get("rotate_90", True)),
    )
    val_dataset = LOLPairedDataset(val_pairs, training=False)
    return train_dataset, val_dataset


def resolve_amp_dtype(name: str, device: torch.device) -> torch.dtype:
    normalized = name.lower().replace("_", "")
    if normalized in {"bfloat16", "bf16"}:
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            print("Warning: CUDA device does not support bfloat16; falling back to float16.")
            return torch.float16
        return torch.bfloat16
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    raise ValueError("training.amp_dtype must be 'bfloat16' or 'float16'")


def make_grad_scaler(enabled: bool) -> Any:
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: CombinedLoss,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    grad_clip: float,
) -> dict[str, float]:
    model.train()
    sums: dict[str, float] = defaultdict(float)
    sample_count = 0
    for batch in loader:
        low = batch["low"].to(device, non_blocking=True)
        high = batch["high"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            enhanced, curve = model.enhance(low)
        # Compute loss outside autocast. SSIM's variance calculation is unstable
        # in float16 and can otherwise produce enormous negative values.
        total, components = criterion(low, enhanced, high, curve)
        if not torch.isfinite(total) or abs(float(total.detach().item())) > 1e6:
            values = {
                name: float(value.detach().item())
                for name, value in components.items()
            }
            raise FloatingPointError(
                f"Invalid loss detected before backward: {values}"
            )
        scaler.scale(total).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        batch_size = low.shape[0]
        sample_count += batch_size
        for name, value in components.items():
            sums[name] += float(value.detach().item()) * batch_size
    return {name: sums[name] / sample_count for name in LOSS_NAMES}


@torch.inference_mode()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: CombinedLoss,
    device: torch.device,
) -> tuple[dict[str, float], tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None]:
    model.eval()
    sums: dict[str, float] = defaultdict(float)
    sample_count = 0
    preview = None
    for batch in loader:
        low = batch["low"].to(device)
        high = batch["high"].to(device)
        enhanced, curve = model.enhance(low)
        _, components = criterion(low, enhanced, high, curve)
        metrics = paired_metrics(enhanced, high)
        batch_size = low.shape[0]
        sample_count += batch_size
        for name, value in components.items():
            sums[f"val_loss_{name}"] += float(value.item()) * batch_size
        for name, value in metrics.items():
            sums[f"val_{name}"] += value * batch_size
        if preview is None:
            preview = (low.cpu(), enhanced.cpu(), high.cpu())
    if sample_count == 0:
        raise RuntimeError("Validation loader produced no samples")
    return {name: value / sample_count for name, value in sums.items()}, preview


def append_csv(path: Path, row: dict[str, float | int], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_training(config: dict[str, Any], resume: str | None = None) -> dict[str, Any]:
    train_cfg = config["training"]
    seed = int(train_cfg.get("seed", 42))
    seed_everything(seed)
    device = resolve_device(str(train_cfg.get("device", "auto")))
    output_dir = Path(train_cfg.get("output_dir", "outputs"))
    checkpoint_dir = output_dir / "checkpoints"
    preview_dir = output_dir / "previews"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "resolved_config.yaml")

    train_dataset, val_dataset = build_datasets(config)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg.get("batch_size", 8)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=int(train_cfg.get("num_workers", 4)) > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = build_model(config).to(device)
    criterion = CombinedLoss(config).to(device)
    optimizer = Adam(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
    )
    epochs = int(train_cfg.get("epochs", 100))
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    amp_enabled = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    amp_dtype = resolve_amp_dtype(
        str(train_cfg.get("amp_dtype", "bfloat16")),
        device,
    )
    # bfloat16 has float32-like exponent range and does not need loss scaling.
    scaler = make_grad_scaler(amp_enabled and amp_dtype == torch.float16)
    start_epoch = 0
    best_psnr = -math.inf

    if resume:
        checkpoint = load_checkpoint(
            resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device,
        )
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_psnr = float(checkpoint.get("best_psnr", -math.inf))

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(output_dir / "tensorboard")
    except ImportError:
        pass

    csv_fields = ["epoch", "lr", *LOSS_NAMES]
    csv_fields += [f"val_loss_{name}" for name in LOSS_NAMES]
    csv_fields += ["val_psnr", "val_ssim", "val_mae"]
    history_path = output_dir / "train_log.csv"

    for epoch in range(start_epoch, epochs):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            amp_enabled,
            amp_dtype,
            float(train_cfg.get("grad_clip", 1.0)),
        )
        val_metrics, preview = validate(model, val_loader, criterion, device)
        learning_rate = optimizer.param_groups[0]["lr"]
        scheduler.step()
        row: dict[str, float | int] = {
            "epoch": epoch + 1,
            "lr": learning_rate,
            **train_metrics,
            **val_metrics,
        }
        append_csv(history_path, row, csv_fields)

        if writer is not None:
            for name, value in row.items():
                if name != "epoch":
                    writer.add_scalar(name, value, epoch + 1)

        preview_interval = int(train_cfg.get("preview_interval", 5))
        if preview is not None and ((epoch + 1) % preview_interval == 0 or epoch == start_epoch):
            save_training_preview(*preview, preview_dir / f"epoch_{epoch + 1:04d}.png")

        current_psnr = float(val_metrics["val_psnr"])
        is_best = current_psnr > best_psnr
        if is_best:
            best_psnr = current_psnr
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_psnr": best_psnr,
            "config": config,
        }
        checkpoint_interval = int(train_cfg.get("checkpoint_interval", 1))
        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(state, checkpoint_dir / "last.pt")
        if is_best:
            save_checkpoint(state, checkpoint_dir / "best.pt")

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"loss={train_metrics['total']:.6f} "
            f"val_psnr={current_psnr:.3f} val_ssim={val_metrics['val_ssim']:.4f}"
        )

    if writer is not None:
        writer.close()
    return {
        "output_dir": str(output_dir),
        "best_psnr": best_psnr,
        "epochs_completed": max(0, epochs - start_epoch),
        "synthetic": bool(config["data"].get("synthetic", False)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train UltraLiteDCE on paired LOL data")
    parser.add_argument("--config", default="configs/ultralite_dce.yaml")
    parser.add_argument("--resume")
    parser.add_argument("--device")
    parser.add_argument("--output-dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override a YAML value, e.g. --set model.width=12",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = list(args.overrides)
    mapping = {
        "device": ("training.device", args.device),
        "output_dir": ("training.output_dir", args.output_dir),
        "epochs": ("training.epochs", args.epochs),
        "batch_size": ("training.batch_size", args.batch_size),
        "num_workers": ("training.num_workers", args.num_workers),
    }
    for _, (key, value) in mapping.items():
        if value is not None:
            overrides.append(f"{key}={value}")
    if args.synthetic:
        overrides.append("data.synthetic=true")
    config = load_project_config(args.config, overrides)
    run_training(config, resume=args.resume)


if __name__ == "__main__":
    main()
