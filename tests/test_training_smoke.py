from __future__ import annotations

from pathlib import Path

from train import run_training


def test_synthetic_training_smoke(tmp_path: Path) -> None:
    config = {
        "model": {
            "name": "ultralite_dce",
            "width": 4,
            "num_blocks": 1,
            "num_iterations": 2,
            "curve_mode": "shared",
            "prediction_scale": 0.5,
            "convolution": "depthwise_separable",
        },
        "data": {
            "synthetic": True,
            "synthetic_train_size": 2,
            "synthetic_val_size": 1,
        },
        "loss": {
            "spatial": 1.0,
            "exposure": 1.0,
            "color": 0.5,
            "curve_tv": 20.0,
            "l1": 1.0,
            "ssim": 0.2,
            "edge": 0.1,
            "chromaticity": 1.0,
            "saturation": 0.1,
            "exposure_target": 0.6,
            "saturation_threshold": 0.95,
        },
        "training": {
            "epochs": 1,
            "batch_size": 1,
            "crop_size": 16,
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "num_workers": 0,
            "amp": False,
            "grad_clip": 1.0,
            "seed": 42,
            "device": "cpu",
            "output_dir": str(tmp_path / "training"),
            "preview_interval": 1,
            "checkpoint_interval": 1,
        },
    }
    result = run_training(config)
    assert result["epochs_completed"] == 1
    assert result["synthetic"] is True
    assert (tmp_path / "training" / "checkpoints" / "last.pt").is_file()
    assert (tmp_path / "training" / "checkpoints" / "best.pt").is_file()
    assert (tmp_path / "training" / "train_log.csv").is_file()
