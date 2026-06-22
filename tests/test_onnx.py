from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from export_onnx import export_and_validate
from models import build_model
from utils.checkpoint import save_checkpoint


def test_onnx_export_and_consistency(tmp_path: Path) -> None:
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
        "onnx": {
            "opset": 18,
            "error_threshold": 1e-4,
            "warmup": 1,
            "iterations": 2,
        },
    }
    model = build_model(config)
    checkpoint = tmp_path / "model.pt"
    save_checkpoint({"model": model.state_dict(), "config": config}, checkpoint)
    output = tmp_path / "model.onnx"
    result = export_and_validate(config, str(checkpoint), output, height=31, width=37)
    assert output.is_file()
    assert result["max_abs_error"] <= 1e-4
    assert result["dynamic_max_abs_error"] <= 1e-4
