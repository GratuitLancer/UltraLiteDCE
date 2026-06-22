from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from infer import run_inference
from models import build_model
from utils.checkpoint import save_checkpoint


def _config(tmp_path: Path) -> dict:
    return {
        "model": {
            "name": "ultralite_dce",
            "width": 8,
            "num_blocks": 3,
            "num_iterations": 4,
            "curve_mode": "shared",
            "prediction_scale": 0.5,
            "convolution": "depthwise_separable",
        },
        "training": {"device": "cpu", "output_dir": str(tmp_path / "outputs")},
    }


def test_single_image_inference_integration(tmp_path: Path) -> None:
    config = _config(tmp_path)
    model = build_model(config)
    checkpoint = tmp_path / "model.pt"
    save_checkpoint({"model": model.state_dict(), "config": config}, checkpoint)
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "result.png"
    Image.fromarray(
        np.random.default_rng(0).integers(0, 256, size=(35, 41, 3), dtype=np.uint8),
        mode="RGB",
    ).save(input_path)
    outputs = run_inference(
        checkpoint,
        input_path,
        output_path,
        config=config,
        device_name="cpu",
    )
    assert outputs == [output_path]
    assert output_path.is_file()
    with Image.open(output_path) as result:
        assert result.size == (41, 35)
        assert result.mode == "RGB"

