from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from utils.image import list_images, load_image


class ImageCalibrationReader:
    def __init__(self, image_dir: str | Path, limit: int | None = None) -> None:
        paths = list_images(image_dir)
        self.paths = paths[:limit] if limit else paths
        self.index = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self.index >= len(self.paths):
            return None
        image = load_image(self.paths[self.index]).unsqueeze(0).numpy()
        self.index += 1
        return {"input": image}

    def rewind(self) -> None:
        self.index = 0


def quantize_static_model(
    model_input: str,
    model_output: str,
    calibration_images: str,
    limit: int | None = None,
) -> None:
    try:
        from onnxruntime.quantization import (
            CalibrationMethod,
            QuantFormat,
            QuantType,
            quantize_static,
        )
    except ImportError as exc:
        raise ImportError("Static quantization requires `pip install onnxruntime`") from exc
    reader = ImageCalibrationReader(calibration_images, limit)
    Path(model_output).parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        model_input,
        model_output,
        reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a calibrated static INT8 ONNX model. Dynamic quantization is not "
            "used because Conv-heavy models generally do not benefit from it."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--calibration-images", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    quantize_static_model(args.model, args.output, args.calibration_images, args.limit)
    print(f"Saved statically quantized model to {args.output}")


if __name__ == "__main__":
    main()

