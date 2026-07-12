# UltraLiteDCE

UltraLiteDCE is a lightweight PyTorch project for low-light image enhancement (LLIE) on the LOL paired dataset. It keeps the core Zero-DCE curve formulation:

```text
I_next = I + A * I * (1 - I)
```

The goal is to keep the model small and deployment-friendly while improving visual naturalness, especially in dark regions where color noise and over-enhancement often appear.

The original Chinese README has been preserved as [README_cn.md](README_cn.md).

## Highlights

- PyTorch implementation of a lightweight Zero-DCE-style LLIE model.
- LOL-v1 paired dataset support.
- Shared and per-step curve modes.
- Half-resolution curve prediction.
- Depthwise separable convolution support.
- No BatchNorm in the default model.
- Color-safe curve parameterization.
- Dark-region smoothness loss for noise suppression.
- Paired supervision with L1, SSIM, and Sobel edge losses.
- Training, evaluation, inference, benchmark, ONNX export, and comparison scripts.
- ONNX Runtime validation with dynamic batch/height/width support.
- ZeroDCE-style baseline comparison workflow.

## Model Architecture

Default UltraLiteDCE:

```text
RGB input: B x 3 x H x W
  -> 0.5x bilinear resize
  -> 3x3 stem conv: 3 -> width
  -> N x depthwise separable conv
  -> 1x1 curve head
  -> tanh / coupled curve parameterization
  -> upsample curve map to H x W
  -> shared or per-step Zero-DCE curve iterations
  -> optional dark denoise head
  -> clamp to [0, 1]
```

Default optimized configuration:

```yaml
model:
  name: ultralite_dce
  width: 8
  num_blocks: 3
  num_iterations: 4
  curve_mode: shared
  prediction_scale: 0.5
  convolution: depthwise_separable
  curve_color_mode: coupled
  curve_chroma_scale: 0.05
  identity_init: true
  use_dark_denoise_head: false
```

## Difference from the ZeroDCE-style Baseline

Compared with the heavier ZeroDCE-style baseline used in this repository:

- width is reduced from 32 to 8;
- standard convolutions are replaced by depthwise separable convolutions;
- curve maps are predicted at half resolution;
- the default model predicts one shared RGB curve map instead of 8 per-step RGB maps;
- curve iterations are reduced from 8 to 4;
- BatchNorm is not used;
- forward inference directly returns the enhanced image;
- color-safe initialization and curve coupling reduce fixed color bias;
- dark-region smoothness and saturation losses reduce noise and overexposure.

The baseline configuration is available at:

```text
configs/comparison/zerodce_baseline_60.yaml
```

## Project Structure

```text
configs/
├── ultralite_dce.yaml
├── ultralite_dce_denoise.yaml
├── baseline_dce.yaml
├── ablation/
└── comparison/
    ├── ultralite_color_safe_gpu_denoise_60.yaml
    └── zerodce_baseline_60.yaml

models/
├── blocks.py
├── curve.py
└── ultralite_dce.py

losses/
├── zero_reference.py
├── supervised.py
└── combined.py

datasets/
└── lol_dataset.py

utils/
├── benchmark.py
├── checkpoint.py
├── config.py
├── image.py
├── metrics.py
└── seed.py

train.py
evaluate.py
infer.py
benchmark.py
export_onnx.py
compare_baseline.py
quantize_onnx.py
tests/
```

## Environment Setup

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For GPU training, install the PyTorch build that matches your CUDA driver. Optional metrics:

```bash
pip install lpips
pip install pyiqa
```

For ONNX export and validation:

```bash
pip install onnx onnxruntime onnxscript
```

## LOL Dataset Layout

Default expected layout:

```text
data/LOL/
├── our485/
│   ├── low/
│   └── high/
└── eval15/
    ├── low/
    └── high/
```

YAML fields:

```yaml
data:
  root: data/LOL
  train_low: our485/low
  train_high: our485/high
  test_low: eval15/low
  test_high: eval15/high
```

Low/high images are matched by filename stem. Training uses paired random crop and paired geometric augmentation, so low/high alignment is preserved. Evaluation and inference keep the original image resolution.

## Loss Functions

Each loss component is logged separately.

Zero-reference losses:

- `spatial`: local spatial consistency.
- `exposure`: exposure control toward `exposure_target`.
- `color`: RGB channel mean consistency.
- `curve_tv`: curve map total variation smoothness.

Paired losses:

- `l1`: pixel reconstruction loss.
- `ssim`: structural similarity loss.
- `edge`: Sobel gradient edge preservation.
- `chromaticity`: optional RGB ratio constraint.

Noise and overexposure protection:

- `dark_smooth`: suppresses high-frequency variations in dark input regions.
- `saturation`: `mean(relu(enhanced - saturation_threshold))`.

Optimized loss configuration:

```yaml
loss:
  spatial: 1.0
  exposure: 0.3
  color: 0.5
  curve_tv: 50.0
  l1: 2.0
  ssim: 0.5
  edge: 0.2
  chromaticity: 0.0
  dark_smooth: 0.2
  saturation: 0.2
  exposure_target: 0.55
  dark_threshold: 0.25
  saturation_threshold: 0.95
```

## Training

Recommended optimized UltraLiteDCE training:

```powershell
python train.py `
  --config configs\ultralite_dce_denoise.yaml `
  --epochs 100 `
  --batch-size 8 `
  --num-workers 2 `
  --device cuda `
  --output-dir outputs\ultralite_dce_denoise
```

Resume training:

```powershell
python train.py `
  --config configs\ultralite_dce_denoise.yaml `
  --resume outputs\ultralite_dce_denoise\checkpoints\last.pt
```

The 60-epoch UltraLiteDCE configuration used for the final comparison:

```powershell
python train.py `
  --config configs\comparison\ultralite_color_safe_gpu_denoise_60.yaml `
  --epochs 60 `
  --num-workers 2 `
  --device cuda
```

The 60-epoch ZeroDCE-style baseline:

```powershell
python train.py `
  --config configs\comparison\zerodce_baseline_60.yaml `
  --epochs 60 `
  --num-workers 2 `
  --device cuda
```

Training outputs:

```text
outputs/<run_name>/
├── checkpoints/
│   ├── last.pt
│   ├── best.pt
│   ├── best_psnr.pt
│   ├── best_ssim.pt
│   └── final.pt
├── previews/
│   ├── epoch_XXXX_low_enhanced_gt.png
│   ├── epoch_XXXX_enhanced.png
│   └── epoch_XXXX_curve_map.png
├── tensorboard/
├── train_log.csv
├── train_loss.csv
├── validation_loss.csv
└── resolved_config.yaml
```

`best.pt` and `best_psnr.pt` are selected by validation PSNR. `best_ssim.pt` is selected by validation SSIM.

## Evaluation

Evaluate on LOL eval15:

```powershell
python evaluate.py `
  --config configs\ultralite_dce_denoise.yaml `
  --checkpoint outputs\ultralite_dce_denoise\checkpoints\best.pt `
  --device cuda `
  --output-dir outputs\ultralite_dce_denoise\evaluation
```

Saved outputs:

```text
evaluation/
├── enhanced/
├── comparisons/
├── per_image_metrics.csv
└── summary.json
```

Metrics include:

- PSNR
- SSIM
- MAE
- parameters
- checkpoint size
- MACs
- CPU inference time
- FPS

Optional metrics:

```powershell
python evaluate.py --config CONFIG --checkpoint CKPT --lpips --niqe
```

## Inference

Single image:

```powershell
python infer.py `
  --checkpoint outputs\ultralite_dce_denoise\checkpoints\best.pt `
  --input path\to\low.png `
  --output outputs\infer\enhanced.png
```

Folder:

```powershell
python infer.py `
  --checkpoint outputs\ultralite_dce_denoise\checkpoints\best.pt `
  --input data\LOL\eval15\low `
  --output outputs\infer_eval15
```

## CPU Benchmark

```powershell
python benchmark.py `
  --config configs\ultralite_dce_denoise.yaml `
  --checkpoint outputs\ultralite_dce_denoise\checkpoints\best.pt `
  --sizes 256 512 `
  --warmup 20 `
  --iterations 100 `
  --threads 1
```

Benchmark uses batch size 1, `torch.inference_mode()`, warm-up iterations, and `time.perf_counter()`.

## ONNX Export

Export and validate UltraLiteDCE:

```powershell
python export_onnx.py `
  --config configs\comparison\ultralite_color_safe_gpu_denoise_60.yaml `
  --checkpoint outputs\color_safe_gpu_denoise\checkpoints\best_psnr.pt `
  --output outputs\onnx\ultralite_color_safe_gpu_denoise.onnx `
  --height 256 `
  --width 256 `
  --warmup 20 `
  --iterations 100
```

Export and validate the ZeroDCE-style baseline:

```powershell
python export_onnx.py `
  --config configs\comparison\zerodce_baseline_60.yaml `
  --checkpoint outputs\zerodce_baseline_60\checkpoints\best_psnr.pt `
  --output outputs\onnx\zerodce_baseline_60.onnx `
  --height 256 `
  --width 256 `
  --warmup 20 `
  --iterations 100
```

ONNX export uses:

- opset 18 by default;
- input name: `input`;
- output name: `enhanced`;
- dynamic batch, height, and width;
- ONNX checker validation;
- ONNX Runtime CPU output comparison;
- ONNX Runtime CPU benchmark.

The export report is saved next to the `.onnx` file as `.json`.

Dynamic INT8 quantization is not claimed to be effective for this convolution-heavy model. Static QDQ quantization can be prepared with representative low-light calibration images:

```powershell
python quantize_onnx.py `
  --model outputs\onnx\ultralite_color_safe_gpu_denoise.onnx `
  --output outputs\onnx\ultralite_color_safe_gpu_denoise_int8.onnx `
  --calibration-images data\LOL\our485\low `
  --limit 100
```

## 60-Epoch UltraLiteDCE vs ZeroDCE-style Baseline

Run the full comparison:

```powershell
python compare_baseline.py --device cuda
```

Quick smoke comparison:

```powershell
python compare_baseline.py --quick --device cuda
```

Outputs:

```text
outputs/comparison_60/
├── comparison_summary.csv
├── comparison_summary.json
├── comparison_table.md
├── ultralitedce_color_safe_gpu_denoise/
│   ├── evaluation/
│   └── onnx/
└── zerodce_style_baseline/
    ├── evaluation/
    └── onnx/
```

Final measured comparison from the current workspace:

| Model | Curve Mode | Iterations | Scale | Params | MACs 256 | PSNR | SSIM | MAE | PyTorch CPU 256 ms | ONNX CPU 256 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UltraLiteDCE color_safe_gpu_denoise | shared | 4 | 0.5 | 707 | 10,616,832 | 18.5606 | 0.5509 | 0.1210 | 3.5953 | 1.3367 |
| ZeroDCE-style baseline | per_step | 8 | 1.0 | 66,424 | 4,334,813,184 | 18.1758 | 0.5561 | 0.1164 | 79.7110 | 9.1191 |

Additional 512x512 PyTorch CPU benchmark:

| Model | CPU 512 ms | FPS 512 |
|---|---:|---:|
| UltraLiteDCE color_safe_gpu_denoise | 14.7501 | 67.7964 |
| ZeroDCE-style baseline | 307.8673 | 3.2482 |

ONNX validation errors:

| Model | Max Abs Error | Dynamic Max Abs Error |
|---|---:|---:|
| UltraLiteDCE color_safe_gpu_denoise | 2.38e-7 | 2.98e-7 |
| ZeroDCE-style baseline | 2.98e-7 | 3.58e-7 |

Notes:

- Both models were trained for 60 epochs.
- Metrics are measured on LOL eval15.
- CPU timing uses batch size 1.
- ONNX timing uses ONNX Runtime CPUExecutionProvider.
- The baseline has slightly higher SSIM and lower MAE in this run, while UltraLiteDCE has higher PSNR and is dramatically smaller/faster.

## Tests

Run all tests:

```powershell
pytest -q
```

Current verified result:

```text
22 passed
```

The test suite covers:

- model shape and output range;
- shared and per-step curve modes;
- odd image sizes;
- paired crop alignment;
- finite losses;
- backward gradients;
- checkpoint save/load;
- ONNX export and output consistency;
- single-image inference integration;
- synthetic training smoke test.

## Troubleshooting

- `CUDA was requested but is not available`: install a CUDA-enabled PyTorch build or use `--device cpu`.
- Windows DataLoader hangs: retry with `--num-workers 0`.
- Purple or strong color cast: do not resume old independent RGB curve checkpoints; retrain with the coupled curve configuration.
- Dark-region color noise: use the optimized denoise config and inspect `dark_smooth`, `curve_tv`, and preview images.
- Overexposure: reduce `loss.exposure`, reduce `exposure_target`, or increase `loss.saturation`.
- Low PSNR/SSIM: verify LOL low/high pairing and dataset paths.
- ONNX package missing: install `onnx`, `onnxruntime`, and `onnxscript`.
- ONNX mismatch: ensure the checkpoint and config match exactly.

