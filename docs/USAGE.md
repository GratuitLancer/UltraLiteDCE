# Usage

This page keeps the operational commands out of the project landing page.

## Environment

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

For GPU training, install a CUDA-enabled PyTorch build that matches the machine.

Optional quality metrics:

```bash
pip install lpips pyiqa
```

## LOL-v1 dataset layout

```text
data/LOL/
├── our485/
│   ├── low/
│   └── high/
└── eval15/
    ├── low/
    └── high/
```

Low/high images are matched by filename stem. Training preserves paired alignment during random crop and geometric augmentation.

## Train UltraLiteDCE

Recommended comparison configuration:

```powershell
python train.py `
  --config configs\comparison\ultralite_color_safe_gpu_denoise_60.yaml `
  --epochs 60 `
  --num-workers 2 `
  --device cuda
```

Linux/macOS equivalent:

```bash
python train.py \
  --config configs/comparison/ultralite_color_safe_gpu_denoise_60.yaml \
  --epochs 60 \
  --num-workers 2 \
  --device cuda
```

Resume:

```bash
python train.py \
  --config configs/comparison/ultralite_color_safe_gpu_denoise_60.yaml \
  --resume outputs/color_safe_gpu_denoise/checkpoints/last.pt
```

Typical training outputs:

```text
outputs/<run_name>/
├── checkpoints/
│   ├── last.pt
│   ├── best.pt
│   ├── best_psnr.pt
│   ├── best_ssim.pt
│   └── final.pt
├── previews/
├── tensorboard/
├── train_log.csv
├── train_loss.csv
├── validation_loss.csv
└── resolved_config.yaml
```

## Evaluate

```bash
python evaluate.py \
  --config configs/comparison/ultralite_color_safe_gpu_denoise_60.yaml \
  --checkpoint outputs/color_safe_gpu_denoise/checkpoints/best_psnr.pt \
  --device cuda \
  --output-dir outputs/color_safe_gpu_denoise/evaluation
```

Saved evaluation output includes enhanced images, visual comparisons, per-image metrics, and a summary.

Core metrics:

- PSNR
- SSIM
- MAE
- parameter count
- checkpoint size
- MACs
- CPU latency
- FPS

Optional:

```bash
python evaluate.py --config CONFIG --checkpoint CKPT --lpips --niqe
```

## Inference

Single image:

```bash
python infer.py \
  --checkpoint outputs/color_safe_gpu_denoise/checkpoints/best_psnr.pt \
  --input path/to/low.png \
  --output outputs/infer/enhanced.png
```

Folder:

```bash
python infer.py \
  --checkpoint outputs/color_safe_gpu_denoise/checkpoints/best_psnr.pt \
  --input data/LOL/eval15/low \
  --output outputs/infer_eval15
```

## CPU benchmark

```bash
python benchmark.py \
  --config configs/comparison/ultralite_color_safe_gpu_denoise_60.yaml \
  --checkpoint outputs/color_safe_gpu_denoise/checkpoints/best_psnr.pt \
  --sizes 256 512 \
  --warmup 20 \
  --iterations 100 \
  --threads 1
```

The benchmark uses batch size 1, `torch.inference_mode()`, warm-up runs, and `time.perf_counter()`.

## ONNX export

```bash
python export_onnx.py \
  --config configs/comparison/ultralite_color_safe_gpu_denoise_60.yaml \
  --checkpoint outputs/color_safe_gpu_denoise/checkpoints/best_psnr.pt \
  --output outputs/onnx/ultralite_dce.onnx \
  --height 256 \
  --width 256 \
  --warmup 20 \
  --iterations 100
```

The exporter validates:

- ONNX checker compatibility
- PyTorch vs ONNX Runtime output agreement
- dynamic batch size
- dynamic height and width
- ONNX Runtime CPU latency

## Static INT8 preparation

For convolution-heavy networks, dynamic INT8 quantization is not necessarily useful. The repository instead includes static QDQ preparation with representative calibration images.

```bash
python quantize_onnx.py \
  --model outputs/onnx/ultralite_dce.onnx \
  --output outputs/onnx/ultralite_dce_int8.onnx \
  --calibration-images data/LOL/our485/low \
  --limit 100
```

## Full baseline comparison

```bash
python compare_baseline.py --device cuda
```

Smoke test:

```bash
python compare_baseline.py --quick --device cuda
```

## Tests

```bash
pytest -q
```

The test suite covers the model, curve modes, odd image sizes, paired dataset alignment, losses, gradients, checkpointing, ONNX consistency, inference integration, and synthetic training.
