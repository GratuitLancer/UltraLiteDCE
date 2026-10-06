# Benchmarks

This page records the measured comparison currently documented in the repository.

## Setup

Both models were trained for 60 epochs and evaluated on LOL-v1 `eval15`.

The comparison is intended to study the engineering trade-off between the lightweight UltraLiteDCE configuration and the repository's heavier ZeroDCE-style baseline.

CPU timing uses batch size 1. ONNX measurements use ONNX Runtime CPUExecutionProvider.

## Quality and efficiency

| Model | Params | MACs @256 | PSNR | SSIM | MAE | PyTorch CPU @256 | ONNX CPU @256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| UltraLiteDCE | 707 | 10,616,832 | 18.5606 | 0.5509 | 0.1210 | 3.5953 ms | 1.3367 ms |
| ZeroDCE-style baseline | 66,424 | 4,334,813,184 | 18.1758 | 0.5561 | 0.1164 | 79.7110 ms | 9.1191 ms |

### Relative size

- parameter count: approximately **94× smaller**
- MACs @256: approximately **408× lower**
- PyTorch CPU @256: approximately **22× faster**
- ONNX CPU @256: approximately **6.8× faster**

These ratios come from the measured values above and are not hardware-independent guarantees.

## 512×512 PyTorch CPU

| Model | CPU latency | FPS |
|---|---:|---:|
| UltraLiteDCE | 14.7501 ms | 67.7964 |
| ZeroDCE-style baseline | 307.8673 ms | 3.2482 |

## ONNX numerical validation

| Model | Max abs error | Dynamic-shape max abs error |
|---|---:|---:|
| UltraLiteDCE | 2.38e-7 | 2.98e-7 |
| ZeroDCE-style baseline | 2.98e-7 | 3.58e-7 |

The errors indicate close numerical agreement between PyTorch and ONNX Runtime for the tested inputs.

## Interpretation

The useful result is not that UltraLiteDCE wins every image-quality metric.

In this run:

- UltraLiteDCE has slightly higher PSNR.
- The heavier baseline has slightly higher SSIM and lower MAE.
- UltraLiteDCE is dramatically smaller and faster.

That makes the project primarily an efficiency / deployment experiment.

## Reproduce

Run the full comparison:

```bash
python compare_baseline.py --device cuda
```

Expected output structure:

```text
outputs/comparison_60/
├── comparison_summary.csv
├── comparison_summary.json
├── comparison_table.md
├── ultralitedce_color_safe_gpu_denoise/
└── zerodce_style_baseline/
```

## Caveat

The baseline in this repository is described as **ZeroDCE-style**.

It restores a heavier network shape and per-step curve prediction, but the table should not be presented as a strict reproduction of the original paper's full experimental setup.
