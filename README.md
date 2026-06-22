# UltraLiteDCE

UltraLiteDCE 是一个面向 CPU、移动端和 ONNX 部署的轻量低光照图像增强项目。它保留 Zero-DCE 的可微曲线：

```text
I_next = I + A * I * (1 - I)
```

默认模型仅在半分辨率预测一组 3 通道 RGB 曲线，并复用该曲线迭代 4 次。项目使用 LOL paired dataset 训练，同时组合 zero-reference 与 paired supervision，目标是降低参数量、MACs 和 CPU 延迟，并抑制过曝与颜色偏移。

> README 中未实际跑出的质量和速度数值均标为“待测”，不包含推测结果。

## 模型结构

```text
RGB input
  -> 0.5x bilinear resize
  -> 3x3 stem conv (3 -> width)
  -> N x depthwise separable conv
  -> 1x1 curve head
  -> tanh
  -> resize curve to original resolution
  -> shared/per-step Zero-DCE curve iterations
  -> clamp [0, 1]
```

默认配置为 `width=8`、3 个 depthwise separable block、4 次 shared curve 迭代、`prediction_scale=0.5`，且不使用 BatchNorm。增强公式已包含在模型 `forward` 中，使用的算子可导出 ONNX。

与常见 Zero-DCE-style baseline 的主要区别：

- curve predictor 默认宽度从 32 降到 8；
- 标准卷积替换为 depthwise separable convolution；
- curve map 在半分辨率预测；
- 默认只预测 3 通道共享 curve，而非 8 步共 24 通道；
- 默认迭代次数从 8 降到 4；
- 加入 paired L1、SSIM、Sobel edge 与 saturation protection；
- forward 直接输出最终增强图，便于部署。

`configs/baseline_dce.yaml` 提供 width=32、8 次 per-step curve、全分辨率和标准卷积的对照配置。`configs/ablation/` 中的局部配置会自动继承默认 YAML。

## 环境安装

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux/macOS：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如需与系统 CUDA 精确匹配，请先按 PyTorch 官方说明安装对应的 `torch` wheel，再安装其余依赖。LPIPS 与 NIQE 是可选项：

```bash
pip install lpips
pip install pyiqa
```

## LOL-v1 数据目录

默认路径：

```text
data/LOL/
├── our485/
│   ├── low/
│   └── high/
└── eval15/
    ├── low/
    └── high/
```

也可在 YAML 中设置：

```yaml
data:
  root: data/LOL
  train_low: our485/low
  train_high: our485/high
  test_low: eval15/low
  test_high: eval15/high
```

low/high 按不区分大小写的文件 stem 配对，因此 `scene.png` 可匹配 `scene.jpg`。若存在缺失或重复 stem，训练会在启动阶段给出明确错误。训练增强使用相同的 crop、翻转和 90° 旋转参数，不会破坏配对关系；评估保留原始分辨率，奇数尺寸可直接推理。

## 训练与恢复

```bash
python train.py --config configs/ultralite_dce.yaml
python train.py --config configs/ultralite_dce.yaml --resume outputs/checkpoints/last.pt
```

CLI 可覆盖常用参数，也可用任意点路径覆盖：

```bash
python train.py --config configs/ultralite_dce.yaml \
  --epochs 20 --batch-size 4 --device cuda \
  --set model.width=12 \
  --set model.curve_mode=per_step \
  --set model.num_iterations=6
```

没有 LOL 数据时可验证完整训练管线，但 synthetic 模式只用于 smoke test：

```bash
python train.py --config configs/ultralite_dce.yaml --synthetic \
  --epochs 1 --batch-size 2 --num-workers 0 \
  --set training.crop_size=64 \
  --set data.synthetic_train_size=4 \
  --set data.synthetic_val_size=2
```

训练功能包括 Adam、cosine scheduler、CUDA AMP、gradient clipping、固定种子、train/validation split、last/best checkpoint、CSV/TensorBoard 日志和定期图像预览。best checkpoint 按 validation PSNR 保存。

## 损失函数

每一项均单独记录到 `train_log.csv` 和 TensorBoard：

- Spatial consistency：约束输入与增强图的局部亮度梯度关系；
- Exposure control：将局部平均曝光约束到 `exposure_target`；
- Color constancy：抑制 RGB 通道均值偏移；
- Curve TV：约束 curve map 的空间平滑性；
- L1 reconstruction：paired 像素重建；
- SSIM loss：结构相似性；
- Edge preservation：Sobel 水平/垂直梯度 L1；
- Saturation protection：`mean(relu(enhanced - saturation_threshold))`。

权重、曝光目标和饱和阈值均在 YAML 的 `loss` 节配置。

## 评估与推理

```bash
python evaluate.py --config configs/ultralite_dce.yaml --checkpoint PATH
python infer.py --checkpoint PATH --input INPUT --output OUTPUT
```

`infer.py` 支持单张图像或目录。标准 checkpoint 内嵌完整配置，因此推理通常不需要 `--config`；旧 checkpoint 可显式传入配置。

评估保存：

```text
outputs/evaluation/
├── enhanced/
├── comparisons/
├── per_image_metrics.csv
└── summary.json
```

默认指标为 PSNR、SSIM、MAE；`--lpips` 和 `--niqe` 在安装可选依赖后启用。NIQE 是无参考指标，只对增强图计算。

## 参数量、MACs 与 CPU benchmark

```bash
python benchmark.py --config configs/ultralite_dce.yaml --checkpoint PATH
python benchmark.py --config configs/ultralite_dce.yaml \
  --sizes 256 512 --warmup 20 --iterations 100 --threads 1
```

benchmark 固定 batch size 1，使用 `torch.inference_mode()` 与 `time.perf_counter()`，先预热，再报告平均延迟、总体标准差和 FPS。MACs 由实际 Conv2d 输出尺寸和卷积参数统计，`approx_flops = 2 × MACs`。Python 无法可靠覆盖 PyTorch native allocator 的 CPU peak memory，因此当前没有伪造该字段。

## ONNX 导出、验证与量化准备

```bash
python export_onnx.py --checkpoint PATH \
  --output outputs/onnx/ultralite_dce.onnx
```

导出默认使用 opset 18，输入名为 `input`、输出名为 `enhanced`，batch/height/width 为动态轴。脚本执行 ONNX checker、ONNX Runtime 数值对齐，并报告最大/平均绝对误差与 ORT CPU benchmark。默认最大误差阈值是 `1e-4`。

卷积主导模型通常不会从 ONNX Runtime dynamic quantization 获得有效 INT8 卷积加速，因此项目不宣称 dynamic INT8 有效。可用代表性低光图做 calibrated static QDQ quantization：

```bash
python quantize_onnx.py \
  --model outputs/onnx/ultralite_dce.onnx \
  --output outputs/onnx/ultralite_dce_int8.onnx \
  --calibration-images data/LOL/our485/low \
  --limit 100
```

量化后的精度与速度必须在目标 ONNX Runtime execution provider 和硬件上重新测试。

## 配置参数

核心消融参数：

```yaml
model:
  width: 8                 # 8 / 12 / 16
  num_blocks: 3
  num_iterations: 4       # 2 / 4 / 6 / 8
  curve_mode: shared      # shared / per_step
  prediction_scale: 0.5   # 1.0 / 0.5 / 0.25
  convolution: depthwise_separable  # standard / depthwise_separable
```

`shared` 输出 3 通道 curve；`per_step` 输出 `3 × num_iterations` 通道。其余数据、loss、训练、评估、benchmark 和 ONNX 参数见 `configs/ultralite_dce.yaml`。

## 输出目录

```text
outputs/
├── checkpoints/
│   ├── last.pt
│   └── best.pt
├── previews/
├── tensorboard/
├── train_log.csv
├── resolved_config.yaml
├── evaluation/
├── benchmark.json
└── onnx/
```

## 测试

```bash
pytest -q
```

测试覆盖 shape/range、shared/per-step、奇数尺寸、paired crop 对齐、loss finite、backward、checkpoint、ONNX 一致性、单图推理和 synthetic training smoke test。缺少 ONNX 依赖时 ONNX 测试会明确 skip。

## 常见问题

- `Image directory does not exist`：检查 `data.root` 与四个 low/high 相对路径。
- `Low/high pairing failed`：检查缺失图片、命名差异或同一目录中的重复 stem。
- Windows DataLoader 卡住：先用 `--num-workers 0`，并确保从带 `if __name__ == "__main__"` 的脚本启动。
- CUDA OOM：减小 batch size/crop size，或使用 `prediction_scale=0.25`；验证固定 batch size 1。
- checkpoint 结构不匹配：推理使用训练时相同配置，或直接依赖 checkpoint 内嵌配置。
- ONNX 导出缺包：安装 `onnx onnxruntime`；新版 PyTorch 若提示还缺 `onnxscript`，按提示安装。
- ONNX 数值误差超阈值：确认模型为 eval 模式、输入范围 `[0,1]`，并检查 execution provider。
- PSNR/SSIM 很低：确认 low/high 配对和方向正确；synthetic smoke test 的指标不能作为实验结果。

## 实验与消融表

下表中的参数量与 256×256 Conv MACs/FLOPs 已由项目 hook 实际统计；UltraLite-Shared CPU 延迟来自本次本机 PyTorch 2.12.1 CPU、batch=1、单线程、20 次预热和 100 次计时。LOL PSNR/SSIM 尚未训练评估，保持待测。

| Model             | Width | Curve maps | Iterations | Scale | Params | FLOPs | PSNR | SSIM | CPU ms |
| ----------------- | ----: | ---------: | ---------: | ----: | -----: | ----: | ---: | ---: | -----: |
| Baseline          |    32 |         24 |          8 |   1.0 | 66,424 |  8.67G | 待测 | 待测 |   待测 |
| UltraLite-Shared  |     8 |          3 |          4 |   0.5 |    707 | 21.23M | 待测 | 待测 |    3.36 |
| UltraLite-PerStep |     8 |         12 |          4 |   0.5 |    788 | 23.59M | 待测 | 待测 |   待测 |

训练结果应同时记录随机种子、数据划分、输入 crop、checkpoint、PyTorch/ONNX Runtime 版本、CPU 型号和线程数。
