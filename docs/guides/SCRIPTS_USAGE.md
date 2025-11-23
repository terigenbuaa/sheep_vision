# 脚本使用指南

本文档说明所有脚本的命令行参数和使用方法。

> 💡 **新手入门**：如果您是第一次使用本项目，建议先查看 [QUICK_START.md](../../QUICK_START.md) 快速上手指南。

## 目录

- [训练脚本](#训练脚本)
- [评估脚本](#评估脚本)
- [测试脚本](#测试脚本)
- [分析脚本](#分析脚本)
- [工具脚本](#工具脚本)

---

## 训练脚本

### train.py - 标准训练脚本

训练RF-DETR模型。

```bash
python scripts/train.py [选项]
```

**主要参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--dataset_dir` | str | `dataset` | 数据集根目录 |
| `--output_dir` | str | `output` | 模型输出目录 |
| `--epochs` | int | `100` | 训练轮数 |
| `--batch_size` | int | `8` | 批次大小 |
| `--grad_accum_steps` | int | `2` | 梯度累积步数 |
| `--lr` | float | `1e-4` | 学习率 |
| `--lr_encoder` | float | `1.5e-4` | 编码器学习率 |
| `--resume` | str | `None` | 恢复训练的检查点路径 |
| `--no_tensorboard` | flag | - | 禁用TensorBoard |
| `--no_early_stopping` | flag | - | 禁用早停 |
| `--early_stopping_patience` | int | `10` | 早停耐心值 |
| `--checkpoint_interval` | int | `10` | 检查点保存间隔 |
| `--no_ema` | flag | - | 禁用EMA |
| `--eval` | flag | - | 仅评估模式 |

**示例：**

```bash
# 使用默认参数训练
python scripts/train.py

# 指定数据集和输出目录
python scripts/train.py --dataset_dir dataset --output_dir output

# 自定义训练参数
python scripts/train.py --epochs 50 --batch_size 16 --lr 2e-4

# 恢复训练
python scripts/train.py --resume checkpoints/finetuned/default_model/checkpoints/latest.pth

# 禁用某些功能
python scripts/train.py --no_tensorboard --no_ema
```

### train_optimized.py - 优化训练脚本

针对损失偏高问题的优化训练脚本（参数与train.py相同）。

---

## 评估脚本

### evaluate_model.py - 模型评估

在数据集上评估模型性能。

```bash
python scripts/evaluate_model.py [选项]
```

**主要参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--checkpoint` | str | `output/checkpoint_best_total.pth` | 模型检查点路径 |
| `--dataset_dir` | str | `dataset` | 数据集目录 |
| `--split` | str | `test` | 数据集分割: `train`/`valid`/`test`/`all` |
| `--threshold` | float | `0.3` | 置信度阈值 |

**示例：**

```bash
# 评估测试集（默认）
python scripts/evaluate_model.py

# 评估指定数据集
python scripts/evaluate_model.py --split valid
python scripts/evaluate_model.py --split all

# 使用自定义模型和阈值
python scripts/evaluate_model.py --checkpoint checkpoints/finetuned/default_model/best/checkpoint_best_ema.pth --threshold 0.5
```

---

## 测试脚本

### test_model.py - 单张图片测试

测试单张图片的检测结果。

```bash
python scripts/test_model.py [选项]
```

**主要参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--image` | str | `data/eating.jpg` | 输入图片路径 |
| `--checkpoint` | str | `output/checkpoint_best_total.pth` | 模型检查点路径 |
| `--threshold` | float | `0.3` | 置信度阈值 |
| `--output` | str | `None` | 输出图片路径（默认自动生成） |
| `--dataset_dir` | str | `None` | 数据集目录（用于显示真实标注） |

**示例：**

```bash
# 使用默认图片测试
python scripts/test_model.py

  # 测试指定图片
  python scripts/test_model.py --image data/test/images/test.jpg

# 使用自定义模型和阈值
python scripts/test_model.py --checkpoint checkpoints/finetuned/default_model/best/checkpoint_best_ema.pth --threshold 0.5

# 指定输出文件
python scripts/test_model.py --image test.jpg --output result.jpg
```

### quick_test.py - 快速测试

快速测试模型（使用数据集中的第一张图片）。

```bash
python scripts/quick_test.py [选项]
```

**主要参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--checkpoint` | str | `output/checkpoint_best_total.pth` | 模型检查点路径 |
| `--dataset_dir` | str | `dataset` | 数据集目录 |
| `--threshold` | float | `0.3` | 置信度阈值 |
| `--output` | str | `None` | 输出图片路径（默认自动生成） |
| `--split` | str | `None` | 指定数据集分割（train/valid/test） |

**示例：**

```bash
# 使用默认参数快速测试
python scripts/quick_test.py

# 指定数据集目录
python scripts/quick_test.py --dataset_dir dataset

# 使用自定义模型和阈值
python scripts/quick_test.py --checkpoint checkpoints/finetuned/default_model/best/checkpoint_best_ema.pth --threshold 0.5

# 指定输出文件
python scripts/quick_test.py --output output/my_test_result.jpg
```

### test_dataset.py - 批量测试

在数据集上进行批量测试并保存结果。

```bash
python scripts/test_dataset.py [选项]
```

**主要参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--dataset_dir` | str | `dataset/test` | 数据集目录 |
| `--checkpoint` | str | `output/checkpoint_best_total.pth` | 模型检查点路径 |
| `--threshold` | float | `0.3` | 置信度阈值 |
| `--output_dir` | str | `None` | 输出目录（默认自动生成） |

**示例：**

```bash
# 测试默认数据集
python scripts/test_dataset.py

# 测试指定数据集目录
python scripts/test_dataset.py --dataset_dir dataset/test

# 使用自定义模型和阈值
python scripts/test_dataset.py --checkpoint checkpoints/finetuned/default_model/best/checkpoint_best_ema.pth --threshold 0.5

# 指定输出目录
python scripts/test_dataset.py --dataset_dir /mnt/f/data --output_dir output/my_analysis
```

---

## 分析脚本

### analyze_video.py - 视频分析

对视频进行目标检测分析。

```bash
python scripts/analyze_video.py --video <视频路径> [选项]
```

**主要参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--video` | str | **必需** | 视频文件路径 |
| `--checkpoint` | str | `output/checkpoint_best_total.pth` | 模型检查点路径 |
| `--threshold` | float | `0.3` | 置信度阈值 |
| `--frame_interval` | int | `30` | 帧间隔（每隔N帧分析一次） |
| `--output_dir` | str | `None` | 输出目录（默认自动生成） |

**示例：**

```bash
  # 分析视频（必需参数）
  python scripts/analyze_video.py --video data/test/videos/video.mp4

  # 使用自定义模型和参数
  python scripts/analyze_video.py --video data/test/videos/video.mp4 --checkpoint checkpoints/finetuned/default_model/best/checkpoint_best_ema.pth --threshold 0.5

# 指定帧间隔和输出目录
python scripts/analyze_video.py --video video.mp4 --frame_interval 60 --output_dir output/my_analysis
```

---

## 工具脚本

### generate_time_table.py - 生成时间统计表格

从测试结果生成时间统计表格（CSV和Excel）。

```bash
python scripts/generate_time_table.py [选项]
```

**主要参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` | str | `output/test_batch_results/test_results.json` | 输入JSON文件路径 |
| `--csv` | str | `None` | 输出CSV文件路径（默认自动生成） |
| `--excel` | str | `None` | 输出Excel文件路径（默认自动生成） |

**示例：**

```bash
# 使用默认路径生成表格
python scripts/generate_time_table.py

# 指定输入JSON文件
python scripts/generate_time_table.py --input output/analysis_test/test_results.json

# 指定所有输出文件
python scripts/generate_time_table.py --input results.json --csv output.csv --excel output.xlsx
```

### prepare_dataset.py - 准备数据集

准备RF-DETR训练数据集（已有argparse支持）。

```bash
python scripts/prepare_dataset.py --input_json <json文件> --images_dir <图片目录> [选项]
```

### merge_datasets.py - 合并数据集

合并多个Roboflow数据集（使用默认参数）。

---

## 快速参考

### 完整工作流程示例

```bash
# 1. 准备数据集
python scripts/prepare_dataset.py --input_json annotations.json --images_dir images --output_dir dataset

# 2. 训练模型
python scripts/train.py --dataset_dir dataset --output_dir output --epochs 100

# 3. 保存微调模型
./scripts/save_finetuned_model.sh my_model

# 4. 评估模型
python scripts/evaluate_model.py --checkpoint checkpoints/finetuned/my_model/best/checkpoint_best_ema.pth --split all

# 5. 测试单张图片
python scripts/test_model.py --image data/test.jpg --checkpoint checkpoints/finetuned/my_model/best/checkpoint_best_ema.pth

# 6. 批量测试
python scripts/test_dataset.py --dataset_dir dataset/test --checkpoint checkpoints/finetuned/my_model/best/checkpoint_best_ema.pth

# 7. 分析视频
python scripts/analyze_video.py --video data/video.mp4 --checkpoint checkpoints/finetuned/my_model/best/checkpoint_best_ema.pth

# 8. 生成统计表格
python scripts/generate_time_table.py --input output/analysis_test/test_results.json
```

### 获取帮助

所有脚本都支持 `--help` 参数查看详细帮助：

```bash
python scripts/train.py --help
python scripts/evaluate_model.py --help
python scripts/test_model.py --help
# ... 等等
```

---

## 注意事项

1. **路径格式**：
   - Linux/WSL: 使用 `/path/to/file`
   - Windows路径在WSL中: `/mnt/f/path/to/file`
   - 可以使用相对路径或绝对路径

2. **模型路径**：
   - 默认使用 `output/checkpoint_best_total.pth`
   - 推荐使用 `checkpoints/finetuned/<模型名>/best/checkpoint_best_ema.pth`

3. **数据集格式**：
   - 数据集应为COCO格式
   - 包含 `train/`, `valid/`, `test/` 子目录
   - 每个子目录包含 `_annotations.coco.json` 和图片文件

4. **输出目录**：
   - 大多数脚本会自动生成输出目录
   - 也可以使用 `--output_dir` 指定

---

## 常见问题

**Q: 如何查看所有可用参数？**
A: 运行脚本时添加 `--help` 参数，例如：`python scripts/train.py --help`

**Q: 如何恢复训练？**
A: 使用 `--resume` 参数指定检查点路径：`python scripts/train.py --resume output/checkpoint.pth`

**Q: 如何在不同数据集上测试？**
A: 使用 `--dataset_dir` 参数指定数据集目录

**Q: 如何调整检测阈值？**
A: 使用 `--threshold` 参数，例如：`--threshold 0.5`

**Q: 如何保存微调模型？**
A: 使用 `./scripts/save_finetuned_model.sh <模型名称>` 脚本

