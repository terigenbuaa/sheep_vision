# RF-DETR 快速上手指南

本指南帮助您快速开始使用RF-DETR项目进行目标检测训练和推理。

## 📋 目录

- [环境准备](#环境准备)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [完整工作流程](#完整工作流程)
- [常用命令](#常用命令)
- [常见问题](#常见问题)

---

## 🚀 环境准备

### 1. 安装依赖

```bash
# 安装RF-DETR包
pip install rfdetr

# 或从源码安装
pip install git+https://github.com/roboflow/rf-detr.git
```

### 2. 验证安装

```bash
python -c "from rfdetr import RFDETRBase; print('✅ RF-DETR安装成功')"
```

### 3. 检查GPU（可选但推荐）

```bash
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
```

---

## 📁 项目结构

```
rf-detr/
├── scripts/              # 所有脚本文件
│   ├── train.py         # 训练脚本
│   ├── evaluate_model.py # 评估脚本
│   ├── test_model.py    # 单张图片测试
│   └── ...
├── dataset/             # 训练数据集（COCO格式）
│   ├── train/
│   ├── valid/
│   └── test/
├── checkpoints/         # 模型文件
│   ├── pretrained/     # 预训练模型
│   └── finetuned/      # 微调模型
├── output/             # 训练输出
└── data/               # 测试数据文件
    ├── test/
    │   ├── images/     # 测试图片
    │   └── videos/     # 测试视频
    └── train/          # 训练数据（可选）
```

详细结构说明请参考 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

**重要提示**：
- `dataset/` 目录用于训练数据集（COCO格式，包含train/valid/test子目录）
- `data/` 目录用于测试图片和视频（test/images/ 和 test/videos/）
- 训练时使用 `dataset/`，测试时可以使用 `data/test/` 中的文件

---

## ⚡ 快速开始

### 方式1: 使用预训练模型（最快）

如果您只是想测试模型，可以直接使用预训练模型：

```bash
# 1. 快速测试（使用默认数据集）
python scripts/quick_test.py

# 2. 测试单张图片
python scripts/test_model.py --image data/test/images/eating.jpg

# 3. 评估模型
python scripts/evaluate_model.py
```

### 方式2: 训练自己的模型

如果您有自己的数据集，可以按照以下步骤训练：

#### 步骤1: 准备数据集

```bash
# 使用prepare_dataset.py准备数据集
python scripts/prepare_dataset.py \
    --input_json your_annotations.json \
    --images_dir your_images/ \
    --output_dir dataset
```

数据集格式要求：
- COCO格式的JSON标注文件
- 图片文件（JPG/PNG）
- 目录结构：`dataset/train/`, `dataset/valid/`, `dataset/test/`

#### 步骤2: 开始训练

```bash
# 使用默认参数训练
python scripts/train.py

# 或自定义参数
python scripts/train.py \
    --dataset_dir dataset \
    --output_dir output \
    --epochs 100 \
    --batch_size 8
```

#### 步骤3: 评估模型

```bash
# 评估测试集
python scripts/evaluate_model.py --split test

# 评估所有数据集
python scripts/evaluate_model.py --split all
```

#### 步骤4: 保存微调模型

```bash
# 保存训练好的模型
./scripts/save_finetuned_model.sh my_model
```

---

## 🔄 完整工作流程

### 1. 数据准备

```bash
# 如果有Roboflow导出的zip文件
python scripts/merge_datasets.py

# 或手动准备
python scripts/prepare_dataset.py \
    --input_json annotations.json \
    --images_dir images/ \
    --output_dir dataset \
    --train_ratio 0.7 \
    --valid_ratio 0.2 \
    --test_ratio 0.1
```

### 2. 训练模型

```bash
# 基础训练
python scripts/train.py \
    --dataset_dir dataset \
    --output_dir output \
    --epochs 100

# 恢复训练（如果中断）
python scripts/train.py \
    --resume output/checkpoint.pth \
    --epochs 100
```

### 3. 评估和测试

```bash
# 快速测试
python scripts/quick_test.py

# 评估模型性能
python scripts/evaluate_model.py \
    --checkpoint output/checkpoint_best_total.pth \
    --split all

# 测试单张图片（使用data目录下的测试图片）
python scripts/test_model.py \
    --image data/test/images/eating.jpg \
    --checkpoint output/checkpoint_best_total.pth

# 批量测试
python scripts/test_dataset.py \
    --dataset_dir dataset/test \
    --checkpoint output/checkpoint_best_total.pth
```

### 4. 保存和使用模型

```bash
# 保存微调模型
./scripts/save_finetuned_model.sh my_model_v1

# 使用保存的模型（使用data目录下的测试图片）
python scripts/test_model.py \
    --checkpoint checkpoints/finetuned/my_model_v1/best/checkpoint_best_ema.pth \
    --image data/test/images/eating.jpg
```

### 5. 视频分析（可选）

```bash
# 分析视频（使用data目录下的视频文件）
python scripts/analyze_video.py \
    --video data/test/videos/09.25.00-09.30.00[R][0@0][0].mp4 \
    --checkpoint checkpoints/finetuned/my_model_v1/best/checkpoint_best_ema.pth

# 生成时间统计表格
python scripts/generate_time_table.py \
    --input output/analysis_video_*/video_results.json
```

---

## 📝 常用命令

### 训练相关

```bash
# 查看训练帮助
python scripts/train.py --help

# 基础训练
python scripts/train.py

# 自定义参数训练
python scripts/train.py --epochs 50 --batch_size 16 --lr 2e-4

# 恢复训练
python scripts/train.py --resume output/checkpoint.pth
```

### 评估相关

```bash
# 评估测试集
python scripts/evaluate_model.py --split test

# 评估所有数据集
python scripts/evaluate_model.py --split all

# 使用自定义模型
python scripts/evaluate_model.py \
    --checkpoint checkpoints/finetuned/my_model/best/checkpoint_best_ema.pth \
    --threshold 0.5
```

### 测试相关

```bash
# 快速测试
python scripts/quick_test.py

# 测试单张图片（使用data目录下的图片）
python scripts/test_model.py --image data/test/images/eating.jpg

# 批量测试
python scripts/test_dataset.py --dataset_dir dataset/test

# 视频分析（使用data目录下的视频文件）
python scripts/analyze_video.py --video data/test/videos/09.25.00-09.30.00[R][0@0][0].mp4
```

### 模型管理

```bash
# 保存微调模型
./scripts/save_finetuned_model.sh <模型名称>

# 查看所有模型
ls -lh checkpoints/finetuned/*/best/

# 查看模型说明
cat checkpoints/finetuned/<模型名称>/README.md
```

---

## ❓ 常见问题

### Q1: 数据集格式要求是什么？

**A:** 数据集需要是COCO格式：
- JSON标注文件：`_annotations.coco.json`
- 图片文件：JPG或PNG格式
- 目录结构：`dataset/train/`, `dataset/valid/`, `dataset/test/`

使用 `scripts/prepare_dataset.py` 可以自动转换格式。

### Q2: 训练需要多长时间？

**A:** 取决于：
- 数据集大小
- GPU性能
- 训练轮数

示例：1000张图片，RTX 4080，100 epochs，约需2-4小时。

### Q3: 如何选择最佳模型？

**A:** 推荐使用：
- `checkpoint_best_ema.pth` - EMA模型，通常性能最稳定
- `checkpoint_best_regular.pth` - 常规最佳模型
- `checkpoint_best_total.pth` - 总最佳模型（文件最小）

### Q4: 训练中断了怎么办？

**A:** 使用 `--resume` 参数恢复训练：
```bash
python scripts/train.py --resume output/checkpoint.pth
```

### Q5: 如何调整检测阈值？

**A:** 使用 `--threshold` 参数：
```bash
python scripts/test_model.py --image data/test/images/eating.jpg --threshold 0.5
```

### Q6: 如何查看所有可用参数？

**A:** 使用 `--help` 参数：
```bash
python scripts/train.py --help
python scripts/evaluate_model.py --help
python scripts/test_model.py --help
```

### Q7: 数据集在哪里？

**A:** 
- 训练数据集：`dataset/` 目录
- 测试图片和视频：`data/test/images/` 和 `data/test/videos/` 目录
- 原始数据：`data/raw/` 目录
- 如果数据集丢失，参考 [RECOVER_DATASET.md](RECOVER_DATASET.md)

### Q8: 如何保存多个微调模型？

**A:** 使用保存脚本，每次训练后保存：
```bash
./scripts/save_finetuned_model.sh model_v1
./scripts/save_finetuned_model.sh model_v2
```

详细说明参考 [checkpoints/finetuned/README.md](checkpoints/finetuned/README.md)

---

## 📚 更多资源

- **详细脚本使用**: [docs/guides/SCRIPTS_USAGE.md](docs/guides/SCRIPTS_USAGE.md)
- **项目结构说明**: [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
- **数据集恢复**: [RECOVER_DATASET.md](RECOVER_DATASET.md)
- **微调模型管理**: [checkpoints/finetuned/README.md](checkpoints/finetuned/README.md)
- **data目录说明**: [data/README.md](data/README.md)
- **官方文档**: https://rfdetr.roboflow.com

---

## 🎯 快速检查清单

开始训练前，确保：

- [ ] RF-DETR已安装：`pip install rfdetr`
- [ ] 数据集已准备：`dataset/train/` 存在且包含图片和标注
- [ ] 测试数据已准备：`data/test/images/` 和 `data/test/videos/` 中有测试文件
- [ ] GPU可用（可选但推荐）：`torch.cuda.is_available() == True`
- [ ] 有足够的磁盘空间（至少10GB用于模型和输出）

---

## 💡 提示

1. **首次使用**：建议先用 `quick_test.py` 测试环境是否正常
2. **训练参数**：默认参数适合大多数情况，无需修改即可开始
3. **模型保存**：训练完成后记得使用 `save_finetuned_model.sh` 保存模型
4. **查看帮助**：任何脚本都可以用 `--help` 查看详细参数说明
5. **日志查看**：训练日志保存在 `output/log.txt`，TensorBoard日志在 `output/runs/`
6. **测试数据**：测试图片放在 `data/test/images/`，测试视频放在 `data/test/videos/`

---

**祝您使用愉快！如有问题，请查看详细文档或提交Issue。**
