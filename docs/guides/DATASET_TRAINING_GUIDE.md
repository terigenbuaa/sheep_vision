# RF-DETR 数据集准备和训练指南

## 📋 概述

本指南帮助您：
1. 将 Label Studio 导出的 COCO 格式数据整理成 RF-DETR 要求的格式
2. 开始训练 RF-DETR 模型

## 🚀 快速开始

### 方法一：使用快速开始脚本（推荐）

```bash
# 1. 准备数据集并开始训练
./quick_start.sh <COCO_JSON文件> <图片目录> [输出目录]

# 示例：
./quick_start.sh exported_annotations.json ~/.local/share/label-studio/media/upload/4 dataset
```

### 方法二：分步执行

#### 步骤 1: 准备数据集

```bash
python prepare_dataset.py \
    --input_json <Label Studio导出的JSON文件> \
    --images_dir <图片文件所在目录> \
    --output_dir dataset \
    --train_ratio 0.7 \
    --valid_ratio 0.2 \
    --test_ratio 0.1
```

**参数说明：**
- `--input_json`: Label Studio 导出的 COCO JSON 文件路径
- `--images_dir`: 包含所有图片文件的目录
- `--output_dir`: 输出数据集目录（默认: dataset）
- `--train_ratio`: 训练集比例（默认: 0.7）
- `--valid_ratio`: 验证集比例（默认: 0.2）
- `--test_ratio`: 测试集比例（默认: 0.1）
- `--no_copy`: 不复制图片，使用符号链接（节省空间）

**示例：**
```bash
python prepare_dataset.py \
    --input_json exported_annotations.json \
    --images_dir ~/.local/share/label-studio/media/upload/4 \
    --output_dir dataset
```

#### 步骤 2: 开始训练

```bash
python train.py
```

训练脚本会自动：
- ✅ 检查数据集结构
- ✅ 检查 GPU 可用性
- ✅ 初始化模型
- ✅ 开始训练并保存检查点

## 📁 数据集结构要求

RF-DETR 要求的数据集结构：

```
dataset/
├── train/
│   ├── _annotations.coco.json  # 必须命名为 _annotations.coco.json
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── valid/  # 注意：必须是 valid（不是 validation 或 val）
│   ├── _annotations.coco.json
│   ├── image1.jpg
│   └── ...
└── test/  # 可选
    ├── _annotations.coco.json
    └── ...
```

## 🔧 脚本功能说明

### prepare_dataset.py

**功能：**
- ✅ 修复 COCO JSON 中的图片路径（将服务器路径转换为相对路径）
- ✅ 自动划分数据集为 train/valid/test
- ✅ 复制图片文件到对应目录
- ✅ 验证数据集完整性

**处理的问题：**
- 修复 Label Studio 导出的路径不一致问题
- 处理 URL 编码的文件名
- 自动匹配图片文件

### train.py

**功能：**
- ✅ 检查数据集结构
- ✅ 检查 GPU 可用性
- ✅ 初始化 RF-DETR 模型
- ✅ 开始训练并监控

**训练配置：**
- 批次大小: 4 × 4 = 16（可根据 GPU 显存调整）
- 学习率: 1e-4
- 编码器学习率: 1.5e-4
- 训练轮数: 100
- 早停: 启用（10个epoch无改进则停止）
- TensorBoard: 启用

## ⚙️ 调整训练参数

编辑 `train.py` 文件中的配置：

```python
# 批次大小调整（根据 GPU 显存）
BATCH_SIZE = 4            # 如果显存不足，可以改为 2
GRAD_ACCUM_STEPS = 4      # 相应调整为 8（保持总批次大小=16）

# 训练轮数
EPOCHS = 100

# 学习率
LEARNING_RATE = 1e-4
LR_ENCODER = 1.5e-4
```

### GPU 显存建议

| GPU 类型 | batch_size | grad_accum_steps | 总批次大小 |
|----------|------------|------------------|------------|
| A100 (40GB+) | 16 | 1 | 16 |
| RTX 3090/4090 (24GB) | 8 | 2 | 16 |
| RTX 3080 (10GB) | 4 | 4 | 16 |
| T4 (16GB) | 4 | 4 | 16 |
| 较小显存 | 2 | 8 | 16 |

## 📊 监控训练

### TensorBoard

训练过程中，在另一个终端运行：

```bash
tensorboard --logdir output
```

然后在浏览器打开：`http://localhost:6006`

### 训练输出

训练完成后，在 `output` 目录下会生成：

- `checkpoint.pth` - 最新检查点
- `checkpoint_10.pth`, `checkpoint_20.pth` ... - 定期检查点
- `checkpoint_best_ema.pth` - 最佳 EMA 模型
- `checkpoint_best_regular.pth` - 最佳常规模型
- `checkpoint_best_total.pth` - **最终最佳模型（用于推理）**

## 🎯 使用训练好的模型

```python
from rfdetr import RFDETRBase

# 加载训练好的模型
model = RFDETRBase(pretrain_weights="output/checkpoint_best_total.pth")

# 进行推理
detections = model.predict("test_image.jpg", threshold=0.5)
```

## ⚠️ 常见问题

### 1. 数据集目录不存在

**错误：** `❌ 错误: 数据集目录 'dataset' 不存在！`

**解决：** 先运行 `prepare_dataset.py` 准备数据集

### 2. 验证集文件夹名错误

**错误：** `❌ 错误: 验证集目录 'dataset/valid' 不存在！`

**解决：** RF-DETR 要求验证集文件夹名为 `valid`（不是 `validation` 或 `val`）

### 3. 标注文件名错误

**错误：** `❌ 错误: 训练集标注文件不存在！`

**解决：** 确保文件名为 `_annotations.coco.json`（注意下划线和扩展名）

### 4. 图片路径问题

**问题：** Label Studio 导出的 JSON 中图片路径不一致

**解决：** `prepare_dataset.py` 会自动修复路径问题

### 5. GPU 显存不足

**解决：** 减小 `batch_size`，相应增加 `grad_accum_steps`

## 📝 完整示例

```bash
# 1. 准备数据集
python prepare_dataset.py \
    --input_json exported_annotations.json \
    --images_dir ~/.local/share/label-studio/media/upload/4 \
    --output_dir dataset

# 2. 检查数据集结构
ls -la dataset/train/
ls -la dataset/valid/

# 3. 开始训练
python train.py

# 4. 监控训练（另一个终端）
tensorboard --logdir output

# 5. 使用训练好的模型
python -c "
from rfdetr import RFDETRBase
model = RFDETRBase(pretrain_weights='output/checkpoint_best_total.pth')
detections = model.predict('test.jpg', threshold=0.5)
print(detections)
"
```

## 📊 数据集划分说明

### 三种数据集的作用

#### 1. **训练集 (Train Set)** - 70%
**作用**：用于训练模型，让模型学习数据中的模式

**特点**：
- ✅ 模型直接在这些数据上学习
- ✅ 权重会根据这些数据更新
- ✅ 数据量最大（通常70%）

#### 2. **验证集 (Valid/Validation Set)** - 20%
**作用**：在训练过程中评估模型性能，用于调参和选择最佳模型

**特点**：
- ✅ **训练过程中使用**：每个epoch结束后评估
- ✅ **不参与训练**：模型不会在这些数据上学习
- ✅ **用于选择最佳模型**：保存验证集上表现最好的模型
- ✅ **用于早停**：如果验证集性能不再提升，可以提前停止训练
- ✅ **用于调参**：调整超参数（学习率、batch size等）

**注意**：训练日志中的 `test_loss` 实际上是在**验证集**上评估的，因为RF-DETR在训练时使用 `dataset_file='roboflow'` 时，`test` 实际指向 `valid`。

#### 3. **测试集 (Test Set)** - 10%
**作用**：最终评估模型性能，模拟真实应用场景

**特点**：
- ✅ **训练完成后使用**：只在训练结束后评估一次
- ✅ **完全不参与训练**：模型从未见过这些数据
- ✅ **真实性能指标**：反映模型在未知数据上的表现
- ✅ **用于部署决策**：决定模型是否可以部署到生产环境

**重要**：
- ⚠️ **测试集应该只评估一次**，避免过拟合测试集
- ⚠️ **不要用测试集调参**，否则测试集就失去了意义

### 训练流程中的使用

**训练阶段**:
```
训练集 → 模型学习 → 更新权重
验证集 → 评估性能 → 选择最佳模型 → 早停判断
测试集 → ❌ 不使用
```

**评估阶段（训练完成后）**:
```
训练集 → ❌ 不使用（已用于训练）
验证集 → ❌ 不使用（已用于选择模型）
测试集 → ✅ 最终评估 → 报告真实性能
```

### 最终评估

训练完成后，在测试集上进行最终评估：

```bash
# 评估测试集（最终性能）
python evaluate_model.py test

# 评估所有数据集（对比）
python evaluate_model.py all
```

**预期结果**：
- 验证集性能：训练时已知的
- 测试集性能：应该与验证集接近
- 如果差异很大，说明可能过拟合

## 🔗 相关文档

- [RF-DETR 中文使用指南](docs/中文使用指南.md)
- [RF-DETR 训练文档](docs/learn/train/index.md)
- [RF-DETR 官方文档](https://rfdetr.roboflow.com)

