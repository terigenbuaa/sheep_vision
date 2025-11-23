# 训练问题排查指南

## 常见训练错误及解决方案

### 1. 数据集路径问题

**错误**: `FileNotFoundError` 或图片文件找不到

**检查**:
```bash
# 检查数据集结构
ls -la dataset/train/
ls -la dataset/valid/

# 检查JSON文件中的文件名是否与实际文件匹配
python -c "import json; data=json.load(open('dataset/train/_annotations.coco.json')); print(data['images'][0]['file_name'])"
```

**解决方案**:
- 确保所有图片文件都在 `dataset/train/` 和 `dataset/valid/` 目录中
- JSON文件中的 `file_name` 应该只是文件名（如 `image.jpg`），不是完整路径
- 如果文件名不匹配，重新运行 `merge_datasets.py`

### 2. GPU显存不足

**错误**: `CUDA out of memory`

**解决方案**:
- 减小 `batch_size`（从4改为2或1）
- 减小 `grad_accum_steps`（从4改为2）
- 减小 `resolution`（如果支持）
- 使用梯度检查点: `gradient_checkpointing=True`

### 3. 验证集问题

**错误**: 验证集相关错误

**解决方案**:
- 如果验证集有问题，可以设置 `eval=False` 先只训练
- 确保 `dataset/valid/` 目录存在且有 `_annotations.coco.json`
- 确保验证集有图片文件

### 4. 类别数量不匹配

**错误**: 类别数量相关错误

**检查**:
```python
import json
data = json.load(open('dataset/train/_annotations.coco.json'))
print(f"类别数: {len(data['categories'])}")
print(f"类别: {[c['name'] for c in data['categories']]}")
```

**解决方案**:
- 确保所有数据集的类别一致
- 检查合并后的数据集类别是否正确

### 5. 训练脚本参数问题

**检查训练脚本配置**:
```python
# train.py 中的配置
BATCH_SIZE = 4          # 如果显存不足，改为2或1
GRAD_ACCUM_STEPS = 4    # 如果显存不足，改为2
EPOCHS = 100            # 可以先改为10测试
```

## 快速诊断命令

```bash
# 1. 检查数据集
python diagnose_training.py

# 2. 检查JSON和文件匹配
python -c "
import json
from pathlib import Path
data = json.load(open('dataset/train/_annotations.coco.json'))
train_dir = Path('dataset/train')
missing = [img['file_name'] for img in data['images'][:10] 
           if not (train_dir / img['file_name']).exists()]
print(f'缺失文件: {len(missing)}')
"

# 3. 测试模型初始化
python -c "from rfdetr import RFDETRBase; model = RFDETRBase(); print('OK')"

# 4. 检查GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## 推荐的训练配置

### 小显存GPU (4-6GB)
```python
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 2
resolution = 512  # 如果支持
```

### 中等显存GPU (8-12GB)
```python
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
resolution = 640
```

### 大显存GPU (16GB+)
```python
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 2
resolution = 640
```

## 如果训练仍然失败

1. **查看完整错误信息**:
   ```bash
   python train.py 2>&1 | tee training_error.log
   ```

2. **检查日志文件**:
   ```bash
   tail -100 output/log.txt
   ```

3. **尝试最小配置测试**:
   ```python
   model.train(
       dataset_dir="dataset",
       epochs=1,  # 只训练1个epoch
       batch_size=1,  # 最小batch
       grad_accum_steps=1,
       eval=False,  # 禁用评估
       tensorboard=False,  # 禁用tensorboard
   )
   ```

4. **联系支持**: 如果问题仍然存在，请提供完整的错误信息和环境信息。

---

## 训练性能问题排查

### 6. Loss下降缓慢或停滞

**症状**:
- Loss在某个值附近波动，不再下降
- 训练后期Loss下降速度极慢（< 3% per epoch）

**常见原因及解决方案**:

#### 6.1 学习率调度器配置错误（最常见）

**问题**: `lr_drop` 设置不当，导致学习率永远不会下降

**检查**:
```python
# 检查训练脚本中的配置
lr_drop = 100  # 如果训练100个epoch，学习率永远不会下降！
```

**解决方案**:
```python
# 推荐配置（100个epoch）
lr_drop = 80           # 在第80个epoch下降学习率
warmup_epochs = 1.0    # 启用1个epoch的预热
```

**学习率变化**:
- Epoch 1: LR = 0 → 1e-4 (线性预热)
- Epoch 2-80: LR = 1e-4 (高学习率训练)
- Epoch 81-100: LR = 1e-5 (低学习率fine-tune)

**不同训练轮数的建议**:

| 总Epochs | lr_drop | warmup_epochs | 说明 |
|---------|---------|---------------|------|
| 50      | 40      | 1.0           | 前40个epoch高LR，后10个epoch低LR |
| 100     | 80      | 1.0           | 前80个epoch高LR，后20个epoch低LR |
| 150     | 120     | 1.0           | 前120个epoch高LR，后30个epoch低LR |

#### 6.2 损失权重不平衡

**问题**: `bbox_loss_coef` 过高，导致训练不稳定

**检查**:
```python
# 当前配置可能的问题
bbox_loss_coef = 5  # 可能偏高
```

**解决方案**:
```python
# 推荐配置
cls_loss_coef = 2      # 分类损失权重
bbox_loss_coef = 3     # 框回归损失权重（从5降低到3）
giou_loss_coef = 2     # GIoU损失权重

# Matcher成本权重（应与损失权重保持一致）
set_cost_class = 2
set_cost_bbox = 3      # 从5降低到3
set_cost_giou = 2
```

#### 6.3 Loss值偏高

**正常范围**:
- DETR模型在训练后期，总损失通常在 **3-5** 之间
- 如果损失 > 6，说明模型还未充分收敛

**检查各项损失**:
```python
# 训练日志中的损失
loss_ce: 0.7-0.8      # 分类损失（正常）
loss_bbox: 0.15-0.20  # 框回归损失（正常）
loss_giou: 0.4-0.6    # GIoU损失（正常）
```

**如果损失偏高**:
1. 检查学习率调度器配置（见6.1）
2. 调整损失权重（见6.2）
3. 增加训练轮数
4. 检查数据质量

#### 6.4 学习率过高或过低

**检查当前学习率**:
```python
# 训练日志中查看学习率
lr: 0.0001  # 1e-4
```

**调整建议**:
- 如果Loss下降缓慢且波动大：降低学习率到 `5e-5`
- 如果Loss完全不下降：检查学习率调度器（见6.1）
- 如果Loss下降太快：可能需要降低学习率

### 7. 使用优化训练脚本

已创建优化版本的训练脚本 `train_optimized.py`，包含所有优化配置：

```bash
python train_optimized.py
```

**优化内容**:
- ✅ 学习率调度器修复（lr_drop=80, warmup_epochs=1.0）
- ✅ 损失权重优化（bbox_loss_coef=3）
- ✅ Matcher成本权重优化（set_cost_bbox=3）
- ✅ 多尺度特征金字塔

### 8. 监控训练指标

**关键指标**:
- `loss`: 总损失，应该持续下降
- `loss_ce`: 分类损失，应该 < 0.6
- `loss_bbox`: 框回归损失，应该 < 0.2
- `loss_giou`: GIoU损失，应该 < 0.5
- `class_error`: 分类错误率，应该 < 5%

**使用TensorBoard监控**:
```bash
tensorboard --logdir output
```

**预期Loss变化**:
- Epoch 1-10: Loss快速下降（8 → 6）
- Epoch 11-50: Loss稳定下降（6 → 5）
- Epoch 51-80: Loss缓慢下降（5 → 4.5）
- Epoch 81-100: Loss进一步下降（4.5 → 4.0）

---

## 快速修复清单

如果遇到Loss下降缓慢的问题，按以下顺序检查：

1. ✅ **检查学习率调度器**: `lr_drop` 是否 < `epochs`？
2. ✅ **检查学习率预热**: `warmup_epochs` 是否 > 0？
3. ✅ **检查损失权重**: `bbox_loss_coef` 是否 <= 3？
4. ✅ **检查训练轮数**: 是否足够（建议100+ epochs）？
5. ✅ **使用优化脚本**: `python train_optimized.py`

