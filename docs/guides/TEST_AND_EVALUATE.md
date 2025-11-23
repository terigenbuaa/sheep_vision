# 模型测试和评估指南

## 📊 训练结果总结

根据您的训练输出：
- **mAP@50:95**: 0.397 (39.7%)
- **mAP@50**: 0.796 (79.6%)
- **Precision**: 0.718 (71.8%)
- **Recall**: 0.77 (77%)
- **最佳模型**: `output/checkpoint_best_total.pth`

## 🧪 测试方法

### 方法1: 单张图片测试（快速）

```bash
# 修改test_model.py中的图片路径，然后运行
python test_model.py
```

**配置参数**（在`test_model.py`中）：
```python
checkpoint_path = "output/checkpoint_best_total.pth"
image_path = "dataset/test/your_image.jpg"  # 修改为实际图片路径
output_image = "output/test_result.jpg"
threshold = 0.3  # 置信度阈值
```

### 方法2: 批量评估（推荐）

```bash
# 在测试集上进行完整评估
python evaluate_model.py
```

这会：
- ✅ 在测试集/验证集上评估所有图片
- ✅ 统计检测结果
- ✅ 保存详细结果到 `output/evaluation_results.json`

### 方法3: 批量推理（可视化）

如果存在`batch_inference.py`：
```bash
python batch_inference.py --input_dir dataset/test --output_dir output/batch_results
```

## 📝 快速测试示例

### 1. 测试单张图片

```python
from rfdetr import RFDETRBase
from PIL import Image

# 加载模型
model = RFDETRBase(pretrain_weights='output/checkpoint_best_total.pth')

# 加载图片
image = Image.open('dataset/test/your_image.jpg')

# 推理
detections = model.predict(image, threshold=0.3)

# 查看结果
print(f"检测到 {len(detections)} 个目标")
for i, (class_id, conf) in enumerate(zip(detections.class_id, detections.confidence)):
    print(f"{i+1}. 类别 {class_id}: {conf:.3f}")
```

### 2. 查看模型类别

```python
from rfdetr import RFDETRBase
import torch

model = RFDETRBase(pretrain_weights='output/checkpoint_best_total.pth')

# 加载checkpoint获取类别名称
checkpoint = torch.load('output/checkpoint_best_total.pth', map_location='cpu', weights_only=False)
if 'args' in checkpoint and hasattr(checkpoint['args'], 'class_names'):
    print("类别名称:", checkpoint['args'].class_names)
else:
    print("类别名称:", model.class_names)
```

## 📊 评估指标说明

### mAP (Mean Average Precision)
- **mAP@50:95**: 在IoU阈值0.5到0.95（步长0.05）的平均精度
- **mAP@50**: 在IoU阈值0.5时的精度
- **您的结果**: mAP@50=79.6% 表现良好！

### Precision (精确率)
- 预测为正例中实际为正例的比例
- **您的结果**: 71.8% - 说明误检率较低

### Recall (召回率)
- 实际正例中被正确预测的比例
- **您的结果**: 77% - 说明漏检率较低

## 🎯 性能优化建议

### 如果mAP较低
1. **调整置信度阈值**：尝试不同的threshold值（0.2-0.5）
2. **数据增强**：增加训练数据或使用更强的数据增强
3. **训练更长时间**：使用优化配置重新训练

### 如果Precision较低（误检多）
- 提高置信度阈值（如0.4或0.5）
- 检查训练数据质量

### 如果Recall较低（漏检多）
- 降低置信度阈值（如0.2或0.25）
- 增加训练数据

## 🚀 快速开始

### 立即测试单张图片

1. **找到测试图片**：
   ```bash
   ls dataset/test/*.jpg | head -1
   ```

2. **修改test_model.py**：
   ```python
   image_path = "dataset/test/your_image.jpg"  # 替换为实际路径
   ```

3. **运行测试**：
   ```bash
   python test_model.py
   ```

4. **查看结果**：
   - 可视化结果：`output/test_result.jpg`
   - 控制台输出：检测到的目标列表

### 完整评估

```bash
python evaluate_model.py
```

结果保存在：`output/evaluation_results.json`

## 📁 输出文件说明

- `output/test_result.jpg` - 单张图片测试结果（带检测框）
- `output/evaluation_results.json` - 批量评估详细结果
- `output/checkpoint_best_total.pth` - 最佳模型（用于推理）

## 💡 下一步

1. ✅ **测试模型**：使用`test_model.py`测试单张图片
2. ✅ **批量评估**：使用`evaluate_model.py`评估测试集
3. ✅ **部署使用**：使用`checkpoint_best_total.pth`进行实际应用

