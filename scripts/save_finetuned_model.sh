#!/bin/bash
# 保存微调模型脚本
# 用法: ./save_finetuned_model.sh <模型名称> [源目录]

set -e

MODEL_NAME=${1:-""}
SOURCE_DIR=${2:-"output"}

if [ -z "$MODEL_NAME" ]; then
    echo "❌ 错误: 请提供模型名称"
    echo ""
    echo "用法: $0 <模型名称> [源目录]"
    echo ""
    echo "示例:"
    echo "  $0 model_v2                    # 从output目录保存模型"
    echo "  $0 task_detection output       # 从output目录保存模型"
    echo "  $0 2025-11-22 /path/to/output  # 从指定目录保存模型"
    echo ""
    echo "模型名称建议:"
    echo "  - 按版本: model_v1, model_v2"
    echo "  - 按任务: detection_v1, segmentation_v1"
    echo "  - 按日期: 2025-11-22, 2025-11-22_v1"
    echo "  - 按数据集: coco_custom, custom_dataset_v1"
    exit 1
fi

# 检查源目录是否存在
if [ ! -d "$SOURCE_DIR" ]; then
    echo "❌ 错误: 源目录不存在: $SOURCE_DIR"
    exit 1
fi

# 检查必要的文件是否存在
if [ ! -f "$SOURCE_DIR/checkpoint_best_ema.pth" ]; then
    echo "⚠️  警告: 未找到 checkpoint_best_ema.pth"
    echo "   继续执行，但可能缺少一些文件..."
fi

MODEL_DIR="checkpoints/finetuned/$MODEL_NAME"
BEST_DIR="$MODEL_DIR/best"
CHECKPOINTS_DIR="$MODEL_DIR/checkpoints"

echo "📦 保存微调模型: $MODEL_NAME"
echo "   源目录: $SOURCE_DIR"
echo "   目标目录: $MODEL_DIR"
echo ""

# 创建目录结构
mkdir -p "$BEST_DIR"
mkdir -p "$CHECKPOINTS_DIR"

# 复制最佳模型文件
echo "📋 复制最佳模型文件..."
if [ -f "$SOURCE_DIR/checkpoint_best_ema.pth" ]; then
    cp "$SOURCE_DIR/checkpoint_best_ema.pth" "$BEST_DIR/" && echo "   ✅ checkpoint_best_ema.pth"
fi

if [ -f "$SOURCE_DIR/checkpoint_best_regular.pth" ]; then
    cp "$SOURCE_DIR/checkpoint_best_regular.pth" "$BEST_DIR/" && echo "   ✅ checkpoint_best_regular.pth"
fi

if [ -f "$SOURCE_DIR/checkpoint_best_total.pth" ]; then
    cp "$SOURCE_DIR/checkpoint_best_total.pth" "$BEST_DIR/" && echo "   ✅ checkpoint_best_total.pth"
fi

# 复制最新检查点
if [ -f "$SOURCE_DIR/checkpoint.pth" ]; then
    cp "$SOURCE_DIR/checkpoint.pth" "$CHECKPOINTS_DIR/latest.pth" && echo "   ✅ latest.pth"
fi

# 复制评估结果（如果存在）
if [ -f "$SOURCE_DIR/results.json" ]; then
    cp "$SOURCE_DIR/results.json" "$MODEL_DIR/results.json" && echo "   ✅ results.json"
fi

# 创建README模板
if [ ! -f "$MODEL_DIR/README.md" ]; then
    cat > "$MODEL_DIR/README.md" << EOF
# $MODEL_NAME 模型说明

## 模型信息
- **保存日期**: $(date +"%Y-%m-%d %H:%M:%S")
- **源目录**: $SOURCE_DIR
- **模型类型**: 微调模型

## 文件说明
- \`best/checkpoint_best_ema.pth\`: EMA最佳模型（推荐使用）
- \`best/checkpoint_best_regular.pth\`: 常规最佳模型
- \`best/checkpoint_best_total.pth\`: 总最佳模型
- \`checkpoints/latest.pth\`: 最新完整检查点

## 性能指标
（请根据实际评估结果填写）

### 验证集
- mAP@50: 
- Precision: 
- Recall: 

### 测试集
- mAP@50: 
- Precision: 
- Recall: 

## 训练信息
- **数据集**: （请填写）
- **训练轮数**: （请填写）
- **批次大小**: （请填写）
- **学习率**: （请填写）

## 使用说明

\`\`\`python
from rfdetr import RFDETRBase
import torch

checkpoint_path = "checkpoints/finetuned/$MODEL_NAME/best/checkpoint_best_ema.pth"
model = RFDETRBase()
checkpoint = torch.load(checkpoint_path, map_location='cpu')

if 'model' in checkpoint:
    model.load_state_dict(checkpoint['model'])
elif 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'])
else:
    model.load_state_dict(checkpoint)

model.eval()
\`\`\`

## 注意事项
- 模型文件较大，请确保有足够存储空间
- 建议定期备份重要模型
EOF
    echo "   ✅ 创建 README.md 模板"
fi

echo ""
echo "✅ 模型保存完成！"
echo ""
echo "📁 模型位置: $MODEL_DIR"
echo "📖 请编辑 $MODEL_DIR/README.md 填写详细的模型信息"
echo ""
echo "💡 提示: 可以使用以下命令查看模型:"
echo "   ls -lh $MODEL_DIR/best/"
echo "   cat $MODEL_DIR/README.md"

