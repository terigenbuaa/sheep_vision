#!/bin/bash
# RF-DETR 训练启动脚本

echo "============================================================"
echo "🚀 RF-DETR 训练启动"
echo "============================================================"
echo ""

# 检查conda环境
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "⚠️  警告: 未激活conda环境"
    echo "💡 建议激活: conda activate rf-detr"
    echo ""
fi

# 检查数据集
echo "📊 检查数据集..."
if [ ! -d "dataset/train" ]; then
    echo "❌ 错误: dataset/train 目录不存在"
    exit 1
fi

if [ ! -f "dataset/train/_annotations.coco.json" ]; then
    echo "❌ 错误: dataset/train/_annotations.coco.json 不存在"
    exit 1
fi

echo "✅ 数据集检查通过"
echo ""

# 显示数据集统计
python3 << EOF
from pathlib import Path
import json

for split in ['train', 'valid']:
    json_path = Path(f'dataset/{split}/_annotations.coco.json')
    if json_path.exists():
        data = json.load(open(json_path))
        images = len(data['images'])
        annotations = len(data['annotations'])
        print(f"{split.upper()}: {images} 张图片, {annotations} 个标注")
EOF

echo ""
echo "============================================================"
echo "🚀 开始训练..."
echo "============================================================"
echo ""

# 运行训练脚本
python train.py

