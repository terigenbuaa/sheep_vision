#!/bin/bash
# RF-DETR 快速开始脚本

set -e

echo "=========================================="
echo "RF-DETR 数据集准备和训练快速开始"
echo "=========================================="
echo ""

# 检查参数
if [ $# -lt 2 ]; then
    echo "用法: $0 <COCO_JSON文件> <图片目录> [数据集输出目录]"
    echo ""
    echo "示例:"
    echo "  $0 exported_annotations.json ~/.local/share/label-studio/media/upload/4 dataset"
    echo ""
    echo "参数说明:"
    echo "  COCO_JSON文件: Label Studio 导出的 COCO 格式 JSON 文件"
    echo "  图片目录: 包含所有图片文件的目录"
    echo "  数据集输出目录: 整理后的数据集保存位置（可选，默认: dataset）"
    exit 1
fi

JSON_FILE=$1
IMAGES_DIR=$2
OUTPUT_DIR=${3:-dataset}

echo "📋 配置信息:"
echo "   JSON 文件: $JSON_FILE"
echo "   图片目录: $IMAGES_DIR"
echo "   输出目录: $OUTPUT_DIR"
echo ""

# 检查文件是否存在
if [ ! -f "$JSON_FILE" ]; then
    echo "❌ 错误: JSON 文件不存在: $JSON_FILE"
    exit 1
fi

if [ ! -d "$IMAGES_DIR" ]; then
    echo "❌ 错误: 图片目录不存在: $IMAGES_DIR"
    exit 1
fi

# 步骤1: 准备数据集
echo "=========================================="
echo "步骤 1: 准备数据集"
echo "=========================================="
python prepare_dataset.py \
    --input_json "$JSON_FILE" \
    --images_dir "$IMAGES_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --train_ratio 0.7 \
    --valid_ratio 0.2 \
    --test_ratio 0.1

if [ $? -ne 0 ]; then
    echo "❌ 数据集准备失败"
    exit 1
fi

echo ""
echo "=========================================="
echo "步骤 2: 开始训练"
echo "=========================================="
echo ""
echo "💡 提示: 训练可能需要较长时间，可以使用 Ctrl+C 中断"
echo "   中断后可以从检查点恢复训练"
echo ""
read -p "按 Enter 键开始训练，或 Ctrl+C 取消..."

python train.py

echo ""
echo "=========================================="
echo "✅ 完成！"
echo "=========================================="

