#!/bin/bash
# 批量分析多个文件夹的脚本
# 使用方法: ./batch_analyze_folders.sh

echo "================================================================"
echo "📊 RF-DETR 批量分析脚本"
echo "================================================================"
echo ""
echo "将分析以下文件夹："
echo "  1. /mnt/f/24.12.15/extracted_frames/6-全景_frames (1440张)"
echo "  2. /mnt/f/24.12.15/extracted_frames/1-全景_frames (1448张)"
echo "  3. /mnt/f/24.12.15/extracted_frames/2-全景_frames (976张)"
echo ""
echo "每个文件夹会生成独立的分析结果文件夹"
echo ""

# 分析 6-全景_frames (1440张)
echo "开始分析: 6-全景_frames (1440张图片)..."
python test_dataset.py /mnt/f/24.12.15/extracted_frames/6-全景_frames

echo ""
echo "================================================================"
echo "✅ 分析完成: 6-全景_frames"
echo "结果保存在: output/analysis_24.12.15_6-全景_frames/"
echo "================================================================"
echo ""
echo "如需分析其他文件夹，请取消注释下面的命令："
echo ""
echo "# 分析 1-全景_frames (1448张)"
echo "# python test_dataset.py /mnt/f/24.12.15/extracted_frames/1-全景_frames"
echo ""
echo "# 分析 2-全景_frames (976张)"
echo "# python test_dataset.py /mnt/f/24.12.15/extracted_frames/2-全景_frames"

