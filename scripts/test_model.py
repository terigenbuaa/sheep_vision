#!/usr/bin/env python
"""
测试训练好的RF-DETR模型
使用指定的图片进行推理
"""

import json
import argparse
from pathlib import Path
from PIL import Image
import supervision as sv
import torch
from rfdetr import RFDETRBase

def load_trained_model(checkpoint_path):
    """加载训练好的模型"""
    print(f"📦 加载模型检查点: {checkpoint_path}")
    
    # 创建模型实例，使用训练好的权重
    model = RFDETRBase(pretrain_weights=checkpoint_path)
    
    # 加载checkpoint中的类别名称（如果有）
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'args' in checkpoint and hasattr(checkpoint['args'], 'class_names'):
        model.model.class_names = checkpoint['args'].class_names
        model.model.args.class_names = checkpoint['args'].class_names
        print(f"✅ 加载类别名称: {checkpoint['args'].class_names}")
    
    # 设置为评估模式
    model.model.model.eval()
    
    print("✅ 模型加载完成")
    return model

def get_ground_truth_annotations(image_path, dataset_dir):
    """获取图片的真实标注（用于对比）"""
    json_path = Path(dataset_dir) / "_annotations.coco.json"
    
    if not json_path.exists():
        return None
    
    with open(json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # 找到对应的图片ID
    image_name = image_path.name
    image_info = None
    for img in coco_data['images']:
        if img['file_name'] == image_name:
            image_info = img
            break
    
    if not image_info:
        return None
    
    # 获取该图片的所有标注
    annotations = []
    for ann in coco_data['annotations']:
        if ann['image_id'] == image_info['id']:
            annotations.append(ann)
    
    # 获取类别名称
    category_map = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    return {
        'image_info': image_info,
        'annotations': annotations,
        'categories': category_map
    }

def test_model(model, image_path, output_path, threshold=0.3, dataset_dir=None):
    """测试模型并进行可视化"""
    print(f"\n🔍 开始推理...")
    print(f"   图片路径: {image_path}")
    print(f"   置信度阈值: {threshold}")
    
    # 加载图片
    image = Image.open(image_path).convert('RGB')
    print(f"   图片尺寸: {image.size}")
    
    # 进行推理
    detections = model.predict(image, threshold=threshold)
    
    # 获取类别名称
    class_names_dict = model.class_names
    
    # 模型输出的class_id是从0开始的，但class_names字典是从1开始的
    # 需要调整映射
    labels = []
    for idx, class_id in enumerate(detections.class_id):
        # 如果class_names字典从1开始，需要+1
        # 如果从0开始，直接使用
        mapped_id = class_id + 1 if 0 not in class_names_dict else class_id
        class_name = class_names_dict.get(mapped_id, f"class_{class_id}")
        confidence = detections.confidence[idx]
        labels.append(f"{class_name} {confidence:.2f}")
    
    print(f"\n📊 检测结果:")
    print(f"   检测到 {len(detections)} 个目标")
    for i, (class_id, conf) in enumerate(zip(detections.class_id, detections.confidence)):
        mapped_id = class_id + 1 if 0 not in class_names_dict else class_id
        class_name = class_names_dict.get(mapped_id, f"class_{class_id}")
        print(f"   {i+1}. {class_name} (ID: {class_id}): {conf:.3f}")
    
    # 可视化
    annotated_image = image.copy()
    annotated_image = sv.BoxAnnotator(color=sv.ColorPalette.ROBOFLOW).annotate(
        annotated_image, detections
    )
    annotated_image = sv.LabelAnnotator(color=sv.ColorPalette.ROBOFLOW).annotate(
        annotated_image, detections, labels
    )
    
    # 保存结果
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated_image.save(output_path)
    print(f"\n💾 结果已保存到: {output_path}")
    
    # 显示真实标注（如果有）
    if dataset_dir:
        gt = get_ground_truth_annotations(Path(image_path), Path(dataset_dir))
        if gt and len(gt['annotations']) > 0:
            print(f"\n📋 真实标注:")
            print(f"   共有 {len(gt['annotations'])} 个标注")
            for ann in gt['annotations']:
                cat_name = gt['categories'].get(ann['category_id'], f"category_{ann['category_id']}")
                print(f"   - {cat_name}")
    
    return detections

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="RF-DETR 模型测试脚本 - 测试单张图片",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认图片测试
  python test_model.py

  # 测试指定图片
  python test_model.py --image data/eating.jpg

  # 使用自定义模型和阈值
  python test_model.py --checkpoint checkpoints/finetuned/default_model/best/checkpoint_best_ema.pth --threshold 0.5

  # 指定输出文件
  python test_model.py --image test.jpg --output result.jpg
        """
    )
    
    parser.add_argument("--image", type=str, default="data/test/images/eating.jpg",
                       help="输入图片路径（默认: data/test/images/eating.jpg）")
    parser.add_argument("--checkpoint", type=str, default="output/checkpoint_best_total.pth",
                       help="模型检查点路径（默认: output/checkpoint_best_total.pth）")
    parser.add_argument("--threshold", type=float, default=0.3,
                       help="置信度阈值（默认: 0.3）")
    parser.add_argument("--output", type=str, default=None,
                       help="输出图片路径（默认: output/test_result_<原文件名>.jpg）")
    parser.add_argument("--dataset_dir", type=str, default=None,
                       help="数据集目录（用于显示真实标注，可选）")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    checkpoint_path = args.checkpoint
    threshold = args.threshold
    image_path = args.image
    
    # 生成输出文件名
    image_file = Path(image_path)
    if args.output:
        output_image = args.output
    else:
        output_image = f"output/test_result_{image_file.stem}.jpg"
    
    print("=" * 70)
    print("🧪 RF-DETR 模型测试")
    print("=" * 70)
    
    # 检查checkpoint是否存在
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        print(f"❌ 错误: 检查点文件不存在: {checkpoint_path}")
        print(f"💡 提示: 请先完成训练，或指定正确的checkpoint路径")
        return
    
    # 检查图片是否存在
    if not image_file.exists():
        print(f"❌ 错误: 图片文件不存在: {image_path}")
        return
    
    print(f"📸 使用图片: {image_file.name}")
    
    # 加载模型
    model = load_trained_model(checkpoint_path)
    
    # 测试模型
    dataset_dir = args.dataset_dir if args.dataset_dir else None
    detections = test_model(model, image_path, output_image, threshold=threshold, dataset_dir=dataset_dir)
    
    print("\n" + "=" * 70)
    print("✅ 测试完成！")
    print("=" * 70)

if __name__ == "__main__":
    main()

