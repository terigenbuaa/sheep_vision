#!/usr/bin/env python
"""
快速测试脚本 - 测试训练好的模型
"""

import argparse
from pathlib import Path
from PIL import Image
import supervision as sv
import torch
from rfdetr import RFDETRBase

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="RF-DETR 快速测试脚本 - 快速测试训练好的模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认参数快速测试
  python quick_test.py

  # 指定数据集目录
  python quick_test.py --dataset_dir dataset

  # 使用自定义模型和阈值
  python quick_test.py --checkpoint checkpoints/finetuned/default_model/best/checkpoint_best_ema.pth --threshold 0.5

  # 指定输出文件
  python quick_test.py --output output/my_test_result.jpg
        """
    )
    
    parser.add_argument("--checkpoint", type=str, default="output/checkpoint_best_total.pth",
                       help="模型检查点路径（默认: output/checkpoint_best_total.pth）")
    parser.add_argument("--dataset_dir", type=str, default="dataset",
                       help="数据集目录（默认: dataset，会查找test或valid子目录。也可使用data/test）")
    parser.add_argument("--threshold", type=float, default=0.3,
                       help="置信度阈值（默认: 0.3）")
    parser.add_argument("--output", type=str, default=None,
                       help="输出图片路径（默认: output/quick_test_result.jpg）")
    parser.add_argument("--split", type=str, default=None,
                       choices=["train", "valid", "test"],
                       help="指定数据集分割（默认: 自动查找test或valid）")
    
    return parser.parse_args()

def quick_test():
    """快速测试模型"""
    args = parse_args()
    
    print("=" * 70)
    print("🧪 RF-DETR 快速测试")
    print("=" * 70)
    
    # 配置
    checkpoint_path = args.checkpoint
    dataset_dir = Path(args.dataset_dir)
    
    # 查找测试图片
    if args.split:
        test_dir = dataset_dir / args.split
    else:
        test_dir = dataset_dir / "test"
        if not test_dir.exists():
            test_dir = dataset_dir / "valid"
    
    if not test_dir.exists():
        print(f"❌ 错误: 测试目录不存在: {test_dir}")
        print(f"💡 提示: 请确保数据集目录存在，或使用 --split 指定分割")
        return
    
    # 查找PNG或JPG图片
    test_images = list(test_dir.glob("*.png")) + list(test_dir.glob("*.jpg"))
    
    if not test_images:
        print(f"❌ 错误: 在 {test_dir} 中未找到测试图片")
        return
    
    test_image = test_images[0]
    print(f"\n📸 使用测试图片: {test_image.name}")
    
    # 加载模型
    print(f"\n📦 加载模型: {checkpoint_path}")
    model = RFDETRBase(pretrain_weights=checkpoint_path)
    
    # 加载类别名称
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'args' in checkpoint and hasattr(checkpoint['args'], 'class_names'):
        class_names = checkpoint['args'].class_names
        model.model.class_names = class_names
        print(f"✅ 类别: {class_names}")
    else:
        class_names = model.class_names
        print(f"✅ 类别: {class_names}")
    
    # 推理
    threshold = args.threshold
    print(f"\n🔍 开始推理 (阈值={threshold})...")
    image = Image.open(test_image).convert('RGB')
    detections = model.predict(image, threshold=threshold)
    
    # 显示结果
    print(f"\n📊 检测结果:")
    print(f"   检测到 {len(detections)} 个目标")
    
    if len(detections) > 0:
        for i, (class_id, conf, bbox) in enumerate(zip(
            detections.class_id,
            detections.confidence,
            detections.xyxy
        )):
            # 映射类别ID
            mapped_id = class_id + 1 if isinstance(class_names, dict) and 0 not in class_names else class_id
            if isinstance(class_names, dict):
                class_name = class_names.get(mapped_id, f"class_{class_id}")
            else:
                class_name = f"class_{class_id}"
            
            print(f"   {i+1}. {class_name} (ID: {class_id}): {conf:.3f}")
            print(f"      位置: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
    else:
        print("   ⚠️  未检测到任何目标")
    
    # 可视化
    print(f"\n🎨 生成可视化结果...")
    annotated_image = image.copy()
    
    # 创建标签
    labels = []
    for class_id, conf in zip(detections.class_id, detections.confidence):
        mapped_id = class_id + 1 if isinstance(class_names, dict) and 0 not in class_names else class_id
        if isinstance(class_names, dict):
            class_name = class_names.get(mapped_id, f"class_{class_id}")
        else:
            class_name = f"class_{class_id}"
        labels.append(f"{class_name} {conf:.2f}")
    
    annotated_image = sv.BoxAnnotator(color=sv.ColorPalette.ROBOFLOW).annotate(
        annotated_image, detections
    )
    annotated_image = sv.LabelAnnotator(color=sv.ColorPalette.ROBOFLOW).annotate(
        annotated_image, detections, labels
    )
    
    # 保存结果
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("output/quick_test_result.jpg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated_image.save(output_path)
    
    print(f"💾 结果已保存到: {output_path}")
    
    print("\n" + "=" * 70)
    print("✅ 测试完成！")
    print("=" * 70)
    print("\n💡 下一步:")
    print(f"   1. 查看结果图片: {output_path}")
    print("   2. 完整评估: python scripts/evaluate_model.py")
    print("   3. 测试其他图片: python scripts/test_model.py --image <图片路径>")

if __name__ == "__main__":
    quick_test()

