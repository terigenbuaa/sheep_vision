#!/usr/bin/env python
"""
批量测试脚本 - 在测试集上进行测试并保存所有结果
"""

import json
import re
import argparse
from pathlib import Path
from PIL import Image
import supervision as sv
import torch
from rfdetr import RFDETRBase
from collections import defaultdict

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

def test_single_image(model, image_path, images_output_dir, threshold=0.3):
    """测试单张图片并保存结果"""
    # 加载图片
    image = Image.open(image_path).convert('RGB')
    
    # 进行推理
    detections = model.predict(image, threshold=threshold)
    
    # 获取类别名称
    class_names_dict = model.class_names
    
    # 生成标签
    labels = []
    for idx, class_id in enumerate(detections.class_id):
        mapped_id = class_id + 1 if 0 not in class_names_dict else class_id
        class_name = class_names_dict.get(mapped_id, f"class_{class_id}")
        confidence = detections.confidence[idx]
        labels.append(f"{class_name} {confidence:.2f}")
    
    # 可视化
    annotated_image = image.copy()
    annotated_image = sv.BoxAnnotator(color=sv.ColorPalette.ROBOFLOW).annotate(
        annotated_image, detections
    )
    annotated_image = sv.LabelAnnotator(color=sv.ColorPalette.ROBOFLOW).annotate(
        annotated_image, detections, labels
    )
    
    # 保存结果图片到images子文件夹
    image_name = Path(image_path).stem
    output_path = images_output_dir / f"annotated_{image_name}.jpg"
    annotated_image.save(output_path)
    
    # 获取真实标注（如果有）
    gt = get_ground_truth_annotations(image_path, Path(image_path).parent)
    
    # 统计信息
    class_counts = defaultdict(int)
    for class_id in detections.class_id:
        mapped_id = class_id + 1 if 0 not in class_names_dict else class_id
        class_name = class_names_dict.get(mapped_id, f"class_{class_id}")
        class_counts[class_name] += 1
    
    result = {
        'image_name': image_path.name,
        'image_path': str(image_path),
        'num_detections': len(detections),
        'class_counts': dict(class_counts),
        'detections': [
            {
                'class_id': int(class_id),
                'class_name': class_names_dict.get(
                    class_id + 1 if 0 not in class_names_dict else class_id,
                    f"class_{class_id}"
                ),
                'confidence': float(conf),
                'bbox': [float(x) for x in bbox]
            }
            for class_id, conf, bbox in zip(
                detections.class_id,
                detections.confidence,
                detections.xyxy
            )
        ]
    }
    
    # 添加真实标注信息（如果有）
    if gt:
        gt_class_counts = defaultdict(int)
        for ann in gt['annotations']:
            cat_name = gt['categories'].get(ann['category_id'], f"category_{ann['category_id']}")
            gt_class_counts[cat_name] += 1
        
        result['ground_truth'] = {
            'num_annotations': len(gt['annotations']),
            'class_counts': dict(gt_class_counts)
        }
    
    return result, output_path

def extract_folder_name_from_path(path):
    """
    从路径中提取文件夹名称作为分析文件夹名
    例如: /mnt/f/25.8.13/extracted_frames/12-全景_frames -> 25.8.13_12-全景_frames
          dataset/test -> test
    """
    path_obj = Path(path)
    
    # 如果路径是文件，获取父目录
    if path_obj.is_file():
        path_obj = path_obj.parent
    
    # 获取所有目录部分
    parts = path_obj.parts
    folder_name = path_obj.name
    
    # 向上查找包含日期格式的目录（如 25.8.13）
    date_folder = None
    for part in reversed(parts):
        # 检查是否是日期格式（数字.数字.数字）
        if re.match(r'^\d+\.\d+\.\d+$', part):
            date_folder = part
            break
    
    # 如果找到日期格式的文件夹，组合使用
    if date_folder:
        folder_name = f"{date_folder}_{folder_name}"
    
    # 清理文件夹名中的特殊字符，替换为下划线（保留点、横线和下划线）
    folder_name = re.sub(r'[^\w\.-]', '_', folder_name)
    
    return folder_name

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="RF-DETR 批量测试脚本 - 在数据集上进行批量测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 测试默认数据集
  python test_dataset.py

  # 测试指定数据集目录
  python test_dataset.py --dataset_dir dataset/test

  # 使用自定义模型和阈值
  python test_dataset.py --checkpoint checkpoints/finetuned/default_model/best/checkpoint_best_ema.pth --threshold 0.5

  # 指定输出目录
  python test_dataset.py --dataset_dir /mnt/f/data --output_dir output/my_analysis
        """
    )
    
    parser.add_argument("--dataset_dir", type=str, default="dataset/test",
                       help="数据集目录（默认: dataset/test）")
    parser.add_argument("--checkpoint", type=str, default="output/checkpoint_best_total.pth",
                       help="模型检查点路径（默认: output/checkpoint_best_total.pth）")
    parser.add_argument("--threshold", type=float, default=0.3,
                       help="置信度阈值（默认: 0.3）")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="输出目录（默认: output/analysis_<数据集名称>）")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    checkpoint_path = args.checkpoint
    dataset_dir = args.dataset_dir
    threshold = args.threshold
    
    # 确定输出目录
    if args.output_dir:
        base_output_dir = args.output_dir
    else:
        folder_name = extract_folder_name_from_path(dataset_dir)
        base_output_dir = f"output/analysis_{folder_name}"
    
    print("=" * 70)
    print("🧪 RF-DETR 批量测试 - 测试集")
    print("=" * 70)
    
    # 检查checkpoint是否存在
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        print(f"❌ 错误: 检查点文件不存在: {checkpoint_path}")
        return
    
    # 检查数据集目录是否存在
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        print(f"❌ 错误: 数据集目录不存在: {dataset_dir}")
        print(f"\n💡 提示:")
        print(f"   如果是Windows路径，在WSL中需要先挂载Windows驱动器")
        print(f"   例如: F:\\path\\to\\folder -> /mnt/f/path/to/folder")
        print(f"   如果F盘未挂载，可以运行: ./mount_f_drive.sh")
        print(f"   或者手动挂载: sudo mount -t drvfs F: /mnt/f")
        return
    
    # 创建输出目录结构
    output_path = Path(base_output_dir)
    images_dir = output_path / "images"
    output_path.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 输出目录: {base_output_dir}")
    print(f"   - 标注图片: {images_dir}/")
    print(f"   - JSON结果: {output_path}/test_results.json")
    print(f"   - CSV表格: {output_path}/time_statistics.csv")
    
    # 查找所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(dataset_path.glob(f"*{ext}")))
        image_files.extend(list(dataset_path.glob(f"*{ext.upper()}")))
    
    if not image_files:
        print(f"❌ 错误: 在 {dataset_dir} 中没有找到图片文件")
        return
    
    print(f"\n📸 找到 {len(image_files)} 张图片")
    print(f"📁 输出目录: {base_output_dir}")
    print(f"🎯 置信度阈值: {threshold}")
    print("\n" + "=" * 70)
    
    # 加载模型
    model = load_trained_model(checkpoint_path)
    
    # 批量处理
    print(f"\n🔍 开始批量测试...")
    print("=" * 70)
    
    results = []
    total_detections = 0
    total_gt_annotations = 0
    
    for i, image_path in enumerate(image_files, 1):
        try:
            result, saved_path = test_single_image(model, image_path, images_dir, threshold)
            results.append(result)
            total_detections += result['num_detections']
            if 'ground_truth' in result:
                total_gt_annotations += result['ground_truth']['num_annotations']
            
            if i % 10 == 0:
                print(f"   已处理: {i}/{len(image_files)} 张图片")
        except Exception as e:
            print(f"   ⚠️  处理 {image_path.name} 时出错: {e}")
            results.append({
                'image_name': image_path.name,
                'error': str(e)
            })
    
    print("=" * 70)
    print(f"\n✅ 批量测试完成!")
    print(f"   总图片数: {len(image_files)}")
    print(f"   总检测数: {total_detections}")
    print(f"   平均每张: {total_detections/len(image_files):.2f} 个目标")
    if total_gt_annotations > 0:
        print(f"   总真实标注数: {total_gt_annotations}")
        print(f"   检测/标注比: {total_detections/total_gt_annotations:.2f}")
    
    # 保存结果JSON
    summary = {
        'checkpoint': checkpoint_path,
        'dataset_dir': dataset_dir,
        'threshold': threshold,
        'total_images': len(image_files),
        'total_detections': total_detections,
        'total_ground_truth': total_gt_annotations if total_gt_annotations > 0 else None,
        'results': results
    }
    
    json_path = output_path / "test_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 生成时间统计表格
    print(f"\n📊 生成时间统计表格...")
    try:
        from generate_time_table import generate_time_table
        csv_path = output_path / "time_statistics.csv"
        excel_path = output_path / "time_statistics.xlsx"
        generate_time_table(str(json_path), str(csv_path), str(excel_path))
    except Exception as e:
        print(f"   ⚠️  生成表格时出错: {e}")
        csv_path = None
        excel_path = None
    
    print(f"\n💾 结果已保存:")
    print(f"   - 标注图片: {images_dir}/annotated_*.jpg")
    print(f"   - 结果JSON: {json_path}")
    if csv_path:
        print(f"   - CSV表格: {csv_path}")
    if excel_path:
        print(f"   - Excel表格: {excel_path}")
    
    print("\n" + "=" * 70)
    print("✅ 测试完成！")
    print("=" * 70)

if __name__ == "__main__":
    main()

