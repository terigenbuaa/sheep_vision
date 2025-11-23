#!/usr/bin/env python
"""
模型评估脚本 - 在测试集上进行完整评估
"""

import json
import argparse
from pathlib import Path
import torch
import numpy as np
from rfdetr import RFDETRBase
from rfdetr.datasets.coco_eval import CocoEvaluator
from rfdetr.engine import coco_extended_metrics
from pycocotools.coco import COCO

def evaluate_on_dataset(checkpoint_path, dataset_dir="dataset", split="test", threshold=0.3):
    """
    在指定数据集上进行评估
    
    Args:
        checkpoint_path: 模型检查点路径
        dataset_dir: 数据集目录
        split: 数据集分割 ('train', 'valid', 'test')
        threshold: 置信度阈值
    """
    print("=" * 70)
    print(f"📊 RF-DETR 模型评估 - {split.upper()}集")
    print("=" * 70)
    
    # 检查checkpoint
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        print(f"❌ 错误: 检查点文件不存在: {checkpoint_path}")
        return None
    
    # 检查数据集
    data_dir = Path(dataset_dir) / split
    if not data_dir.exists():
        print(f"❌ 错误: {split}集目录不存在: {data_dir}")
        return None
    
    data_json = data_dir / "_annotations.coco.json"
    if not data_json.exists():
        print(f"❌ 错误: {split}集标注文件不存在: {data_json}")
        return None
    
    print(f"\n📦 加载模型: {checkpoint_path}")
    model = RFDETRBase(pretrain_weights=checkpoint_path)
    
    # 加载类别名称
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'args' in checkpoint and hasattr(checkpoint['args'], 'class_names'):
        model.model.class_names = checkpoint['args'].class_names
        print(f"✅ 类别: {checkpoint['args'].class_names}")
    
    # 读取数据集信息
    with open(data_json, 'r') as f:
        data = json.load(f)
    
    images = data.get('images', [])
    annotations = data.get('annotations', [])
    categories = {c['id']: c['name'] for c in data.get('categories', [])}
    
    print(f"\n📸 {split.upper()}集信息:")
    print(f"   图片数量: {len(images)}")
    print(f"   标注数量: {len(annotations)}")
    print(f"   类别: {list(categories.values())}")
    
    # 初始化COCO评估器
    coco_gt = COCO(str(data_json))
    coco_evaluator = CocoEvaluator(coco_gt, iou_types=["bbox"])
    
    # 进行推理
    print(f"\n🔍 开始评估 (阈值={threshold})...")
    print("=" * 70)
    
    results = []
    total_detections = 0
    predictions = {}  # 用于COCO评估的预测结果
    
    for i, img_info in enumerate(images):
        img_path = data_dir / img_info['file_name']
        if not img_path.exists():
            print(f"⚠️  跳过: {img_info['file_name']} (文件不存在)")
            continue
        
        # 推理
        detections = model.predict(str(img_path), threshold=threshold)
        
        # 统计
        num_detections = len(detections)
        total_detections += num_detections
        
        # 获取真实标注
        gt_annotations = [ann for ann in annotations if ann['image_id'] == img_info['id']]
        
        # 准备COCO格式的预测结果
        if num_detections > 0:
            # COCO评估器期望xyxy格式，它会自动转换为xywh
            boxes_xyxy = torch.tensor(detections.xyxy, dtype=torch.float32)
            
            # 类别ID直接使用，因为数据集的类别ID就是0,1,2
            labels = torch.tensor(detections.class_id, dtype=torch.int64)
            scores = torch.tensor(detections.confidence, dtype=torch.float32)
            
            predictions[img_info['id']] = {
                'boxes': boxes_xyxy,  # xyxy格式，prepare_for_coco_detection会转换为xywh
                'labels': labels,      # 直接使用类别ID（0,1,2）
                'scores': scores
            }
        
        results.append({
            'image_id': img_info['id'],
            'image_name': img_info['file_name'],
            'num_detections': num_detections,
            'num_ground_truth': len(gt_annotations),
            'detections': [
                {
                    'class_id': int(class_id),
                    'confidence': float(conf),
                    'bbox': [float(x) for x in bbox]
                }
                for class_id, conf, bbox in zip(
                    detections.class_id,
                    detections.confidence,
                    detections.xyxy
                )
            ]
        })
        
        if (i + 1) % 10 == 0:
            print(f"   已处理: {i+1}/{len(images)} 张图片")
    
    # 更新COCO评估器
    if predictions:
        coco_evaluator.update(predictions)
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        
        # 获取扩展指标
        metrics = coco_extended_metrics(coco_evaluator.coco_eval["bbox"])
        
        # 从class_map中找到'all'类别的指标
        all_metrics = None
        for cm in metrics['class_map']:
            if cm['class'] == 'all':
                all_metrics = cm
                break
        
        print("=" * 70)
        print(f"\n📊 评估指标:")
        print("=" * 70)
        if all_metrics:
            print(f"\n🎯 整体指标:")
            print(f"   mAP@50:95: {all_metrics['map@50:95']*100:.2f}%")
            print(f"   mAP@50:    {all_metrics['map@50']*100:.2f}%")
            print(f"   Precision: {all_metrics['precision']*100:.2f}%")
            print(f"   Recall:    {all_metrics['recall']*100:.2f}%")
        else:
            print(f"\n🎯 整体指标:")
            print(f"   mAP@50:    {metrics['map']*100:.2f}%")
            print(f"   Precision: {metrics['precision']*100:.2f}%")
            print(f"   Recall:    {metrics['recall']*100:.2f}%")
        
        print(f"\n📋 各类别指标:")
        for class_metric in metrics['class_map']:
            if class_metric['class'] == 'all':
                continue
            print(f"\n   {class_metric['class']}:")
            print(f"      mAP@50:95: {class_metric['map@50:95']*100:.2f}%")
            print(f"      mAP@50:    {class_metric['map@50']*100:.2f}%")
            print(f"      Precision: {class_metric['precision']*100:.2f}%")
            print(f"      Recall:    {class_metric['recall']*100:.2f}%")
        
        # 保存指标到结果中
        if all_metrics:
            evaluation_metrics = {
                'mAP@50:95': float(all_metrics['map@50:95']),
                'mAP@50': float(all_metrics['map@50']),
                'precision': float(all_metrics['precision']),
                'recall': float(all_metrics['recall']),
                'per_class_metrics': [
                    {
                        'class': cm['class'],
                        'mAP@50:95': float(cm['map@50:95']),
                        'mAP@50': float(cm['map@50']),
                        'precision': float(cm['precision']),
                        'recall': float(cm['recall'])
                    }
                    for cm in metrics['class_map']
                ]
            }
        else:
            evaluation_metrics = {
                'mAP@50': float(metrics['map']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'per_class_metrics': [
                    {
                        'class': cm['class'],
                        'mAP@50:95': float(cm['map@50:95']),
                        'mAP@50': float(cm['map@50']),
                        'precision': float(cm['precision']),
                        'recall': float(cm['recall'])
                    }
                    for cm in metrics['class_map']
                ]
            }
    else:
        evaluation_metrics = None
        print("=" * 70)
        print(f"\n⚠️  警告: 没有检测到任何目标，无法计算评估指标")
    
    print("=" * 70)
    print(f"\n✅ 评估完成!")
    print(f"   总图片数: {len(results)}")
    print(f"   总检测数: {total_detections}")
    if len(results) > 0:
        print(f"   平均每张: {total_detections/len(results):.2f} 个目标")
    
    # 保存结果
    output_file = Path("output") / f"evaluation_results_{split}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'checkpoint': str(checkpoint_path),
        'dataset_split': split,
        'threshold': threshold,
        'total_images': len(results),
        'total_detections': total_detections,
        'results': results
    }
    
    if evaluation_metrics:
        output_data['metrics'] = evaluation_metrics
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 结果已保存到: {output_file}")
    
    return results, evaluation_metrics

def evaluate_all_splits(checkpoint_path, dataset_dir="dataset", threshold=0.3):
    """
    在所有数据集分割上进行评估（验证集和测试集）
    """
    print("=" * 70)
    print("📊 RF-DETR 完整评估（验证集 + 测试集）")
    print("=" * 70)
    
    results = {}
    metrics_dict = {}
    
    # 评估验证集
    print("\n" + "=" * 70)
    valid_results, valid_metrics = evaluate_on_dataset(checkpoint_path, dataset_dir, "valid", threshold)
    if valid_results:
        results['valid'] = valid_results
        if valid_metrics:
            metrics_dict['valid'] = valid_metrics
    
    # 评估测试集
    print("\n" + "=" * 70)
    test_results, test_metrics = evaluate_on_dataset(checkpoint_path, dataset_dir, "test", threshold)
    if test_results:
        results['test'] = test_results
        if test_metrics:
            metrics_dict['test'] = test_metrics
    
    # 总结
    print("\n" + "=" * 70)
    print("📊 评估总结")
    print("=" * 70)
    
    if 'valid' in results:
        valid_total = sum(r['num_detections'] for r in results['valid'])
        print(f"\n✅ 验证集:")
        print(f"   图片数: {len(results['valid'])}")
        print(f"   总检测数: {valid_total}")
        print(f"   平均每张: {valid_total/len(results['valid']):.2f} 个目标")
        if 'valid' in metrics_dict:
            m = metrics_dict['valid']
            print(f"   mAP@50:95: {m['mAP@50:95']*100:.2f}%")
            print(f"   mAP@50:    {m['mAP@50']*100:.2f}%")
            print(f"   Precision: {m['precision']*100:.2f}%")
            print(f"   Recall:    {m['recall']*100:.2f}%")
    
    if 'test' in results:
        test_total = sum(r['num_detections'] for r in results['test'])
        print(f"\n✅ 测试集:")
        print(f"   图片数: {len(results['test'])}")
        print(f"   总检测数: {test_total}")
        print(f"   平均每张: {test_total/len(results['test']):.2f} 个目标")
        if 'test' in metrics_dict:
            m = metrics_dict['test']
            print(f"   mAP@50:95: {m['mAP@50:95']*100:.2f}%")
            print(f"   mAP@50:    {m['mAP@50']*100:.2f}%")
            print(f"   Precision: {m['precision']*100:.2f}%")
            print(f"   Recall:    {m['recall']*100:.2f}%")
    
    return results

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="RF-DETR 模型评估脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 评估测试集（默认）
  python evaluate_model.py

  # 评估指定数据集
  python evaluate_model.py --split test
  python evaluate_model.py --split valid
  python evaluate_model.py --split all

  # 使用自定义模型和数据集
  python evaluate_model.py --checkpoint checkpoints/finetuned/default_model/best/checkpoint_best_ema.pth --dataset_dir dataset

  # 自定义置信度阈值
  python evaluate_model.py --threshold 0.5
        """
    )
    
    parser.add_argument("--checkpoint", type=str, default="output/checkpoint_best_total.pth",
                       help="模型检查点路径（默认: output/checkpoint_best_total.pth）")
    parser.add_argument("--dataset_dir", type=str, default="dataset",
                       help="数据集目录（默认: dataset）")
    parser.add_argument("--split", type=str, default="test",
                       choices=["train", "valid", "test", "all"],
                       help="数据集分割: train/valid/test/all（默认: test）")
    parser.add_argument("--threshold", type=float, default=0.3,
                       help="置信度阈值（默认: 0.3）")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    checkpoint_path = args.checkpoint
    dataset_dir = args.dataset_dir
    threshold = args.threshold
    split = args.split.lower()
    
    print("\n" + "=" * 70)
    print("📊 RF-DETR 模型评估脚本")
    print("=" * 70)
    print(f"   检查点: {checkpoint_path}")
    print(f"   数据集: {dataset_dir}")
    print(f"   分割: {split}")
    print(f"   阈值: {threshold}")
    print("=" * 70)
    
    if split == 'all':
        # 评估所有数据集
        results = evaluate_all_splits(checkpoint_path, dataset_dir, threshold)
    else:
        # 评估指定数据集
        results, metrics = evaluate_on_dataset(checkpoint_path, dataset_dir, split, threshold)
    
    if results:
        print("\n" + "=" * 70)
        print("✅ 评估完成！")
        print("=" * 70)
        print("\n💡 下一步:")
        print("   1. 查看详细结果: output/evaluation_results_*.json")
        print("   2. 测试单张图片: python test_model.py")
        print("   3. 快速测试: python quick_test.py")

if __name__ == "__main__":
    main()

