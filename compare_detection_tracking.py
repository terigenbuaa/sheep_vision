#!/usr/bin/env python
"""
对比检测和跟踪效果
分析为什么跟踪效果差
"""
import cv2
import numpy as np
import json
from pathlib import Path
import sys
from collections import defaultdict

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))

# 直接导入检测器模块
detector_path = Path(__file__).parent / "rfdetr_deepsort" / "detector" / "rfdetr_detector.py"
import importlib.util
spec = importlib.util.spec_from_file_location("rfdetr_detector", detector_path)
rfdetr_detector_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rfdetr_detector_module)
RFDETRDetector = rfdetr_detector_module.RFDETRDetector

import supervision as sv

def analyze_detection_only(video_path, checkpoint_path, threshold=0.4, num_frames=10):
    """只分析检测效果"""
    print("=" * 80)
    print("🔍 RF-DETR 检测效果分析")
    print("=" * 80)
    
    # 初始化检测器
    print(f"\n📦 初始化检测器...")
    detector = RFDETRDetector(
        checkpoint_path=checkpoint_path,
        confidence_threshold=threshold,
    )
    
    # 获取类别名称
    class_names_dict = detector.model.class_names if hasattr(detector.model, 'class_names') else {}
    print(f"\n📋 类别信息:")
    print(f"   类别字典: {class_names_dict}")
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n📊 视频信息:")
    print(f"   总帧数: {total_frames}")
    print(f"   FPS: {fps:.2f}")
    print(f"   分析前 {num_frames} 帧")
    
    # 统计信息
    stats = {
        'total_detections': 0,
        'class_distribution_raw': defaultdict(int),  # 原始ID
        'class_distribution_mapped': defaultdict(int),  # 映射后ID
        'bbox_sizes': [],
        'confidences': [],
        'frame_stats': []
    }
    
    frame_count = 0
    
    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 执行检测
        detections = detector.detect(frame_rgb)
        
        # 统计
        frame_detections = {
            'frame': frame_count,
            'count': len(detections),
            'raw_classes': [],
            'mapped_classes': [],
            'confidences': [],
            'bboxes': []
        }
        
        if len(detections) > 0:
            stats['total_detections'] += len(detections)
            
            for i in range(len(detections)):
                class_id_raw = detections.class_id[i]
                confidence = detections.confidence[i]
                bbox = detections.xyxy[i]
                
                # 统计原始类别
                stats['class_distribution_raw'][class_id_raw] += 1
                frame_detections['raw_classes'].append(int(class_id_raw))
                
                # 映射类别ID
                if isinstance(class_names_dict, dict):
                    mapped_id = class_id_raw + 1 if 0 not in class_names_dict else class_id_raw
                else:
                    mapped_id = class_id_raw
                
                class_name = class_names_dict.get(mapped_id, f"class_{mapped_id}") if isinstance(class_names_dict, dict) else f"class_{mapped_id}"
                stats['class_distribution_mapped'][class_name] += 1
                frame_detections['mapped_classes'].append(class_name)
                
                # 边界框大小
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                stats['bbox_sizes'].append((w, h))
                stats['confidences'].append(confidence)
                
                frame_detections['confidences'].append(float(confidence))
                frame_detections['bboxes'].append([float(x) for x in bbox.tolist()])
        
        stats['frame_stats'].append(frame_detections)
        
        # 详细输出前5帧
        if frame_count <= 5:
            print(f"\n{'='*80}")
            print(f"📸 第 {frame_count} 帧:")
            print(f"{'='*80}")
            print(f"   检测数量: {len(detections)}")
            
            if len(detections) > 0:
                print(f"\n   原始类别ID分布:")
                frame_raw_dist = defaultdict(int)
                for cid in frame_detections['raw_classes']:
                    frame_raw_dist[cid] += 1
                for cid, count in sorted(frame_raw_dist.items()):
                    print(f"     - 原始ID {cid}: {count} 个")
                
                print(f"\n   映射后类别分布:")
                frame_mapped_dist = defaultdict(int)
                for cname in frame_detections['mapped_classes']:
                    frame_mapped_dist[cname] += 1
                for cname, count in sorted(frame_mapped_dist.items()):
                    print(f"     - {cname}: {count} 个")
                
                print(f"\n   前10个检测详情:")
                for i in range(min(10, len(detections))):
                    print(f"     {i+1}. 原始ID: {frame_detections['raw_classes'][i]}, "
                          f"映射ID: {frame_detections['mapped_classes'][i]}, "
                          f"置信度: {frame_detections['confidences'][i]:.3f}, "
                          f"位置: {frame_detections['bboxes'][i]}")
    
    cap.release()
    
    # 总结统计
    print(f"\n{'='*80}")
    print(f"📊 总体统计（前 {num_frames} 帧）:")
    print(f"{'='*80}")
    
    print(f"\n🔍 检测统计:")
    print(f"   总检测数: {stats['total_detections']}")
    print(f"   平均每帧: {stats['total_detections']/num_frames:.2f} 个")
    
    print(f"\n   原始类别ID分布:")
    for class_id, count in sorted(stats['class_distribution_raw'].items()):
        print(f"     - 原始ID {class_id}: {count} 个 ({count/stats['total_detections']*100:.1f}%)")
    
    print(f"\n   映射后类别分布:")
    for class_name, count in sorted(stats['class_distribution_mapped'].items()):
        print(f"     - {class_name}: {count} 个 ({count/stats['total_detections']*100:.1f}%)")
    
    if stats['bbox_sizes']:
        avg_w = np.mean([s[0] for s in stats['bbox_sizes']])
        avg_h = np.mean([s[1] for s in stats['bbox_sizes']])
        min_w = np.min([s[0] for s in stats['bbox_sizes']])
        min_h = np.min([s[1] for s in stats['bbox_sizes']])
        max_w = np.max([s[0] for s in stats['bbox_sizes']])
        max_h = np.max([s[1] for s in stats['bbox_sizes']])
        print(f"\n   边界框大小统计:")
        print(f"     平均: {avg_w:.1f} x {avg_h:.1f}")
        print(f"     最小: {min_w:.1f} x {min_h:.1f}")
        print(f"     最大: {max_w:.1f} x {max_h:.1f}")
    
    if stats['confidences']:
        avg_conf = np.mean(stats['confidences'])
        min_conf = np.min(stats['confidences'])
        max_conf = np.max(stats['confidences'])
        print(f"\n   置信度统计:")
        print(f"     平均: {avg_conf:.3f}")
        print(f"     最小: {min_conf:.3f}")
        print(f"     最大: {max_conf:.3f}")
    
    # 保存详细结果
    output_file = Path("output/detection_analysis.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'video_path': video_path,
            'checkpoint_path': checkpoint_path,
            'threshold': threshold,
            'num_frames_analyzed': num_frames,
            'class_names_dict': dict(class_names_dict) if isinstance(class_names_dict, dict) else class_names_dict,
            'stats': {
                'total_detections': stats['total_detections'],
                'class_distribution_raw': {str(k): int(v) for k, v in stats['class_distribution_raw'].items()},
                'class_distribution_mapped': dict(stats['class_distribution_mapped']),
                'avg_bbox_size': [float(np.mean([s[0] for s in stats['bbox_sizes']])), 
                                 float(np.mean([s[1] for s in stats['bbox_sizes']]))] if stats['bbox_sizes'] else None,
                'avg_confidence': float(np.mean(stats['confidences'])) if stats['confidences'] else None,
            },
            'frame_stats': stats['frame_stats']
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 详细结果已保存到: {output_file}")
    
    print(f"\n{'='*80}")
    print("✅ 检测分析完成！")
    print("=" * 80)
    
    # 问题诊断
    print(f"\n🔍 问题诊断:")
    if stats['class_distribution_raw'].get(0, 0) > stats['total_detections'] * 0.5:
        print(f"   ⚠️  检测结果中超过50%是原始ID=0，这可能是背景类或类别映射问题")
    
    if stats['class_distribution_mapped'].get('normal', 0) > stats['total_detections'] * 0.8:
        print(f"   ⚠️  检测结果中超过80%是normal类别，可能存在类别不平衡问题")
    
    if stats['confidences']:
        low_conf_count = sum(1 for c in stats['confidences'] if c < 0.5)
        if low_conf_count > len(stats['confidences']) * 0.3:
            print(f"   ⚠️  超过30%的检测置信度低于0.5，可能需要调整阈值")

if __name__ == "__main__":
    video_path = "rfdetr_deepsort/11.25.00-11.30.00[R][0@0][0].mp4"
    checkpoint_path = "output/checkpoint_best_ema.pth"
    threshold = 0.4
    
    analyze_detection_only(video_path, checkpoint_path, threshold, num_frames=10)
