#!/usr/bin/env python
"""
单独测试 RF-DETR 检测功能
确保能够正确识别类别标签
"""
import cv2
import numpy as np
from pathlib import Path
import sys

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))

# 直接导入检测器模块，避免导入问题
from pathlib import Path
detector_path = Path(__file__).parent / "rfdetr_deepsort" / "detector" / "rfdetr_detector.py"
import importlib.util
spec = importlib.util.spec_from_file_location("rfdetr_detector", detector_path)
rfdetr_detector_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rfdetr_detector_module)
RFDETRDetector = rfdetr_detector_module.RFDETRDetector

import supervision as sv

def test_detection():
    """测试检测功能"""
    print("=" * 60)
    print("RF-DETR 检测功能测试")
    print("=" * 60)
    
    # 初始化检测器
    checkpoint_path = "output/checkpoint_best_ema.pth"
    confidence_threshold = 0.4
    
    print(f"\n📦 初始化检测器...")
    print(f"   检查点: {checkpoint_path}")
    print(f"   置信度阈值: {confidence_threshold}")
    
    detector = RFDETRDetector(
        checkpoint_path=checkpoint_path,
        confidence_threshold=confidence_threshold,
    )
    
    # 检查模型类别名称
    print(f"\n📋 检查模型类别信息...")
    if hasattr(detector.model, 'class_names'):
        class_names = detector.model.class_names
        print(f"   类别名称类型: {type(class_names)}")
        print(f"   类别名称: {class_names}")
        
        if isinstance(class_names, dict):
            print(f"   类别数量: {len(class_names)}")
            for class_id, class_name in class_names.items():
                print(f"     - ID {class_id}: {class_name}")
        elif isinstance(class_names, list):
            print(f"   类别数量: {len(class_names)}")
            for idx, class_name in enumerate(class_names):
                print(f"     - ID {idx}: {class_name}")
    else:
        print("   ⚠️  模型没有 class_names 属性")
    
    # 读取测试视频的第一帧
    video_path = "rfdetr_deepsort/11.25.00-11.30.00[R][0@0][0].mp4"
    print(f"\n🎥 读取测试视频: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"   ❌ 无法打开视频文件")
        return
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"   ❌ 无法读取视频帧")
        return
    
    print(f"   ✅ 成功读取帧，尺寸: {frame.shape}")
    
    # 转换为RGB（RF-DETR需要RGB格式）
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 执行检测
    print(f"\n🔍 执行检测...")
    detections = detector.detect(frame_rgb)
    
    print(f"\n📊 检测结果:")
    print(f"   检测数量: {len(detections)}")
    
    if len(detections) > 0:
        print(f"\n   详细信息:")
        print(f"   {'索引':<6} {'类别ID':<10} {'类别名称':<20} {'置信度':<10} {'边界框':<30}")
        print(f"   {'-'*6} {'-'*10} {'-'*20} {'-'*10} {'-'*30}")
        
        for i in range(min(len(detections), 20)):  # 最多显示20个
            class_id_raw = detections.class_id[i] if hasattr(detections, 'class_id') else None
            confidence = detections.confidence[i] if hasattr(detections, 'confidence') else None
            bbox = detections.xyxy[i] if hasattr(detections, 'xyxy') else None
            
            # 映射类别ID（与 analyze_video.py 一致）
            class_id = class_id_raw
            if class_id_raw is not None:
                if hasattr(detector.model, 'class_names'):
                    class_names = detector.model.class_names
                    if isinstance(class_names, dict):
                        # 如果字典中没有键0，则将class_id加1（0-based -> 1-based）
                        mapped_id = class_id_raw + 1 if 0 not in class_names else class_id_raw
                        class_id = mapped_id
            
            # 获取类别名称
            class_name = "Unknown"
            if class_id is not None:
                if hasattr(detector.model, 'class_names'):
                    class_names = detector.model.class_names
                    if isinstance(class_names, dict):
                        class_name = class_names.get(class_id, f"class_{class_id}")
                    elif isinstance(class_names, list) and class_id < len(class_names):
                        class_name = class_names[class_id]
                    else:
                        class_name = f"class_{class_id}"
            
            bbox_str = f"[{bbox[0]:.1f},{bbox[1]:.1f},{bbox[2]:.1f},{bbox[3]:.1f}]" if bbox is not None else "N/A"
            conf_str = f"{confidence:.3f}" if confidence is not None else "N/A"
            
            print(f"   {i:<6} {class_id_raw if class_id_raw is not None else 'N/A':<5} -> {class_id if class_id is not None else 'N/A':<5} {class_name:<20} {conf_str:<10} {bbox_str:<30}")
        
        # 统计各类别数量（使用映射后的ID）
        print(f"\n   类别统计（原始ID -> 映射ID）:")
        if hasattr(detections, 'class_id'):
            unique_classes, counts = np.unique(detections.class_id, return_counts=True)
            class_names = detector.model.class_names if hasattr(detector.model, 'class_names') else {}
            
            for class_id_raw, count in zip(unique_classes, counts):
                # 映射类别ID
                if isinstance(class_names, dict):
                    mapped_id = class_id_raw + 1 if 0 not in class_names else class_id_raw
                else:
                    mapped_id = class_id_raw
                
                # 获取类别名称
                class_name = "Unknown"
                if isinstance(class_names, dict):
                    class_name = class_names.get(mapped_id, f"class_{mapped_id}")
                elif isinstance(class_names, list) and mapped_id < len(class_names):
                    class_name = class_names[mapped_id]
                else:
                    class_name = f"class_{mapped_id}"
                
                print(f"     - 原始ID {class_id_raw} -> 映射ID {mapped_id} ({class_name}): {count} 个")
    else:
        print(f"   ⚠️  未检测到任何目标")
        print(f"   可能原因:")
        print(f"     1. 置信度阈值太高（当前: {confidence_threshold}）")
        print(f"     2. 图像中没有目标")
        print(f"     3. 模型未正确加载")
    
    # 保存带标注的图像
    if len(detections) > 0:
        print(f"\n💾 保存标注图像...")
        
        # 创建标注器
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        
        # 准备标签（使用映射后的类别ID）
        labels = []
        if hasattr(detections, 'class_id'):
            class_names = detector.model.class_names if hasattr(detector.model, 'class_names') else {}
            
            for i in range(len(detections)):
                class_id_raw = detections.class_id[i]
                confidence = detections.confidence[i]
                
                # 映射类别ID（与 analyze_video.py 一致）
                if isinstance(class_names, dict):
                    mapped_id = class_id_raw + 1 if 0 not in class_names else class_id_raw
                else:
                    mapped_id = class_id_raw
                
                # 获取类别名称
                class_name = "Unknown"
                if isinstance(class_names, dict):
                    class_name = class_names.get(mapped_id, f"class_{mapped_id}")
                elif isinstance(class_names, list) and mapped_id < len(class_names):
                    class_name = class_names[mapped_id]
                else:
                    class_name = f"class_{mapped_id}"
                
                labels.append(f"{class_name} {confidence:.2f}")
        
        # 绘制边界框和标签
        annotated_frame = box_annotator.annotate(
            scene=frame_rgb.copy(),
            detections=detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels
        )
        
        # 转换回BGR保存
        annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        output_path = "test_detection_result.jpg"
        cv2.imwrite(output_path, annotated_frame_bgr)
        print(f"   ✅ 已保存到: {output_path}")
    
    print(f"\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

if __name__ == "__main__":
    test_detection()

