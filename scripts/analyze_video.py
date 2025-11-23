#!/usr/bin/env python
"""
视频分析脚本 - 对视频进行目标检测分析
"""

import json
import re
import argparse
from pathlib import Path
import cv2
import supervision as sv
import torch
from rfdetr import RFDETRBase
from collections import defaultdict
import numpy as np

def load_trained_model(checkpoint_path):
    """加载训练好的模型"""
    print(f"📦 加载模型检查点: {checkpoint_path}")
    
    model = RFDETRBase(pretrain_weights=checkpoint_path)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'args' in checkpoint and hasattr(checkpoint['args'], 'class_names'):
        model.model.class_names = checkpoint['args'].class_names
        model.model.args.class_names = checkpoint['args'].class_names
        print(f"✅ 加载类别名称: {checkpoint['args'].class_names}")
    
    model.model.model.eval()
    print("✅ 模型加载完成")
    return model

def extract_time_from_filename(filename):
    """从文件名中提取时间信息"""
    # 匹配时间格式 _HH-MM-SS
    pattern1 = r'_(\d{2})-(\d{2})-(\d{2})\.(jpg|png|mp4)'
    match = re.search(pattern1, filename)
    if match:
        hour, minute, second = match.groups()[:3]
        try:
            return f"{int(hour):02d}-{int(minute):02d}-{int(second):02d}"
        except:
            pass
    
    # 匹配其他时间格式
    patterns = [
        r'(\d{2})-(\d{2})-(\d{2})',
        r'(\d{2}):(\d{2}):(\d{2})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            hour, minute, second = match.groups()
            try:
                return f"{int(hour):02d}-{int(minute):02d}-{int(second):02d}"
            except:
                continue
    
    return None

def extract_folder_name_from_path(path):
    """从路径中提取文件夹名称作为分析文件夹名"""
    path_obj = Path(path)
    
    if path_obj.is_file():
        path_obj = path_obj.parent
    
    parts = path_obj.parts
    folder_name = path_obj.name
    
    # 向上查找包含日期格式的目录
    date_folder = None
    for part in reversed(parts):
        if re.match(r'^\d+\.\d+\.\d+$', part) or re.match(r'^\d+_\d+_\d+$', part):
            date_folder = part
            break
    
    if date_folder:
        folder_name = f"{date_folder}_{folder_name}"
    
    folder_name = re.sub(r'[^\w\.-]', '_', folder_name)
    return folder_name

def analyze_video(video_path, model, output_dir, threshold=0.5, frame_interval=30):
    """
    分析视频文件
    
    Args:
        video_path: 视频文件路径
        model: 训练好的模型
        output_dir: 输出目录
        threshold: 置信度阈值
        frame_interval: 每隔多少帧分析一次（默认30帧，约1秒）
    """
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"❌ 错误: 视频文件不存在: {video_path}")
        return None
    
    print(f"\n🎬 开始分析视频: {video_path.name}")
    print(f"   帧间隔: {frame_interval} 帧（约 {frame_interval/30:.1f} 秒）")
    
    # 打开视频
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ 错误: 无法打开视频文件")
        return None
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"   视频信息: {total_frames} 帧, {fps:.2f} FPS, {duration:.1f} 秒")
    
    # 创建输出目录
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # 分析结果
    results = []
    frame_count = 0
    analyzed_count = 0
    total_detections = 0
    
    class_names_dict = model.class_names
    
    print(f"\n🔍 开始逐帧分析...")
    print("=" * 70)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 每隔frame_interval帧分析一次
        if frame_count % frame_interval == 0:
            # 转换BGR到RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 进行推理
            detections = model.predict(frame_rgb, threshold=threshold)
            
            # 额外的过滤：过滤掉太小的边界框（可能是误检）
            if len(detections) > 0:
                valid_indices = []
                for i in range(len(detections)):
                    bbox = detections.xyxy[i]
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]
                    # 过滤掉宽度或高度小于30像素的检测（减少误检）
                    if w >= 30 and h >= 30:
                        valid_indices.append(i)
                
                if len(valid_indices) < len(detections):
                    # 只保留有效的检测
                    detections = detections[valid_indices]
            
            # 统计各类别数量
            class_counts = defaultdict(int)
            for class_id in detections.class_id:
                mapped_id = class_id + 1 if 0 not in class_names_dict else class_id
                class_name = class_names_dict.get(mapped_id, f"class_{class_id}")
                class_counts[class_name] += 1
            
            # 生成标签
            labels = []
            for idx, class_id in enumerate(detections.class_id):
                mapped_id = class_id + 1 if 0 not in class_names_dict else class_id
                class_name = class_names_dict.get(mapped_id, f"class_{class_id}")
                confidence = detections.confidence[idx]
                labels.append(f"{class_name} {confidence:.2f}")
            
            # 可视化
            annotated_frame = frame_rgb.copy()
            annotated_frame = sv.BoxAnnotator(color=sv.ColorPalette.ROBOFLOW).annotate(
                annotated_frame, detections
            )
            annotated_frame = sv.LabelAnnotator(color=sv.ColorPalette.ROBOFLOW).annotate(
                annotated_frame, detections, labels
            )
            
            # 转换回BGR保存
            annotated_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            
            # 保存标注后的帧
            frame_time = frame_count / fps if fps > 0 else frame_count
            frame_filename = f"frame_{frame_count:06d}_{frame_time:.1f}s.jpg"
            frame_path = images_dir / frame_filename
            cv2.imwrite(str(frame_path), annotated_bgr)
            
            # 记录结果
            result = {
                'frame_number': frame_count,
                'time_seconds': float(frame_time),
                'time_formatted': f"{int(frame_time//3600):02d}-{int((frame_time%3600)//60):02d}-{int(frame_time%60):02d}",
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
            results.append(result)
            total_detections += len(detections)
            analyzed_count += 1
            
            if analyzed_count % 10 == 0:
                print(f"   已分析: {analyzed_count} 帧 (总帧数: {frame_count}/{total_frames})")
        
        frame_count += 1
    
    cap.release()
    
    print("=" * 70)
    print(f"\n✅ 视频分析完成!")
    print(f"   总帧数: {total_frames}")
    print(f"   分析帧数: {analyzed_count}")
    print(f"   总检测数: {total_detections}")
    print(f"   平均每帧: {total_detections/analyzed_count:.2f} 个目标" if analyzed_count > 0 else "")
    
    return {
        'video_path': str(video_path),
        'video_name': video_path.name,
        'total_frames': total_frames,
        'analyzed_frames': analyzed_count,
        'fps': float(fps),
        'duration_seconds': float(duration),
        'frame_interval': frame_interval,
        'total_detections': total_detections,
        'results': results
    }

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="RF-DETR 视频分析脚本 - 对视频进行目标检测分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 分析视频（必需参数）
  python analyze_video.py --video data/video.mp4

  # 使用自定义模型和参数
  python analyze_video.py --video video.mp4 --checkpoint checkpoints/finetuned/default_model/best/checkpoint_best_ema.pth --threshold 0.5

  # 指定帧间隔和输出目录
  python analyze_video.py --video video.mp4 --frame_interval 60 --output_dir output/my_analysis
        """
    )
    
    parser.add_argument("--video", type=str, required=True,
                       help="视频文件路径（必需）")
    parser.add_argument("--checkpoint", type=str, default="output/checkpoint_best_total.pth",
                       help="模型检查点路径（默认: output/checkpoint_best_total.pth）")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="置信度阈值（默认: 0.5，建议范围: 0.5-0.7，值越高误检越少但可能漏检）")
    parser.add_argument("--frame_interval", type=int, default=30,
                       help="帧间隔，每隔N帧分析一次（默认: 30，约1秒）")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="输出目录（默认: output/analysis_video_<视频名称>）")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    checkpoint_path = args.checkpoint
    threshold = args.threshold
    frame_interval = args.frame_interval
    video_path = args.video
    
    # 确定输出目录
    if args.output_dir:
        base_output_dir = args.output_dir
    else:
        video_file = Path(video_path)
        video_name = video_file.stem
        folder_name = extract_folder_name_from_path(video_path)
        base_output_dir = f"output/analysis_video__{folder_name}"
    
    print("=" * 70)
    print("🎬 RF-DETR 视频分析")
    print("=" * 70)
    
    # 检查checkpoint是否存在
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        print(f"❌ 错误: 检查点文件不存在: {checkpoint_path}")
        return
    
    # 创建输出目录
    output_path = Path(base_output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 输出目录: {base_output_dir}")
    print(f"   - 标注帧图片: {output_path}/images/")
    print(f"   - JSON结果: {output_path}/video_results.json")
    print(f"   - CSV表格: {output_path}/time_statistics.csv")
    
    # 加载模型
    model = load_trained_model(checkpoint_path)
    
    # 分析视频
    video_data = analyze_video(video_path, model, output_path, threshold, frame_interval)
    
    if not video_data:
        return
    
    # 保存结果JSON
    json_path = output_path / "video_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(video_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 结果已保存:")
    print(f"   - 标注帧图片: {output_path}/images/")
    print(f"   - JSON结果: {json_path}")
    
    # 生成时间统计表格
    print(f"\n📊 生成时间统计表格...")
    try:
        from generate_time_table import generate_time_table
        
        # 转换视频结果为测试结果格式
        test_results = {
            'checkpoint': checkpoint_path,
            'dataset_dir': str(video_path),
            'threshold': threshold,
            'total_images': video_data['analyzed_frames'],
            'total_detections': video_data['total_detections'],
            'results': [
                {
                    'image_name': f"frame_{r['frame_number']:06d}",
                    'image_path': f"frame_{r['frame_number']:06d}",
                    'num_detections': r['num_detections'],
                    'class_counts': r['class_counts'],
                    'detections': r['detections']
                }
                for r in video_data['results']
            ]
        }
        
        # 保存临时JSON用于生成表格
        temp_json = output_path / "temp_test_results.json"
        with open(temp_json, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)
        
        csv_path = output_path / "time_statistics.csv"
        excel_path = output_path / "time_statistics.xlsx"
        generate_time_table(str(temp_json), str(csv_path), str(excel_path))
        
        # 删除临时文件
        temp_json.unlink()
        
        print(f"   - CSV表格: {csv_path}")
        print(f"   - Excel表格: {excel_path}")
    except Exception as e:
        print(f"   ⚠️  生成表格时出错: {e}")
    
    print("\n" + "=" * 70)
    print("✅ 视频分析完成！")
    print("=" * 70)

if __name__ == "__main__":
    main()

