#!/usr/bin/env python
"""
调试跟踪过程中的类别变化
"""
import cv2
import numpy as np
from pathlib import Path
import sys

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))

# 直接导入检测器模块
detector_path = Path(__file__).parent / "rfdetr_deepsort" / "detector" / "rfdetr_detector.py"
import importlib.util
spec = importlib.util.spec_from_file_location("rfdetr_detector", detector_path)
rfdetr_detector_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rfdetr_detector_module)
RFDETRDetector = rfdetr_detector_module.RFDETRDetector

# 导入跟踪器主模块
main_path = Path(__file__).parent / "rfdetr_deepsort" / "main.py"
spec_main = importlib.util.spec_from_file_location("rfdetr_deepsort_main", main_path)
main_module = importlib.util.module_from_spec(spec_main)

# 临时修复导入问题
import sys
original_path = sys.path.copy()
sys.path.insert(0, str(Path(__file__).parent / "rfdetr_deepsort"))

spec_main.loader.exec_module(main_module)
RFDETRDeepSort = main_module.RFDETRDeepSort

sys.path = original_path

def debug_tracking_classes(video_path, checkpoint_path, threshold=0.4, num_frames=10):
    """调试跟踪过程中的类别变化"""
    print("=" * 80)
    print("🔍 跟踪类别变化调试")
    print("=" * 80)
    
    # 初始化跟踪器
    print(f"\n📦 初始化跟踪器...")
    tracker = RFDETRDeepSort(
        checkpoint_path=checkpoint_path,
        confidence_threshold=threshold,
    )
    
    # 获取类别名称
    class_names_dict = tracker.detector.model.class_names if hasattr(tracker.detector.model, 'class_names') else {}
    print(f"\n📋 类别信息: {class_names_dict}")
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return
    
    frame_count = 0
    
    # 记录每个轨迹的类别变化历史
    track_class_history = {}  # {track_id: [(frame, class_id, votes), ...]}
    
    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # 更新跟踪器
        tracks = tracker.update(frame)
        
        # 记录每个轨迹的类别信息
        for track in tracks:
            if track.is_confirmed():
                track_id = track.track_id
                if track_id not in track_class_history:
                    track_class_history[track_id] = []
                
                class_id = track.class_id
                votes = track.class_votes.copy() if hasattr(track, 'class_votes') else {}
                class_name = class_names_dict.get(class_id, f"class_{class_id}") if isinstance(class_names_dict, dict) else f"class_{class_id}"
                
                track_class_history[track_id].append({
                    'frame': frame_count,
                    'class_id': class_id,
                    'class_name': class_name,
                    'votes': dict(votes) if votes else {},
                    'hits': track.hits
                })
        
        # 详细输出前5帧
        if frame_count <= 5:
            print(f"\n{'='*80}")
            print(f"📸 第 {frame_count} 帧:")
            print(f"{'='*80}")
            
            confirmed_tracks = [t for t in tracks if t.is_confirmed()]
            print(f"   确认轨迹数: {len(confirmed_tracks)}")
            
            if len(confirmed_tracks) > 0:
                print(f"\n   轨迹详情:")
                for track in confirmed_tracks[:10]:  # 只显示前10个
                    class_id = track.class_id
                    class_name = class_names_dict.get(class_id, f"class_{class_id}") if isinstance(class_names_dict, dict) else f"class_{class_id}"
                    votes = track.class_votes if hasattr(track, 'class_votes') else {}
                    
                    print(f"     ID {track.track_id}: {class_name} (匹配次数: {track.hits})")
                    if votes:
                        print(f"       类别投票: {votes}")
                        # 找出最常见的类别
                        max_votes = max(votes.values()) if votes else 0
                        most_common = [k for k, v in votes.items() if v == max_votes]
                        print(f"       最常见类别: {most_common} (票数: {max_votes})")
    
    cap.release()
    
    # 分析类别变化
    print(f"\n{'='*80}")
    print(f"📊 轨迹类别变化分析:")
    print(f"{'='*80}")
    
    for track_id, history in sorted(track_class_history.items()):
        if len(history) < 2:
            continue
        
        print(f"\n   轨迹 ID {track_id}:")
        print(f"     总匹配次数: {history[-1]['hits']}")
        
        # 统计类别变化
        class_changes = []
        for i in range(len(history)):
            class_id = history[i]['class_id']
            class_name = history[i]['class_name']
            votes = history[i]['votes']
            
            if i == 0:
                print(f"     第 {history[i]['frame']} 帧: {class_name} (初始)")
            else:
                prev_class = history[i-1]['class_name']
                if class_name != prev_class:
                    print(f"     第 {history[i]['frame']} 帧: {prev_class} -> {class_name} (变化!)")
                    class_changes.append((history[i-1]['frame'], history[i]['frame'], prev_class, class_name))
                else:
                    print(f"     第 {history[i]['frame']} 帧: {class_name} (保持不变)")
            
            if votes:
                print(f"       投票详情: {votes}")
        
        if class_changes:
            print(f"     ⚠️  类别变化次数: {len(class_changes)}")
            for change in class_changes:
                print(f"       帧 {change[0]} -> {change[1]}: {change[2]} -> {change[3]}")
    
    print(f"\n{'='*80}")
    print("✅ 调试完成！")
    print("=" * 80)

if __name__ == "__main__":
    video_path = "rfdetr_deepsort/11.25.00-11.30.00[R][0@0][0].mp4"
    checkpoint_path = "output/checkpoint_best_ema.pth"
    threshold = 0.4
    
    debug_tracking_classes(video_path, checkpoint_path, threshold, num_frames=10)

