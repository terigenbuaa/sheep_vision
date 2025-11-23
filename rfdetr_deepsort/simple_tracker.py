#!/usr/bin/env python
"""
简化的RF-DETR跟踪器 - 专注于准确跟踪所有检测到的目标
主要依赖IoU匹配，简化特征匹配
"""
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict
import time
from collections import defaultdict

# 添加路径以导入rfdetr
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from rfdetr import RFDETRBase
import torch


class SimpleTrack:
    """简化的轨迹类"""
    def __init__(self, track_id, bbox, class_id, frame_id):
        self.track_id = track_id
        self.bbox = np.array(bbox, dtype=np.float32)  # [x1, y1, x2, y2]
        self.class_id = class_id
        self.last_seen = frame_id
        self.hits = 1
        self.age = 0
        self.confirmed = False
        self.n_init = 2  # 需要2帧连续匹配才确认
        
    def update(self, bbox, frame_id):
        """更新轨迹"""
        # 使用指数移动平均平滑边界框
        alpha = 0.7
        self.bbox = alpha * np.array(bbox, dtype=np.float32) + (1 - alpha) * self.bbox
        self.last_seen = frame_id
        self.hits += 1
        self.age = 0
        if self.hits >= self.n_init:
            self.confirmed = True
    
    def predict(self):
        """预测下一帧的位置（简单保持当前位置）"""
        self.age += 1
    
    def is_confirmed(self):
        return self.confirmed
    
    def is_deleted(self, max_age=30):
        """如果超过max_age帧未更新，标记为删除"""
        return self.age > max_age


def iou(bbox1, bbox2):
    """计算两个边界框的IoU"""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def associate_detections_to_tracks(detections, tracks, iou_threshold=0.3):
    """
    将检测结果关联到轨迹（使用IoU匹配）
    
    Args:
        detections: 检测结果列表，每个元素是 (bbox, class_id, confidence)
        tracks: 轨迹列表
        iou_threshold: IoU阈值
    
    Returns:
        matches: 匹配对 [(det_idx, track_idx), ...]
        unmatched_dets: 未匹配的检测索引
        unmatched_trks: 未匹配的轨迹索引
    """
    if len(tracks) == 0:
        return [], list(range(len(detections))), []
    
    if len(detections) == 0:
        return [], [], list(range(len(tracks)))
    
    # 计算IoU矩阵
    iou_matrix = np.zeros((len(detections), len(tracks)))
    for i, (det_bbox, _, _) in enumerate(detections):
        for j, track in enumerate(tracks):
            iou_matrix[i, j] = iou(det_bbox, track.bbox)
    
    # 贪心匹配：为每个检测找到IoU最大的轨迹
    matches = []
    unmatched_dets = list(range(len(detections)))
    unmatched_trks = list(range(len(tracks)))
    
    # 按IoU从大到小排序
    iou_pairs = []
    for i in range(len(detections)):
        for j in range(len(tracks)):
            if iou_matrix[i, j] >= iou_threshold:
                iou_pairs.append((iou_matrix[i, j], i, j))
    
    iou_pairs.sort(reverse=True)
    
    # 贪心匹配
    used_dets = set()
    used_trks = set()
    for iou_val, det_idx, trk_idx in iou_pairs:
        if det_idx not in used_dets and trk_idx not in used_trks:
            matches.append((det_idx, trk_idx))
            used_dets.add(det_idx)
            used_trks.add(trk_idx)
    
    unmatched_dets = [i for i in range(len(detections)) if i not in used_dets]
    unmatched_trks = [i for i in range(len(tracks)) if i not in used_trks]
    
    return matches, unmatched_dets, unmatched_trks


class SimpleTracker:
    """简化的跟踪器"""
    
    def __init__(self, checkpoint_path, confidence_threshold=0.5, iou_threshold=0.3):
        """
        Args:
            checkpoint_path: RF-DETR模型路径
            confidence_threshold: 检测置信度阈值
            iou_threshold: IoU匹配阈值
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_id = 1
        
        # 加载RF-DETR模型
        print(f"📦 加载RF-DETR模型: {checkpoint_path}")
        self.model = RFDETRBase(pretrain_weights=checkpoint_path)
        
        # 加载类别名称
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'args' in checkpoint and hasattr(checkpoint['args'], 'class_names'):
            class_names_raw = checkpoint['args'].class_names
            # 转换为字典格式：{1: 'normal', 2: 'offence_eating', 3: 'offence_not_eating'}
            if isinstance(class_names_raw, list):
                # 列表格式：['normal', 'offence_eating', 'offence_not_eating']
                # 转换为字典：{1: 'normal', 2: 'offence_eating', 3: 'offence_not_eating'}
                self.class_names = {i+1: name for i, name in enumerate(class_names_raw)}
            elif isinstance(class_names_raw, dict):
                # 已经是字典格式
                self.class_names = class_names_raw
            else:
                self.class_names = {}
            print(f"✅ 类别名称: {self.class_names}")
        else:
            self.class_names = {}
        
        self.model.model.model.eval()
        print("✅ 模型加载完成")
    
    def update(self, frame_rgb, frame_id):
        """
        更新跟踪器
        
        Args:
            frame_rgb: RGB格式的图像
            frame_id: 帧ID
        
        Returns:
            tracks: 当前帧的轨迹列表
        """
        # 执行检测
        detections_sv = self.model.predict(frame_rgb, threshold=self.confidence_threshold)
        
        # 转换为内部格式
        detections = []
        if len(detections_sv) > 0:
            for i in range(len(detections_sv)):
                bbox = detections_sv.xyxy[i]  # [x1, y1, x2, y2]
                confidence = detections_sv.confidence[i]
                class_id_raw = detections_sv.class_id[i]
                
                # 映射类别ID
                if isinstance(self.class_names, dict) and self.class_names:
                    if 0 not in self.class_names:
                        mapped_id = class_id_raw + 1
                    else:
                        mapped_id = class_id_raw
                else:
                    mapped_id = class_id_raw
                
                # 只保留在类别字典中的检测
                if isinstance(self.class_names, dict) and mapped_id not in self.class_names:
                    continue
                
                detections.append((bbox, mapped_id, confidence))
        
        # 预测所有轨迹
        for track in self.tracks:
            track.predict()
        
        # 关联检测到轨迹
        matches, unmatched_dets, unmatched_trks = associate_detections_to_tracks(
            detections, self.tracks, self.iou_threshold
        )
        
        # 更新匹配的轨迹
        for det_idx, trk_idx in matches:
            bbox, class_id, confidence = detections[det_idx]
            self.tracks[trk_idx].update(bbox, frame_id)
            # 更新类别ID（使用最新检测的类别）
            if class_id is not None:
                self.tracks[trk_idx].class_id = class_id
        
        # 为未匹配的检测创建新轨迹
        for det_idx in unmatched_dets:
            bbox, class_id, confidence = detections[det_idx]
            new_track = SimpleTrack(self.next_id, bbox, class_id, frame_id)
            self.tracks.append(new_track)
            self.next_id += 1
        
        # 删除过期的轨迹
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        
        return self.tracks
    
    def process_video(self, input_path, output_path=None, show=False, save=True):
        """处理视频"""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {input_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"视频信息: {width}x{height} @ {fps}fps, {total_frames} 帧")
        
        # 设置输出视频
        writer = None
        if save:
            if output_path is None:
                input_path_obj = Path(input_path)
                output_path = str(input_path_obj.parent / f"{input_path_obj.stem}_simple_tracked{input_path_obj.suffix}")
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"保存输出到: {output_path}")
        
        frame_count = 0
        fps_list = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                start_time = time.time()
                
                # 转换为RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 更新跟踪器
                tracks = self.update(frame_rgb, frame_count)
                
                # 绘制结果
                frame_drawn = frame.copy()
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    
                    x1, y1, x2, y2 = track.bbox.astype(int)
                    
                    # 裁剪到图像范围
                    x1 = max(0, min(x1, width - 1))
                    y1 = max(0, min(y1, height - 1))
                    x2 = max(0, min(x2, width - 1))
                    y2 = max(0, min(y2, height - 1))
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    # 生成颜色
                    color = self._generate_color(track.track_id)
                    
                    # 绘制边界框
                    cv2.rectangle(frame_drawn, (x1, y1), (x2, y2), color, 2)
                    
                    # 准备标签
                    if isinstance(self.class_names, dict) and track.class_id in self.class_names:
                        class_name = self.class_names[track.class_id]
                    else:
                        class_name = f"class_{track.class_id}"
                    label = f"ID:{track.track_id} {class_name}"
                    
                    # 绘制标签
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    cv2.rectangle(
                        frame_drawn,
                        (x1, y1 - text_height - baseline - 5),
                        (x1 + text_width, y1),
                        color,
                        -1
                    )
                    cv2.putText(
                        frame_drawn,
                        label,
                        (x1, y1 - baseline - 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )
                
                # 显示FPS
                elapsed_time = time.time() - start_time
                fps_current = 1.0 / elapsed_time if elapsed_time > 0 else 0
                fps_list.append(fps_current)
                
                cv2.putText(
                    frame_drawn,
                    f"FPS: {fps_current:.1f} | Frame: {frame_count}/{total_frames} | Tracks: {len([t for t in tracks if t.is_confirmed()])}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                
                if show:
                    cv2.imshow("Simple Tracker", frame_drawn)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                if writer:
                    writer.write(frame_drawn)
                
                # 进度信息
                if frame_count % 30 == 0:
                    avg_fps = np.mean(fps_list[-30:])
                    progress = frame_count / total_frames * 100
                    confirmed = len([t for t in tracks if t.is_confirmed()])
                    print(f"Progress: {progress:.1f}% | FPS: {avg_fps:.1f} | Tracks: {confirmed}")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if show:
                try:
                    cv2.destroyAllWindows()
                except:
                    pass
            
            if fps_list:
                print(f"\n平均FPS: {np.mean(fps_list):.2f}")
                print("处理完成！")
    
    def _generate_color(self, track_id):
        """根据track_id生成颜色"""
        import colorsys
        hue = (track_id * 137.508) % 360
        rgb = colorsys.hsv_to_rgb(hue / 360, 0.7, 0.9)
        bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        return bgr


def main():
    parser = argparse.ArgumentParser(description="简化的RF-DETR跟踪器")
    parser.add_argument("--input", "-i", type=str, required=True, help="输入视频路径")
    parser.add_argument("--output", "-o", type=str, default=None, help="输出视频路径")
    parser.add_argument("--checkpoint", type=str, default=None, help="模型检查点路径")
    parser.add_argument("--confidence", type=float, default=0.5, help="检测置信度阈值")
    parser.add_argument("--iou-threshold", type=float, default=0.3, help="IoU匹配阈值")
    parser.add_argument("--no-show", action="store_true", help="不显示视频")
    parser.add_argument("--save", action="store_true", help="保存输出视频")
    
    args = parser.parse_args()
    
    # 确定checkpoint路径
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        project_root = Path(__file__).parent.parent
        best_checkpoints = [
            project_root / "output" / "checkpoint_best_total.pth",
            project_root / "output" / "checkpoint_best_ema.pth",
            project_root / "output" / "checkpoint_best_regular.pth",
        ]
        
        for checkpoint in best_checkpoints:
            if checkpoint.exists():
                checkpoint_path = str(checkpoint)
                print(f"✅ 自动使用最佳模型: {checkpoint_path}")
                break
        
        if checkpoint_path is None:
            raise ValueError("未找到模型检查点，请使用 --checkpoint 指定")
    
    # 处理视频路径（支持方括号）
    input_path = args.input
    if not Path(input_path).exists():
        # 尝试在多个位置查找
        search_paths = [
            Path.cwd() / input_path,
            Path(__file__).parent / input_path,
            Path(__file__).parent.parent / input_path,
        ]
        for path in search_paths:
            if path.exists():
                input_path = str(path)
                break
        else:
            raise ValueError(f"视频文件不存在: {args.input}")
    
    # 创建跟踪器
    tracker = SimpleTracker(
        checkpoint_path=checkpoint_path,
        confidence_threshold=args.confidence,
        iou_threshold=args.iou_threshold
    )
    
    # 处理视频
    tracker.process_video(
        input_path=input_path,
        output_path=args.output,
        show=not args.no_show,
        save=args.save
    )


if __name__ == "__main__":
    main()

