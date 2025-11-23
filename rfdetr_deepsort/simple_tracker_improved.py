#!/usr/bin/env python
"""
改进的RF-DETR跟踪器 - 解决目标交叉时的ID交换问题
主要改进：
1. 使用卡尔曼滤波进行运动预测
2. 使用匈牙利算法进行最优匹配
3. 添加轨迹历史信息和速度计算
4. 多特征融合匹配（IoU + 运动 + 位置）
5. 轨迹一致性检查
"""
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import time
from collections import deque
from scipy.optimize import linear_sum_assignment

# 添加路径以导入rfdetr
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from rfdetr import RFDETRBase
import torch


class KalmanFilter:
    """简化的卡尔曼滤波器，用于预测目标位置"""
    
    def __init__(self):
        # 状态向量: [cx, cy, s, r, vx, vy, vs, vr]
        # cx, cy: 中心点坐标
        # s: 面积 (width * height)
        # r: 宽高比
        # vx, vy, vs, vr: 对应的速度
        self.ndim = 8
        self.dt = 1.0  # 时间步长
        
        # 状态转移矩阵 F
        self.F = np.eye(self.ndim, dtype=np.float32)
        self.F[0, 4] = self.dt  # cx += vx * dt
        self.F[1, 5] = self.dt  # cy += vy * dt
        self.F[2, 6] = self.dt  # s += vs * dt
        self.F[3, 7] = self.dt  # r += vr * dt
        
        # 观测矩阵 H (只观测位置和尺寸，不直接观测速度)
        self.H = np.zeros((4, self.ndim), dtype=np.float32)
        self.H[0, 0] = 1  # 观测 cx
        self.H[1, 1] = 1  # 观测 cy
        self.H[2, 2] = 1  # 观测 s
        self.H[3, 3] = 1  # 观测 r
        
        # 过程噪声协方差 Q
        self.Q = np.eye(self.ndim, dtype=np.float32) * 0.1
        
        # 观测噪声协方差 R
        self.R = np.eye(4, dtype=np.float32) * 1.0
        
        # 状态协方差 P
        self.P = np.eye(self.ndim, dtype=np.float32) * 1000.0
        
    def init(self, bbox):
        """初始化滤波器状态"""
        # bbox: [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        s = w * h
        r = w / (h + 1e-6)
        
        # 初始状态：位置已知，速度为0
        self.x = np.array([cx, cy, s, r, 0, 0, 0, 0], dtype=np.float32)
        self.P = np.eye(self.ndim, dtype=np.float32) * 1000.0
        
    def predict(self):
        """预测下一状态"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()
    
    def update(self, bbox):
        """使用观测更新状态"""
        # bbox: [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        s = w * h
        r = w / (h + 1e-6)
        
        z = np.array([cx, cy, s, r], dtype=np.float32)
        
        # 卡尔曼增益
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # 更新状态
        y = z - self.H @ self.x  # 残差
        self.x = self.x + K @ y
        self.P = (np.eye(self.ndim) - K @ self.H) @ self.P
        
    def get_bbox(self):
        """从状态向量获取边界框"""
        cx, cy, s, r = self.x[0], self.x[1], self.x[2], self.x[3]
        
        # 确保面积和宽高比为正数
        s = max(s, 1.0)  # 面积至少为1
        r = max(r, 0.1)  # 宽高比至少为0.1
        r = min(r, 10.0)  # 宽高比最多为10
        
        # 计算宽高
        w = np.sqrt(s * r)
        h = s / (w + 1e-6)
        
        # 确保宽高合理
        w = max(w, 1.0)
        h = max(h, 1.0)
        
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        
        bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
        
        # 确保边界框有效
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            # 如果无效，返回一个小的默认边界框
            bbox = np.array([cx - 10, cy - 10, cx + 10, cy + 10], dtype=np.float32)
        
        return bbox
    
    def get_velocity(self):
        """获取速度向量"""
        return self.x[4:8].copy()


class ImprovedTrack:
    """改进的轨迹类，包含运动预测和历史信息"""
    
    def __init__(self, track_id, bbox, class_id, frame_id):
        self.track_id = track_id
        self.class_id = class_id
        self.last_seen = frame_id
        self.hits = 1
        self.age = 0
        self.confirmed = False
        self.n_init = 3  # 需要3帧连续匹配才确认
        
        # 卡尔曼滤波器
        self.kf = KalmanFilter()
        self.kf.init(bbox)
        self.bbox = self.kf.get_bbox()
        
        # 历史位置（用于计算速度和方向）
        self.history = deque(maxlen=10)  # 保存最近10帧的位置
        self.history.append(self.bbox.copy())
        
        # 速度（像素/帧）
        self.velocity = np.zeros(2, dtype=np.float32)  # [vx, vy]
        
    def update(self, bbox, frame_id):
        """更新轨迹"""
        # 更新卡尔曼滤波器
        self.kf.update(bbox)
        self.bbox = self.kf.get_bbox()
        
        # 更新历史
        self.history.append(self.bbox.copy())
        
        # 计算速度（使用最近几帧的平均速度）
        if len(self.history) >= 2:
            recent_centers = []
            for h_bbox in list(self.history)[-5:]:  # 最近5帧
                cx = (h_bbox[0] + h_bbox[2]) / 2.0
                cy = (h_bbox[1] + h_bbox[3]) / 2.0
                # 确保坐标有效
                if not (np.isnan(cx) or np.isnan(cy) or np.isinf(cx) or np.isinf(cy)):
                    recent_centers.append([cx, cy])
            
            if len(recent_centers) >= 2:
                # 计算平均速度
                velocities = []
                for i in range(1, len(recent_centers)):
                    vx = recent_centers[i][0] - recent_centers[i-1][0]
                    vy = recent_centers[i][1] - recent_centers[i-1][1]
                    # 限制速度范围，避免异常值
                    vx = np.clip(vx, -100, 100)
                    vy = np.clip(vy, -100, 100)
                    velocities.append([vx, vy])
                
                if len(velocities) > 0:
                    self.velocity = np.mean(velocities, axis=0)
                    # 确保速度有效
                    self.velocity = np.nan_to_num(self.velocity, nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    self.velocity = np.zeros(2, dtype=np.float32)
            else:
                self.velocity = np.zeros(2, dtype=np.float32)
        else:
            self.velocity = np.zeros(2, dtype=np.float32)
        
        self.last_seen = frame_id
        self.hits += 1
        self.age = 0
        if self.hits >= self.n_init:
            self.confirmed = True
    
    def predict(self):
        """预测下一帧的位置"""
        self.kf.predict()
        predicted_bbox = self.kf.get_bbox()
        
        # 验证预测的边界框是否有效
        if np.all(np.isfinite(predicted_bbox)) and predicted_bbox[2] > predicted_bbox[0] and predicted_bbox[3] > predicted_bbox[1]:
            self.bbox = predicted_bbox
        # 如果无效，保持当前边界框
        
        self.age += 1
    
    def is_confirmed(self):
        return self.confirmed
    
    def is_deleted(self, max_age=30):
        """如果超过max_age帧未更新，标记为删除"""
        return self.age > max_age
    
    def get_center(self):
        """获取边界框中心点"""
        return np.array([
            (self.bbox[0] + self.bbox[2]) / 2.0,
            (self.bbox[1] + self.bbox[3]) / 2.0
        ], dtype=np.float32)
    
    def get_predicted_center(self):
        """获取预测的中心点（考虑速度）"""
        center = self.get_center()
        predicted = center + self.velocity
        
        # 确保预测值有效（不是 NaN 或 Inf）
        if np.any(np.isnan(predicted)) or np.any(np.isinf(predicted)):
            return center  # 如果无效，返回当前中心
        
        return predicted


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


def calculate_motion_cost(det_center, track_predicted_center, max_cost=100.0):
    """计算运动成本（欧氏距离）"""
    distance = np.linalg.norm(det_center - track_predicted_center)
    # 归一化到 [0, 1]，距离越大成本越高
    cost = min(distance / max_cost, 1.0)
    return cost


def associate_detections_to_tracks_improved(
    detections, 
    tracks, 
    iou_threshold=0.3,
    motion_weight=0.3,
    iou_weight=0.7,
    max_motion_distance=100.0
):
    """
    改进的检测-轨迹关联算法
    使用多特征融合：IoU + 运动预测
    
    Args:
        detections: 检测结果列表，每个元素是 (bbox, class_id, confidence)
        tracks: 轨迹列表
        iou_threshold: IoU阈值
        motion_weight: 运动成本权重
        iou_weight: IoU成本权重
        max_motion_distance: 最大运动距离（用于归一化）
    
    Returns:
        matches: 匹配对 [(det_idx, track_idx), ...]
        unmatched_dets: 未匹配的检测索引
        unmatched_trks: 未匹配的轨迹索引
    """
    if len(tracks) == 0:
        return [], list(range(len(detections))), []
    
    if len(detections) == 0:
        return [], [], list(range(len(tracks)))
    
    # 计算成本矩阵
    cost_matrix = np.ones((len(detections), len(tracks))) * 1e6
    
    for i, (det_bbox, _, _) in enumerate(detections):
        det_center = np.array([
            (det_bbox[0] + det_bbox[2]) / 2.0,
            (det_bbox[1] + det_bbox[3]) / 2.0
        ], dtype=np.float32)
        
        for j, track in enumerate(tracks):
            # IoU成本（1 - IoU，因为我们要最小化成本）
            iou_val = iou(det_bbox, track.bbox)
            iou_cost = 1.0 - iou_val
            
            # 运动成本
            track_predicted_center = track.get_predicted_center()
            motion_cost = calculate_motion_cost(det_center, track_predicted_center, max_motion_distance)
            
            # 综合成本
            total_cost = iou_weight * iou_cost + motion_weight * motion_cost
            
            # 如果IoU太低，设置高成本
            if iou_val < iou_threshold:
                total_cost = 1e6
            
            cost_matrix[i, j] = total_cost
    
    # 清理成本矩阵中的无效值（NaN, Inf）
    cost_matrix = np.nan_to_num(cost_matrix, nan=1e6, posinf=1e6, neginf=1e6)
    
    # 使用匈牙利算法进行最优匹配
    if cost_matrix.size > 0:
        # 确保成本矩阵是有限的
        if not np.all(np.isfinite(cost_matrix)):
            # 如果还有无效值，用大值替换
            cost_matrix[~np.isfinite(cost_matrix)] = 1e6
        
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        matches = []
        unmatched_dets = []
        unmatched_trks = []
        
        # 检查匹配的有效性
        used_dets = set()
        used_trks = set()
        
        for i, j in zip(row_indices, col_indices):
            if cost_matrix[i, j] < 1e5:  # 有效匹配
                matches.append((i, j))
                used_dets.add(i)
                used_trks.add(j)
        
        # 找出未匹配的检测和轨迹
        for i in range(len(detections)):
            if i not in used_dets:
                unmatched_dets.append(i)
        
        for j in range(len(tracks)):
            if j not in used_trks:
                unmatched_trks.append(j)
        
        return matches, unmatched_dets, unmatched_trks
    
    return [], list(range(len(detections))), list(range(len(tracks)))


class ImprovedTracker:
    """改进的跟踪器，解决ID交换问题"""
    
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
            if isinstance(class_names_raw, list):
                self.class_names = {i+1: name for i, name in enumerate(class_names_raw)}
            elif isinstance(class_names_raw, dict):
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
        
        # 关联检测到轨迹（使用改进的算法）
        matches, unmatched_dets, unmatched_trks = associate_detections_to_tracks_improved(
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
            new_track = ImprovedTrack(self.next_id, bbox, class_id, frame_id)
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
                output_path = str(input_path_obj.parent / f"{input_path_obj.stem}_improved_tracked{input_path_obj.suffix}")
            
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
                    
                    # 绘制速度向量（可选）
                    center = track.get_center()
                    if np.linalg.norm(track.velocity) > 0.5:  # 只显示有明显速度的
                        end_point = (int(center[0] + track.velocity[0] * 5), 
                                   int(center[1] + track.velocity[1] * 5))
                        cv2.arrowedLine(frame_drawn, 
                                     (int(center[0]), int(center[1])), 
                                     end_point, 
                                     color, 2, tipLength=0.3)
                    
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
                    cv2.imshow("Improved Tracker", frame_drawn)
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
    parser = argparse.ArgumentParser(description="改进的RF-DETR跟踪器 - 解决ID交换问题")
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
    
    # 处理视频路径
    input_path = args.input
    if not Path(input_path).exists():
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
    tracker = ImprovedTracker(
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

