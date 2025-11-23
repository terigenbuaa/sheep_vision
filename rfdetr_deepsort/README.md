# Simple Tracker - RF-DETR 目标跟踪器

## 📋 项目概述

`simple_tracker.py` 是一个基于 RF-DETR 检测模型的简化目标跟踪系统。该系统专注于准确跟踪视频中的所有检测目标，采用 IoU（交并比）匹配算法，简化了特征匹配过程，实现了高效且稳定的多目标跟踪。

## 🎯 核心功能

1. **目标检测**：使用 RF-DETR 模型进行实时目标检测
2. **多目标跟踪**：基于 IoU 匹配算法跟踪多个目标
3. **轨迹管理**：自动创建、更新和删除轨迹
4. **视频处理**：支持视频输入、实时显示和结果保存
5. **可视化**：在视频帧上绘制跟踪框、ID 和类别标签

## 🏗️ 核心组件

### 1. SimpleTrack 类

轨迹类，用于表示单个目标的跟踪状态。

**属性：**
- `track_id`: 轨迹唯一标识符
- `bbox`: 边界框坐标 `[x1, y1, x2, y2]`
- `class_id`: 目标类别ID
- `last_seen`: 最后出现的帧ID
- `hits`: 匹配次数
- `age`: 未更新帧数
- `confirmed`: 是否已确认（需要连续2帧匹配）

**核心方法：**

```python
def update(self, bbox, frame_id):
    """更新轨迹位置，使用指数移动平均平滑边界框"""
    alpha = 0.7  # 平滑系数
    self.bbox = alpha * new_bbox + (1 - alpha) * self.bbox
```

```python
def predict(self):
    """预测下一帧位置（简单保持当前位置）"""
    self.age += 1  # 增加未更新计数
```

```python
def is_deleted(self, max_age=30):
    """判断轨迹是否应被删除（超过30帧未更新）"""
    return self.age > max_age
```

### 2. SimpleTracker 类

主跟踪器类，整合检测和跟踪功能。

**初始化流程：**
1. 加载 RF-DETR 模型检查点
2. 从检查点中提取类别名称
3. 设置模型为评估模式

**核心方法：**

#### `update(frame_rgb, frame_id)`

每帧调用的核心更新方法，执行以下步骤：

1. **目标检测**
   ```python
   detections_sv = self.model.predict(frame_rgb, threshold=self.confidence_threshold)
   ```

2. **数据转换**
   - 将检测结果转换为内部格式 `(bbox, class_id, confidence)`
   - 处理类别ID映射

3. **轨迹预测**
   ```python
   for track in self.tracks:
       track.predict()  # 增加age计数
   ```

4. **检测-轨迹关联**
   ```python
   matches, unmatched_dets, unmatched_trks = associate_detections_to_tracks(
       detections, self.tracks, self.iou_threshold
   )
   ```

5. **轨迹更新**
   - 更新匹配的轨迹
   - 为未匹配的检测创建新轨迹
   - 删除过期的轨迹

#### `process_video(input_path, output_path, show, save)`

视频处理主循环：

- 读取视频帧
- 逐帧调用 `update()` 进行跟踪
- 绘制跟踪结果（边界框、ID、类别）
- 显示/保存处理后的视频
- 实时显示 FPS 和进度信息

### 3. IoU 匹配算法

#### `iou(bbox1, bbox2)` 函数

计算两个边界框的交并比（Intersection over Union）：

```python
intersection = (x2 - x1) * (y2 - y1)
union = area1 + area2 - intersection
iou = intersection / union
```

#### `associate_detections_to_tracks()` 函数

使用贪心算法将检测结果关联到现有轨迹：

**算法流程：**
1. 计算所有检测-轨迹对的 IoU 矩阵
2. 筛选 IoU ≥ 阈值的候选对
3. 按 IoU 值从大到小排序
4. 贪心匹配：优先匹配 IoU 最大的对
5. 返回匹配对、未匹配检测和未匹配轨迹

**关键参数：**
- `iou_threshold=0.3`: IoU 匹配阈值，低于此值的检测-轨迹对不匹配

## 🔧 关键算法详解

### 1. 轨迹确认机制

新轨迹需要连续 **2 帧**匹配才能被确认（`n_init = 2`），避免误检导致的轨迹抖动。

```python
if self.hits >= self.n_init:
    self.confirmed = True
```

### 2. 边界框平滑

使用指数移动平均（EMA）平滑边界框，减少检测噪声：

```python
alpha = 0.7  # 新检测权重70%，历史权重30%
self.bbox = alpha * new_bbox + (1 - alpha) * self.bbox
```

### 3. 轨迹生命周期管理

- **创建**：未匹配的检测创建新轨迹
- **更新**：匹配的检测更新对应轨迹
- **删除**：超过 30 帧未更新的轨迹自动删除

### 4. 颜色生成

为每个轨迹ID生成唯一颜色，使用HSV色彩空间：

```python
hue = (track_id * 137.508) % 360  # 黄金角度确保颜色分布均匀
rgb = colorsys.hsv_to_rgb(hue / 360, 0.7, 0.9)
```

## 📖 使用方法

### 基本用法

```bash
python rfdetr_deepsort/simple_tracker.py \
    --input "rfdetr_deepsort/test.mp4" \
    --confidence 0.5 \
    --save
```

### 参数说明

| 参数 | 简写 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--input` | `-i` | str | **必需** | 输入视频路径 |
| `--output` | `-o` | str | None | 输出视频路径（默认自动生成） |
| `--checkpoint` | - | str | None | 模型检查点路径（默认自动查找） |
| `--confidence` | - | float | 0.5 | 检测置信度阈值 |
| `--iou-threshold` | - | float | 0.3 | IoU匹配阈值 |
| `--no-show` | - | flag | False | 不显示视频窗口 |
| `--save` | - | flag | False | 保存输出视频 |

### 模型检查点自动查找

如果未指定 `--checkpoint`，程序会按以下顺序自动查找：

1. `output/checkpoint_best_total.pth`
2. `output/checkpoint_best_ema.pth`
3. `output/checkpoint_best_regular.pth`

### 输出文件命名

如果未指定输出路径，会自动生成：
- 输入：`test.mp4`
- 输出：`test_simple_tracked.mp4`

## 📊 代码结构

```
simple_tracker.py
├── SimpleTrack 类          # 轨迹状态管理
│   ├── __init__()          # 初始化轨迹
│   ├── update()            # 更新轨迹位置
│   ├── predict()           # 预测下一帧
│   ├── is_confirmed()      # 检查是否确认
│   └── is_deleted()        # 检查是否应删除
│
├── SimpleTracker 类        # 主跟踪器
│   ├── __init__()          # 初始化模型和参数
│   ├── update()            # 核心跟踪更新逻辑
│   ├── process_video()     # 视频处理主循环
│   └── _generate_color()  # 生成轨迹颜色
│
├── iou()                   # IoU计算函数
├── associate_detections_to_tracks()  # 检测-轨迹关联
└── main()                  # 命令行入口
```

## 🔍 工作流程

```
视频输入
    ↓
逐帧读取
    ↓
RGB转换
    ↓
RF-DETR检测
    ↓
检测结果转换
    ↓
轨迹预测（age++）
    ↓
IoU匹配
    ↓
轨迹更新/创建/删除
    ↓
绘制结果
    ↓
显示/保存
```

## 💡 设计特点

1. **简化设计**：专注于 IoU 匹配，避免复杂的特征提取和匹配
2. **高效稳定**：贪心匹配算法时间复杂度低，适合实时处理
3. **鲁棒性**：轨迹确认机制和边界框平滑减少误检影响
4. **易用性**：自动查找模型检查点，支持相对路径

## 🎨 可视化特性

- **边界框**：每个轨迹使用唯一颜色绘制
- **标签**：显示 `ID: {track_id} {class_name}`
- **信息叠加**：实时显示 FPS、帧数、轨迹数量
- **进度提示**：每30帧输出处理进度

## 📝 注意事项

1. 确保 RF-DETR 模型检查点文件存在
2. 视频文件路径支持相对路径和绝对路径
3. 类别名称从检查点的 `args.class_names` 中读取
4. 建议在 GPU 环境下运行以获得更好的性能

## 🔗 依赖项

- `rfdetr`: RF-DETR 检测模型
- `torch`: PyTorch 深度学习框架
- `cv2`: OpenCV 图像处理
- `numpy`: 数值计算

## 🔄 改进版本

### simple_tracker_improved.py

针对**目标交叉时ID交换问题**的改进版本，主要改进包括：

1. ✅ **卡尔曼滤波运动预测** - 预测目标下一帧位置
2. ✅ **匈牙利算法** - 全局最优匹配，替代贪心算法
3. ✅ **多特征融合** - IoU + 运动信息综合匹配
4. ✅ **轨迹历史** - 保存历史位置和速度信息
5. ✅ **速度可视化** - 显示目标运动方向

**使用方法**：
```bash
python rfdetr_deepsort/simple_tracker_improved.py \
    --input "rfdetr_deepsort/test.mp4" \
    --confidence 0.5 \
    --save
```

详细改进说明请参考 [IMPROVEMENTS.md](IMPROVEMENTS.md)

---

**版本**: 1.0.0  
**改进版本**: 2.0.0 (simple_tracker_improved.py)  
**作者**: RF-DETR Team  
**许可证**: Apache License 2.0

