# 数据文件目录说明

本目录用于存放测试和训练用的图片、视频等数据文件。

## 目录结构

```
data/
├── test/                    # 测试数据
│   ├── images/             # 测试图片
│   │   ├── eating.jpg
│   │   ├── not_eating.jpg
│   │   └── ...
│   └── videos/             # 测试视频
│       ├── 09.25.00-09.30.00[R][0@0][0].mp4
│       └── ...
│
├── train/                   # 训练数据（可选）
│   ├── images/             # 训练图片
│   └── videos/             # 训练视频
│
├── raw/                    # 原始数据（备份）
└── processed/              # 处理后的数据
```

## 使用说明

### 测试图片

测试图片放在 `data/test/images/` 目录下，可用于：
- 快速测试模型：`python scripts/quick_test.py`
- 单张图片测试：`python scripts/test_model.py --image data/test/images/eating.jpg`

### 测试视频

测试视频放在 `data/test/videos/` 目录下，可用于：
- 视频分析：`python scripts/analyze_video.py --video data/test/videos/video.mp4`

### 文件格式

- **图片格式**：支持 `.jpg`, `.jpeg`, `.png`, `.bmp`
- **视频格式**：支持 `.mp4`, `.avi`, `.mov`, `.mkv`

## 注意事项

- 大文件建议放在 `raw/` 目录，避免占用过多空间
- 处理后的数据可以放在 `processed/` 目录
- 训练数据集应使用 `dataset/` 目录（COCO格式）

## 快速使用

```bash
# 测试图片
python scripts/test_model.py --image data/test/images/eating.jpg

# 分析视频
python scripts/analyze_video.py --video data/test/videos/video.mp4

# 快速测试（会自动查找test目录下的图片）
python scripts/quick_test.py --dataset_dir data/test
```

