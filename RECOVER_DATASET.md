# 数据集恢复指南

## 情况说明

`dataset` 目录已被删除，该目录通常包含：
- `train/` - 训练集（图片 + `_annotations.coco.json`）
- `valid/` - 验证集（图片 + `_annotations.coco.json`）
- `test/` - 测试集（图片 + `_annotations.coco.json`）
- `project-*.zip` - Roboflow导出的原始zip文件

## 恢复方法

### 方法1: 从Roboflow重新下载（推荐）

如果您使用的是Roboflow数据集：

1. 登录 [Roboflow](https://roboflow.com)
2. 找到您的项目
3. 导出数据集为COCO格式
4. 下载zip文件到项目根目录
5. 运行以下命令解压并整理：

```bash
# 如果只有一个zip文件
python scripts/prepare_dataset.py --input_json <解压后的json文件> --images_dir <图片目录> --output_dir dataset

# 如果有多个zip文件（project-*.zip格式）
python scripts/merge_datasets.py
```

### 方法2: 从备份位置恢复

检查以下可能的位置：

1. **F盘备份**：
   ```bash
   ls -la /mnt/f/24.12.15/extracted_frames/
   ```

2. **Windows回收站**：
   - 在Windows文件管理器中检查回收站
   - 路径：`F:\$RECYCLE.BIN\`

3. **其他备份位置**：
   ```bash
   find ~ -name "*dataset*" -type d 2>/dev/null
   find /mnt/f -name "*project*.zip" 2>/dev/null
   ```

### 方法3: 从训练输出重建（如果之前训练过）

如果您之前训练过模型，可以从训练日志或输出中获取数据集信息：

1. 检查 `output/` 目录中的日志文件
2. 查看训练脚本中的数据集路径配置

### 方法4: 重新准备数据集

如果您有原始数据：

```bash
# 使用prepare_dataset.py脚本
python scripts/prepare_dataset.py \
    --input_json <您的标注json文件> \
    --images_dir <图片目录> \
    --output_dir dataset \
    --train_ratio 0.7 \
    --valid_ratio 0.2 \
    --test_ratio 0.1
```

## 数据集目录结构

恢复后的 `dataset` 目录应该如下：

```
dataset/
├── train/
│   ├── _annotations.coco.json
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── valid/
│   ├── _annotations.coco.json
│   ├── image1.jpg
│   └── ...
└── test/
    ├── _annotations.coco.json
    ├── image1.jpg
    └── ...
```

## 验证数据集

恢复后，使用以下命令验证数据集：

```bash
python scripts/test_dataset.py dataset/train
python scripts/test_dataset.py dataset/valid
python scripts/test_dataset.py dataset/test
```

## 预防措施

为避免再次丢失数据：

1. **定期备份**：
   ```bash
   # 备份dataset目录
   tar -czf dataset_backup_$(date +%Y%m%d).tar.gz dataset/
   ```

2. **使用git LFS**（如果数据集不大）：
   ```bash
   git lfs track "dataset/**/*.json"
   git add .gitattributes
   ```

3. **添加到.gitignore但保留备份**：
   - dataset目录已在.gitignore中
   - 建议在外部位置保留备份

## 需要帮助？

如果您需要帮助恢复数据集，请提供：
1. 数据集的来源（Roboflow项目链接、本地路径等）
2. 数据集格式（COCO、YOLO等）
3. 是否有备份位置

