# 批量分析执行指南

## 🚀 执行命令

### 分析所有8个日期目录（推荐）

```bash
python scripts/batch_analyze_all_dates.py \
    --base_dir /mnt/f \
    --checkpoint checkpoints/finetuned/default_model/best/checkpoint_best_ema.pth
```

### 简化版本（使用默认参数）

```bash
python scripts/batch_analyze_all_dates.py --base_dir /mnt/f
```

## 📋 执行前检查

1. **确认模型存在**：
   ```bash
   ls -lh checkpoints/finetuned/default_model/best/checkpoint_best_ema.pth
   ```

2. **预览计划**（不执行）：
   ```bash
   python scripts/batch_analyze_all_dates.py --base_dir /mnt/f --dry_run
   ```

3. **检查磁盘空间**（至少需要20GB）：
   ```bash
   df -h output/
   ```

## ⚙️ 可选参数

```bash
# 跳过已存在的报告（如果之前分析过部分目录）
python scripts/batch_analyze_all_dates.py \
    --base_dir /mnt/f \
    --skip_existing

# 只分析特定日期（例如：只分析2025年的）
python scripts/batch_analyze_all_dates.py \
    --base_dir /mnt/f \
    --date_pattern "2025.*"

# 调整置信度阈值
python scripts/batch_analyze_all_dates.py \
    --base_dir /mnt/f \
    --threshold 0.5

# 指定输出目录
python scripts/batch_analyze_all_dates.py \
    --base_dir /mnt/f \
    --output_dir output/my_reports
```

## 📊 将分析的内容

- **8个日期目录**
- **27个子目录**
- **38,188张图片**
- **预计时间**: 1-5小时
- **磁盘空间**: 约3.8-19 GB

## 📁 输出目录结构

```
output/
├── 2025.06.19/
│   ├── 8-东北全景_frames/
│   ├── 9-西北全景_frames/
│   └── date_summary.json
├── 2025.09.06/
│   ├── 15-全景（两水槽）_frames/
│   ├── 16-全景（两水槽）_frames/
│   ├── 17-全景（两水槽）_frames/
│   ├── 18-全景（两水槽）_frames/
│   └── date_summary.json
├── ... (其他日期目录)
└── all_dates_analysis_summary.json
```

## ⚠️ 注意事项

1. **确保GPU可用**（CPU会很慢）
2. **确保有足够磁盘空间**（至少20GB）
3. **可以随时中断**（使用Ctrl+C），之后用 `--skip_existing` 继续
4. **进度显示**：每10张图片显示一次进度

## ✅ 开始执行

复制并运行以下命令：

```bash
cd /home/teruun/projects/github/rf-detr
python scripts/batch_analyze_all_dates.py --base_dir /mnt/f
```

---

**详细计划**: 查看 [BATCH_ANALYSIS_PLAN.md](BATCH_ANALYSIS_PLAN.md)  
**目录结构**: 查看 [OUTPUT_DIRECTORY_STRUCTURE.md](OUTPUT_DIRECTORY_STRUCTURE.md)


