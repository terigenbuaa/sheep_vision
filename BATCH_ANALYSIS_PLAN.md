# 批量分析执行计划 - 所有日期目录

## 📋 分析概览

**基础目录**: `/mnt/f`  
**模型**: `checkpoints/finetuned/default_model/best/checkpoint_best_ema.pth` ✅ **微调后的模型**  
**模型类型**: EMA最佳模型（推荐使用，性能最稳定）  
**置信度阈值**: `0.3`  
**输出目录**: `output/`

> ✅ **确认**: 脚本默认使用微调后的模型，路径为 `checkpoints/finetuned/default_model/best/checkpoint_best_ema.pth`

## 📅 将分析的日期目录

找到 **8** 个日期目录，共 **27** 个子目录，**38,188** 张图片：

| 序号 | 日期目录 | 子目录数 | 图片数 | 子目录列表 |
|------|---------|---------|--------|-----------|
| 1 | `2025.06.19` | 2 | 2,880 | 8-东北全景_frames, 9-西北全景_frames |
| 2 | `2025.09.06` | 4 | 5,760 | 15-全景（两水槽）_frames × 4 |
| 3 | `2025.09.07` | 3 | 4,320 | 15-全景（两水槽）_frames × 3 |
| 4 | `2025.09.08` | 4 | 5,510 | 15-全景（两水槽）_frames × 4 |
| 5 | `24.12.15` | 5 | 6,744 | 1-全景_frames, 2-全景_frames, 3-全景_frames, 4-全景_frames, 6-全景_frames |
| 6 | `24.12.16` | 5 | 7,200 | 1-全景_frames, 2-全景_frames, 3-全景_frames, 4-全景_frames, 6-全景_frames |
| 7 | `25.8.13` | 2 | 2,895 | 12-全景_frames, 13-全景_frames |
| 8 | `25.8.14` | 2 | 2,879 | 12-全景_frames, 13-全景_frames |

**总计**: 8个日期目录，27个子目录，38,188张图片

## 📁 最终生成的目录结构

```
output/
├── 2025.06.19/                          # 日期目录
│   ├── 8-东北全景_frames/              # 子目录1
│   │   ├── images/                     # 标注图片
│   │   ├── test_results.json          # 子目录报告
│   │   ├── time_statistics.csv
│   │   └── time_statistics.xlsx
│   ├── 9-西北全景_frames/              # 子目录2
│   │   └── ...
│   ├── date_summary.json               # 日期汇总报告
│   ├── date_time_statistics.csv
│   └── date_time_statistics.xlsx
│
├── 2025.09.06/                          # 日期目录
│   ├── 15-全景（两水槽）_frames/
│   ├── 16-全景（两水槽）_frames/
│   ├── 17-全景（两水槽）_frames/
│   ├── 18-全景（两水槽）_frames/
│   ├── date_summary.json
│   ├── date_time_statistics.csv
│   └── date_time_statistics.xlsx
│
├── 2025.09.07/                          # 日期目录
│   └── ...
│
├── 2025.09.08/                          # 日期目录
│   └── ...
│
├── 24.12.15/                            # 日期目录
│   ├── 1-全景_frames/
│   ├── 2-全景_frames/
│   ├── 3-全景_frames/
│   ├── 4-全景_frames/
│   ├── 6-全景_frames/
│   ├── date_summary.json
│   ├── date_time_statistics.csv
│   └── date_time_statistics.xlsx
│
├── 24.12.16/                            # 日期目录
│   └── ...
│
├── 25.8.13/                             # 日期目录
│   └── ...
│
├── 25.8.14/                             # 日期目录
│   └── ...
│
└── all_dates_analysis_summary.json      # 全局汇总报告
```

## 📊 每个目录包含的文件

### 子目录级别（每个子目录下）

- `images/` - 标注后的图片目录
- `test_results.json` - 详细分析结果（JSON格式）
- `time_statistics.csv` - 时间统计表格（CSV格式）
- `time_statistics.xlsx` - 时间统计表格（Excel格式）

### 日期级别（每个日期目录下）

- `date_summary.json` - 日期汇总报告（包含该日期所有子目录的统计）
- `date_time_statistics.csv` - 日期汇总时间统计（CSV格式）
- `date_time_statistics.xlsx` - 日期汇总时间统计（Excel格式）

### 全局级别（output根目录）

- `all_dates_analysis_summary.json` - 所有日期的汇总报告

## ⏱️ 预计执行时间

- **每张图片**: 约 0.1-0.5 秒（取决于GPU性能）
- **每个子目录（约1440张）**: 约 2-12 分钟
- **每个日期目录**: 约 6-48 分钟（取决于子目录数）
- **总计（38,188张）**: 约 **1-5 小时**（取决于GPU性能）

## 💾 磁盘空间需求

- **每张标注图片**: 约 100-500 KB
- **每个子目录**: 约 140-700 MB
- **总计（27个子目录）**: 约 **3.8-19 GB**

## 🚀 执行命令

### 方式1: 分析所有日期目录（推荐）

```bash
python scripts/batch_analyze_all_dates.py \
    --base_dir /mnt/f \
    --checkpoint checkpoints/finetuned/default_model/best/checkpoint_best_ema.pth \
    --threshold 0.3 \
    --output_dir output
```

### 方式2: 逐个日期目录分析

```bash
# 分析单个日期目录
python scripts/batch_analyze_by_date.py \
    --base_dir /mnt/f/2025.09.06/extracted_frames \
    --checkpoint checkpoints/finetuned/default_model/best/checkpoint_best_ema.pth
```

## ⚙️ 可选参数

- `--skip_existing`: 跳过已存在的报告（如果之前分析过）
- `--date_pattern "2025.09.*"`: 只分析匹配模式的日期目录
- `--threshold 0.5`: 调整置信度阈值
- `--dry_run`: 预览模式，不实际执行

## 📝 报告说明

### 子目录报告 (`test_results.json`)

包含该子目录所有图片的详细检测结果：
- 每张图片的检测数量
- 每个类别的统计
- 检测框位置和置信度

### 日期汇总报告 (`date_summary.json`)

包含该日期所有子目录的汇总：
- 总图片数
- 总检测数
- 平均检测数
- 各子目录的统计

### 全局汇总报告 (`all_dates_analysis_summary.json`)

包含所有日期的汇总：
- 总日期目录数
- 总子目录数
- 总图片数
- 各日期的状态

## ⚠️ 注意事项

1. **磁盘空间**: 确保有足够的磁盘空间（至少20GB）
2. **GPU**: 确保GPU可用（如果使用CPU会很慢，可能需要数天）
3. **中断恢复**: 如果中断，可以使用 `--skip_existing` 继续未完成的目录
4. **进度显示**: 分析过程中会显示进度（每10张图片显示一次）
5. **分批执行**: 如果担心时间过长，可以分批执行：
   ```bash
   # 先分析前4个日期
   python scripts/batch_analyze_all_dates.py --base_dir /mnt/f --date_pattern "2025.*"
   
   # 再分析后4个日期
   python scripts/batch_analyze_all_dates.py --base_dir /mnt/f --date_pattern "24.*|25.*"
   ```

## ✅ 确认执行

如果要执行此计划，请运行：

```bash
# 预览计划（不执行）
python scripts/batch_analyze_all_dates.py --base_dir /mnt/f --dry_run

# 执行分析
python scripts/batch_analyze_all_dates.py \
    --base_dir /mnt/f \
    --checkpoint checkpoints/finetuned/default_model/best/checkpoint_best_ema.pth
```

## 📚 详细目录结构

完整目录结构说明请参考：[OUTPUT_DIRECTORY_STRUCTURE.md](OUTPUT_DIRECTORY_STRUCTURE.md)

---

**生成时间**: 2025-11-23  
**计划状态**: 待确认执行  
**总图片数**: 38,188 张  
**预计总时间**: 1-5 小时
