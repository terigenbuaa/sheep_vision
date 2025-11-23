# 批量分析结果目录结构

本文档展示批量分析后生成的完整目录结构。

## 📁 最终目录结构

```
output/
├── 2025.06.19/                          # 日期目录
│   ├── 8-东北全景_frames/              # 子目录1
│   │   ├── images/                     # 标注后的图片
│   │   │   ├── annotated_*.jpg        # 每张图片的标注结果
│   │   │   └── ...
│   │   ├── test_results.json          # 详细分析结果（JSON）
│   │   ├── time_statistics.csv        # 时间统计表格（CSV）
│   │   └── time_statistics.xlsx       # 时间统计表格（Excel）
│   │
│   ├── 9-西北全景_frames/              # 子目录2
│   │   ├── images/
│   │   ├── test_results.json
│   │   ├── time_statistics.csv
│   │   └── time_statistics.xlsx
│   │
│   ├── date_summary.json               # 日期汇总报告（JSON）
│   ├── date_time_statistics.csv       # 日期汇总时间统计（CSV）
│   └── date_time_statistics.xlsx      # 日期汇总时间统计（Excel）
│
├── 2025.09.06/                          # 日期目录
│   ├── 15-全景（两水槽）_frames/
│   │   ├── images/
│   │   ├── test_results.json
│   │   ├── time_statistics.csv
│   │   └── time_statistics.xlsx
│   │
│   ├── 16-全景（两水槽）_frames/
│   │   ├── images/
│   │   ├── test_results.json
│   │   ├── time_statistics.csv
│   │   └── time_statistics.xlsx
│   │
│   ├── 17-全景（两水槽）_frames/
│   │   └── ...
│   │
│   ├── 18-全景（两水槽）_frames/
│   │   └── ...
│   │
│   ├── date_summary.json               # 日期汇总报告
│   ├── date_time_statistics.csv
│   └── date_time_statistics.xlsx
│
├── 2025.09.07/                          # 日期目录
│   ├── 15-全景（两水槽）_frames/
│   ├── 16-全景（两水槽）_frames/
│   ├── 17-全景（两水槽）_frames/
│   ├── date_summary.json
│   ├── date_time_statistics.csv
│   └── date_time_statistics.xlsx
│
├── 2025.09.08/                          # 日期目录
│   ├── 15-全景（两水槽）_frames/
│   ├── 16-全景（两水槽）_frames/
│   ├── 17-全景（两水槽）_frames/
│   ├── 18-全景（两水槽）_frames/
│   ├── date_summary.json
│   ├── date_time_statistics.csv
│   └── date_time_statistics.xlsx
│
├── 24.12.15/                            # 日期目录
│   ├── 1-全景_frames/
│   │   ├── images/
│   │   ├── test_results.json
│   │   ├── time_statistics.csv
│   │   └── time_statistics.xlsx
│   │
│   ├── 2-全景_frames/
│   │   └── ...
│   │
│   ├── 3-全景_frames/
│   │   └── ...
│   │
│   ├── 4-全景_frames/
│   │   └── ...
│   │
│   ├── 6-全景_frames/
│   │   └── ...
│   │
│   ├── date_summary.json               # 日期汇总报告
│   ├── date_time_statistics.csv
│   └── date_time_statistics.xlsx
│
├── 24.12.16/                            # 日期目录
│   ├── 1-全景_frames/
│   ├── 2-全景_frames/
│   ├── 3-全景_frames/
│   ├── 4-全景_frames/
│   ├── 6-全景_frames/
│   ├── date_summary.json
│   ├── date_time_statistics.csv
│   └── date_time_statistics.xlsx
│
├── 25.8.13/                             # 日期目录
│   ├── 12-全景_frames/
│   │   ├── images/
│   │   ├── test_results.json
│   │   ├── time_statistics.csv
│   │   └── time_statistics.xlsx
│   │
│   ├── 13-全景_frames/
│   │   └── ...
│   │
│   ├── date_summary.json               # 日期汇总报告
│   ├── date_time_statistics.csv
│   └── date_time_statistics.xlsx
│
├── 25.8.14/                             # 日期目录
│   ├── 12-全景_frames/
│   ├── 13-全景_frames/
│   ├── date_summary.json
│   ├── date_time_statistics.csv
│   └── date_time_statistics.xlsx
│
└── all_dates_analysis_summary.json      # 全局汇总报告（所有日期）
```

## 📊 文件说明

### 子目录级别文件（每个子目录下）

1. **`images/`** - 标注后的图片目录
   - `annotated_*.jpg` - 每张原始图片的标注结果
   - 文件名格式：`annotated_<原文件名>.jpg`

2. **`test_results.json`** - 详细分析结果（JSON格式）
   ```json
   {
     "checkpoint": "模型路径",
     "dataset_dir": "数据目录",
     "threshold": 0.3,
     "total_images": 1440,
     "total_detections": 5000,
     "results": [
       {
         "image_name": "图片名",
         "num_detections": 3,
         "class_counts": {...},
         "detections": [...]
       },
       ...
     ]
   }
   ```

3. **`time_statistics.csv`** - 时间统计表格（CSV格式）
   - 按分钟统计每个类别的检测数量
   - 包含：时间、类别1数量、类别2数量、类别3数量、总计

4. **`time_statistics.xlsx`** - 时间统计表格（Excel格式）
   - 与CSV内容相同，Excel格式便于查看和编辑

### 日期级别文件（每个日期目录下）

1. **`date_summary.json`** - 日期汇总报告（JSON格式）
   ```json
   {
     "date": "2025.09.06",
     "analysis_time": "2025-11-23T10:00:00",
     "total_subdirectories": 4,
     "total_images": 5760,
     "total_detections": 20000,
     "average_detections_per_image": 3.47,
     "subdirectories": [
       {
         "subdir_name": "15-全景（两水槽）_frames",
         "images": 1440,
         "detections": 5000,
         "report_path": "15-全景（两水槽）_frames/test_results.json"
       },
       ...
     ]
   }
   ```

2. **`date_time_statistics.csv`** - 日期汇总时间统计（CSV格式）
   - 合并该日期下所有子目录的时间统计
   - 按分钟统计整个日期的检测情况

3. **`date_time_statistics.xlsx`** - 日期汇总时间统计（Excel格式）
   - 与CSV内容相同，Excel格式

### 全局汇总文件（output根目录）

**`all_dates_analysis_summary.json`** - 所有日期的汇总报告
```json
{
  "analysis_time": "2025-11-23T12:00:00",
  "base_directory": "/mnt/f",
  "checkpoint": "模型路径",
  "threshold": 0.3,
  "total_date_directories": 8,
  "total_subdirectories": 27,
  "total_images": 38188,
  "results": [
    {
      "date": "2025.09.06",
      "status": "success",
      "subdirs_count": 4,
      "images_count": 5760
    },
    ...
  ]
}
```

## 📈 目录统计

### 按日期统计

| 日期 | 子目录数 | 输出路径 |
|------|---------|---------|
| 2025.06.19 | 2 | `output/2025.06.19/` |
| 2025.09.06 | 4 | `output/2025.09.06/` |
| 2025.09.07 | 3 | `output/2025.09.07/` |
| 2025.09.08 | 4 | `output/2025.09.08/` |
| 24.12.15 | 5 | `output/24.12.15/` |
| 24.12.16 | 5 | `output/24.12.16/` |
| 25.8.13 | 2 | `output/25.8.13/` |
| 25.8.14 | 2 | `output/25.8.14/` |

**总计**: 8个日期目录，27个子目录

## 🔍 查看报告示例

### 查看单个子目录的报告

```bash
# 查看JSON结果
cat output/2025.09.06/15-全景（两水槽）_frames/test_results.json

# 查看时间统计
cat output/2025.09.06/15-全景（两水槽）_frames/time_statistics.csv

# 查看标注图片
ls output/2025.09.06/15-全景（两水槽）_frames/images/
```

### 查看日期汇总报告

```bash
# 查看日期汇总
cat output/2025.09.06/date_summary.json

# 查看日期时间统计
cat output/2025.09.06/date_time_statistics.csv
```

### 查看全局汇总

```bash
# 查看所有日期汇总
cat output/all_dates_analysis_summary.json
```

## 📋 目录层级说明

```
output/                          # 输出根目录
└── <日期>/                      # 第1层：日期目录（8个）
    ├── <子目录名>/              # 第2层：子目录（每个日期2-5个）
    │   ├── images/             # 第3层：标注图片
    │   ├── test_results.json   # 子目录报告
    │   ├── time_statistics.csv
    │   └── time_statistics.xlsx
    │
    ├── date_summary.json        # 日期汇总报告
    ├── date_time_statistics.csv
    └── date_time_statistics.xlsx
```

## 💡 优势

1. **按日期组织**：每个日期的所有结果集中在一个目录
2. **层次清晰**：日期 → 子目录 → 图片，结构清晰
3. **多级汇总**：
   - 子目录级别：每张图片的详细结果
   - 日期级别：该日期所有子目录的汇总
   - 全局级别：所有日期的汇总
4. **易于查找**：通过日期快速定位到相关报告
5. **便于对比**：可以轻松对比不同日期的数据

## 🎯 使用场景

- **查看特定日期的数据**：直接进入 `output/2025.09.06/`
- **查看特定子目录**：进入 `output/2025.09.06/15-全景（两水槽）_frames/`
- **对比不同日期**：查看各日期目录下的 `date_summary.json`
- **全局概览**：查看 `output/all_dates_analysis_summary.json`

---

**生成时间**: 2025-11-23  
**目录结构版本**: v1.0


