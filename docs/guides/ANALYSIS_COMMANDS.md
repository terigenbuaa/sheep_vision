# 批量分析命令

## 分析 1440 张图片的文件夹

```bash
# 分析 6-全景_frames (1440张)
python test_dataset.py /mnt/f/24.12.15/extracted_frames/6-全景_frames
```

**结果将保存在：**
- `output/analysis_24.12.15_6-全景_frames/images/` - 标注后的图片
- `output/analysis_24.12.15_6-全景_frames/test_results.json` - 详细结果JSON
- `output/analysis_24.12.15_6-全景_frames/time_statistics.csv` - 时间统计表格
- `output/analysis_24.12.15_6-全景_frames/time_statistics.xlsx` - Excel表格

## 其他文件夹分析命令

```bash
# 分析 1-全景_frames (1448张)
python test_dataset.py /mnt/f/24.12.15/extracted_frames/1-全景_frames

# 分析 2-全景_frames (976张)
python test_dataset.py /mnt/f/24.12.15/extracted_frames/2-全景_frames
```

## 使用批量脚本

```bash
# 运行批量分析脚本（只分析6-全景_frames）
./batch_analyze_folders.sh
```

## 文件夹信息

| 文件夹 | 图片数量 | 分析命令 |
|--------|---------|---------|
| 6-全景_frames | 1440张 | `python test_dataset.py /mnt/f/24.12.15/extracted_frames/6-全景_frames` |
| 1-全景_frames | 1448张 | `python test_dataset.py /mnt/f/24.12.15/extracted_frames/1-全景_frames` |
| 2-全景_frames | 976张 | `python test_dataset.py /mnt/f/24.12.15/extracted_frames/2-全景_frames` |

## 注意事项

1. 每个文件夹会生成独立的分析结果文件夹
2. 文件夹名称格式：`analysis_24.12.15_文件夹名`
3. 分析时间取决于图片数量（约每张1-2秒）
4. 1440张图片预计需要约30-50分钟

