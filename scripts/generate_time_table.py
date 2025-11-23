#!/usr/bin/env python
"""
生成时间统计表格 - 按分钟统计三个标签的数量
"""

import json
import re
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import csv

def extract_time_from_filename(filename):
    """
    从文件名中提取时间信息
    文件名格式示例1: 15.05.00-15.10.00[R][0@0][0]_15-08-30.png
    文件名格式示例2: project-7-at-2025-11_207f6a55-11.15.00-11.20.00R000_11-17-30.png
    提取: 15-08-30 -> 15-08-30 (保持HH-MM-SS格式)
    """
    # 优先匹配文件名末尾的时间格式 _HH-MM-SS
    pattern1 = r'_(\d{2})-(\d{2})-(\d{2})\.(png|jpg|jpeg|PNG|JPG|JPEG)'
    match = re.search(pattern1, filename)
    if match:
        hour, minute, second = match.groups()[:3]
        try:
            # 返回完整时间格式 HH-MM-SS
            return f"{int(hour):02d}-{int(minute):02d}-{int(second):02d}"
        except:
            pass
    
    # 尝试匹配其他时间格式 HH-MM-SS 或 HH:MM:SS
    patterns = [
        r'(\d{2})-(\d{2})-(\d{2})',  # 11-17-30
        r'(\d{2}):(\d{2}):(\d{2})',  # 11:17:30
        r'(\d{2})\.(\d{2})\.(\d{2})',  # 11.17.30
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            hour, minute, second = match.groups()
            try:
                # 返回完整时间格式 HH-MM-SS
                return f"{int(hour):02d}-{int(minute):02d}-{int(second):02d}"
            except:
                continue
    
    return None

def parse_time_range_from_filename(filename):
    """
    从文件名中提取时间范围
    文件名格式示例: project-7-at-2025-11_207f6a55-11.15.00-11.20.00R000_11-17-30.png
    提取时间范围: 11.15.00-11.20.00
    """
    # 匹配时间范围格式
    pattern = r'(\d{2})\.(\d{2})\.(\d{2})-(\d{2})\.(\d{2})\.(\d{2})'
    match = re.search(pattern, filename)
    
    if match:
        h1, m1, s1, h2, m2, s2 = match.groups()
        try:
            start_time = f"{int(h1):02d}:{int(m1):02d}"
            end_time = f"{int(h2):02d}:{int(m2):02d}"
            return start_time, end_time
        except:
            pass
    
    return None, None

def load_test_results(json_path):
    """加载测试结果JSON文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_time_table(json_path, output_csv=None, output_excel=None):
    """
    生成时间统计表格
    
    Args:
        json_path: 测试结果JSON文件路径
        output_csv: 输出CSV文件路径（可选）
        output_excel: 输出Excel文件路径（可选）
    """
    print("=" * 70)
    print("📊 生成时间统计表格")
    print("=" * 70)
    
    # 加载测试结果
    print(f"\n📖 加载测试结果: {json_path}")
    data = load_test_results(json_path)
    
    # 按时间（HH-MM-SS格式）统计
    time_stats = defaultdict(lambda: {
        'normal': 0,
        'offence_eating': 0,
        'offence_not_eating': 0,
        'total': 0,
        'image_count': 0
    })
    
    print(f"\n🔍 分析 {len(data['results'])} 张图片...")
    
    for result in data['results']:
        if 'error' in result:
            continue
        
        image_name = result['image_name']
        class_counts = result.get('class_counts', {})
        
        # 提取时间信息（HH-MM-SS格式）
        time_str = extract_time_from_filename(image_name)
        
        if not time_str:
            # 如果无法从文件名提取，尝试从时间范围中提取
            start_time, end_time = parse_time_range_from_filename(image_name)
            if start_time:
                # 转换为HH-MM-SS格式（假设秒为00）
                time_str = f"{start_time.replace(':', '-')}-00"
            else:
                print(f"   ⚠️  无法从文件名提取时间: {image_name}")
                continue
        
        # 统计各类别数量
        time_stats[time_str]['normal'] += class_counts.get('normal', 0)
        time_stats[time_str]['offence_eating'] += class_counts.get('offence_eating', 0)
        time_stats[time_str]['offence_not_eating'] += class_counts.get('offence_not_eating', 0)
        time_stats[time_str]['total'] += result.get('num_detections', 0)
        time_stats[time_str]['image_count'] += 1
    
    # 转换为列表并排序（按时间排序）
    table_data = []
    for time_str in sorted(time_stats.keys()):
        stats = time_stats[time_str]
        table_data.append({
            '时间': time_str,  # HH-MM-SS格式
            'normal': stats['normal'],
            'offence_eating': stats['offence_eating'],
            'offence_not_eating': stats['offence_not_eating'],
            'total': stats['total'],
            '图片数量': stats['image_count']
        })
    
    print(f"\n✅ 统计完成!")
    if table_data:
        print(f"   时间范围: {table_data[0]['时间']} - {table_data[-1]['时间']}")
        print(f"   总记录数: {len(table_data)}")
    
    # 保存CSV
    if output_csv:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:  # utf-8-sig for Excel compatibility
            writer = csv.DictWriter(f, fieldnames=['时间', 'normal', 'offence_eating', 'offence_not_eating', 'total', '图片数量'])
            writer.writeheader()
            writer.writerows(table_data)
        
        print(f"\n💾 CSV表格已保存: {output_csv}")
    
    # 保存Excel（如果安装了pandas和openpyxl）
    if output_excel:
        try:
            import pandas as pd
            
            df = pd.DataFrame(table_data)
            output_excel = Path(output_excel)
            output_excel.parent.mkdir(parents=True, exist_ok=True)
            
            df.to_excel(output_excel, index=False, engine='openpyxl')
            print(f"💾 Excel表格已保存: {output_excel}")
        except ImportError:
            print("⚠️  无法保存Excel文件，需要安装: pip install pandas openpyxl")
    
    # 显示前几行预览
    print(f"\n📋 表格预览（前10行）:")
    print("-" * 80)
    print(f"{'时间':<12} {'normal':<10} {'offence_eating':<15} {'offence_not_eating':<18} {'total':<8} {'图片数量':<8}")
    print("-" * 80)
    for row in table_data[:10]:
        print(f"{row['时间']:<12} {row['normal']:<10} {row['offence_eating']:<15} {row['offence_not_eating']:<18} {row['total']:<8} {row['图片数量']:<8}")
    if len(table_data) > 10:
        print(f"... (共 {len(table_data)} 行)")
    
    print("\n" + "=" * 70)
    print("✅ 完成！")
    print("=" * 70)
    
    return table_data

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="生成时间统计表格 - 按分钟统计检测结果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认路径生成表格
  python generate_time_table.py

  # 指定输入JSON文件
  python generate_time_table.py --input output/analysis_test/test_results.json

  # 指定所有输出文件
  python generate_time_table.py --input results.json --csv output.csv --excel output.xlsx
        """
    )
    
    parser.add_argument("--input", type=str, default="output/test_batch_results/test_results.json",
                       help="输入JSON文件路径（默认: output/test_batch_results/test_results.json）")
    parser.add_argument("--csv", type=str, default=None,
                       help="输出CSV文件路径（默认: <输入目录>/time_statistics.csv）")
    parser.add_argument("--excel", type=str, default=None,
                       help="输出Excel文件路径（默认: <输入目录>/time_statistics.xlsx）")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    json_path = args.input
    
    # 确定输出路径
    json_file = Path(json_path)
    if args.csv:
        output_csv = args.csv
    else:
        output_csv = str(json_file.parent / "time_statistics.csv")
    
    if args.excel:
        output_excel = args.excel
    else:
        output_excel = str(json_file.parent / "time_statistics.xlsx")
    
    # 检查输入文件是否存在
    if not json_file.exists():
        print(f"❌ 错误: 测试结果文件不存在: {json_path}")
        print(f"💡 提示: 请先运行批量测试: python test_dataset.py <数据集目录>")
        return
    
    # 生成表格
    generate_time_table(json_path, output_csv, output_excel)

if __name__ == "__main__":
    main()

