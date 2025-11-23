#!/usr/bin/env python
"""
按日期批量分析脚本 - 分析指定目录下所有子目录，生成每个目录的报告
生成格式类似: output/analysis_24.12.15_6-全景_frames
"""

import argparse
import json
import re
from pathlib import Path
from datetime import datetime
import sys

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="按日期批量分析脚本 - 分析目录下所有子目录，生成每个目录的报告",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 分析F盘下的所有子目录
  python batch_analyze_by_date.py --base_dir /mnt/f/2025.09.06/extracted_frames

  # 使用微调模型
  python batch_analyze_by_date.py \\
      --base_dir /mnt/f/2025.09.06/extracted_frames \\
      --checkpoint checkpoints/finetuned/default_model/best/checkpoint_best_ema.pth

  # 指定输出目录
  python batch_analyze_by_date.py \\
      --base_dir /mnt/f/2025.09.06/extracted_frames \\
      --output_dir output

  # 只分析特定目录
  python batch_analyze_by_date.py \\
      --base_dir /mnt/f/2025.09.06/extracted_frames \\
      --dir_pattern "15-全景.*"
        """
    )
    
    parser.add_argument("--base_dir", type=str, required=True,
                       help="基础目录路径（包含子目录的目录，例如: /mnt/f/2025.09.06/extracted_frames）")
    parser.add_argument("--checkpoint", type=str, 
                       default="checkpoints/finetuned/default_model/best/checkpoint_best_ema.pth",
                       help="模型检查点路径（默认: 使用微调模型）")
    parser.add_argument("--threshold", type=float, default=0.3,
                       help="置信度阈值（默认: 0.3）")
    parser.add_argument("--output_dir", type=str, default="output",
                       help="输出目录（默认: output）")
    parser.add_argument("--dir_pattern", type=str, default=None,
                       help="目录名模式过滤（例如: '15-全景.*'，可选）")
    parser.add_argument("--skip_existing", action="store_true",
                       help="跳过已存在的报告")
    parser.add_argument("--dry_run", action="store_true",
                       help="仅显示计划，不执行分析")
    
    return parser.parse_args()

def extract_folder_name_from_path(path):
    """从路径中提取文件夹名称作为分析文件夹名（类似test_dataset.py的逻辑）"""
    path_obj = Path(path)
    
    if path_obj.is_file():
        path_obj = path_obj.parent
    
    parts = path_obj.parts
    folder_name = path_obj.name
    
    # 向上查找包含日期格式的目录
    date_folder = None
    for part in reversed(parts):
        if re.match(r'^\d+\.\d+\.\d+$', part) or re.match(r'^\d+_\d+_\d+$', part):
            date_folder = part
            break
    
    if date_folder:
        # 将日期格式统一为 24.12.15 格式
        date_normalized = date_folder.replace('_', '.')
        folder_name = f"{date_normalized}_{folder_name}"
    
    # 清理文件夹名中的特殊字符，替换为下划线（保留点、横线和下划线）
    folder_name = re.sub(r'[^\w\.-]', '_', folder_name)
    
    return folder_name

def find_subdirectories(base_dir, dir_pattern=None):
    """查找所有包含图片的子目录"""
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"❌ 错误: 基础目录不存在: {base_dir}")
        return []
    
    subdirs = []
    for item in base_path.iterdir():
        if item.is_dir():
            dir_name = item.name
            # 如果指定了目录模式，进行过滤
            if dir_pattern:
                if not re.match(dir_pattern.replace('*', '.*'), dir_name):
                    continue
            
            # 检查目录中是否有图片
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            has_images = False
            for ext in image_extensions:
                if list(item.glob(f"*{ext}")) or list(item.glob(f"*{ext.upper()}")):
                    has_images = True
                    break
            
            if has_images:
                subdirs.append(item)
    
    return sorted(subdirs)

def analyze_directory(subdir, checkpoint, threshold, output_base_dir, date_name=None):
    """分析单个子目录（使用test_dataset.py的逻辑）"""
    folder_name = extract_folder_name_from_path(subdir)
    
    # 如果提供了日期名称，使用日期作为主目录
    if date_name:
        output_dir = Path(output_base_dir) / date_name / folder_name.replace(f"{date_name}_", "")
    else:
        output_dir_name = f"analysis_{folder_name}"
        output_dir = Path(output_base_dir) / output_dir_name
    
    print(f"\n{'='*70}")
    print(f"📁 分析目录: {subdir.name}")
    print(f"   输出目录: {output_dir}")
    print(f"{'='*70}")
    
    # 检查是否已存在报告
    report_json = output_dir / "test_results.json"
    if report_json.exists() and not args.skip_existing:
        print(f"⚠️  报告已存在: {report_json}")
        print(f"   使用 --skip_existing 跳过，或删除文件重新生成")
        return str(report_json)
    
    # 使用test_dataset.py进行分析
    print(f"🔍 开始分析...")
    print(f"   模型: {checkpoint}")
    print(f"   阈值: {threshold}")
    
    try:
        # 导入test_dataset模块
        sys.path.insert(0, str(Path(__file__).parent))
        from test_dataset import load_trained_model, test_single_image
        
        # 加载模型
        model = load_trained_model(checkpoint)
        
        # 查找所有图片
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        images = []
        for ext in image_extensions:
            images.extend(list(subdir.glob(f"*{ext}")))
            images.extend(list(subdir.glob(f"*{ext.upper()}")))
        
        if not images:
            print(f"⚠️  警告: {subdir.name} 中没有找到图片文件")
            return None
        
        print(f"📸 找到 {len(images)} 张图片")
        
        # 创建输出目录结构
        images_dir = output_dir / "images"
        output_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # 分析所有图片
        results = []
        total_detections = 0
        
        print(f"\n🔍 开始批量分析...")
        for i, image_path in enumerate(images, 1):
            try:
                result, saved_path = test_single_image(
                    model, image_path, images_dir, threshold=threshold
                )
                results.append(result)
                total_detections += result.get('num_detections', 0)
                
                if i % 10 == 0:
                    print(f"   已处理: {i}/{len(images)} 张图片")
            except Exception as e:
                print(f"   ⚠️  处理 {image_path.name} 时出错: {e}")
                results.append({
                    'image_name': image_path.name,
                    'error': str(e)
                })
        
        # 生成报告（格式与test_dataset.py一致）
        summary = {
            'checkpoint': checkpoint,
            'dataset_dir': str(subdir),
            'threshold': threshold,
            'total_images': len(images),
            'total_detections': total_detections,
            'results': results
        }
        
        # 保存结果JSON
        with open(report_json, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 生成时间统计表格
        print(f"\n📊 生成时间统计表格...")
        try:
            from generate_time_table import generate_time_table
            csv_path = output_dir / "time_statistics.csv"
            excel_path = output_dir / "time_statistics.xlsx"
            generate_time_table(str(report_json), str(csv_path), str(excel_path))
            print(f"   ✅ CSV表格: {csv_path}")
            print(f"   ✅ Excel表格: {excel_path}")
        except Exception as e:
            print(f"   ⚠️  生成表格时出错: {e}")
        
        print(f"\n✅ 分析完成!")
        print(f"   总图片数: {len(images)}")
        print(f"   总检测数: {total_detections}")
        print(f"   平均每张: {total_detections/len(images):.2f} 个目标")
        print(f"   报告保存: {report_json}")
        
        return str(report_json)
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_date_summary(date_name, date_output_dir, subdir_reports):
    """生成日期级别的汇总报告"""
    print(f"\n📊 生成日期汇总报告: {date_name}")
    
    # 汇总所有子目录的结果
    total_images = 0
    total_detections = 0
    subdir_summaries = []
    
    for report_path in subdir_reports:
        if not report_path:
            continue
        
        report_file = Path(report_path)
        if not report_file.exists():
            continue
        
        try:
            with open(report_file, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
            
            subdir_name = report_file.parent.name
            total_images += report_data.get('total_images', 0)
            total_detections += report_data.get('total_detections', 0)
            
            subdir_summaries.append({
                'subdir_name': subdir_name,
                'images': report_data.get('total_images', 0),
                'detections': report_data.get('total_detections', 0),
                'report_path': str(report_file.relative_to(date_output_dir))
            })
        except Exception as e:
            print(f"   ⚠️  读取报告失败 {report_path}: {e}")
    
    # 生成日期汇总报告
    date_summary = {
        'date': date_name,
        'analysis_time': datetime.now().isoformat(),
        'total_subdirectories': len(subdir_summaries),
        'total_images': total_images,
        'total_detections': total_detections,
        'average_detections_per_image': total_detections / total_images if total_images > 0 else 0,
        'subdirectories': subdir_summaries
    }
    
    # 保存日期汇总JSON
    summary_json = date_output_dir / "date_summary.json"
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump(date_summary, f, indent=2, ensure_ascii=False)
    
    # 生成日期级别的时间统计（合并所有子目录）
    try:
        from generate_time_table import generate_time_table
        
        # 合并所有子目录的结果
        merged_results = {
            'checkpoint': subdir_summaries[0].get('checkpoint', '') if subdir_summaries else '',
            'dataset_dir': date_name,
            'threshold': 0.3,
            'total_images': total_images,
            'total_detections': total_detections,
            'results': []
        }
        
        for report_path in subdir_reports:
            if not report_path:
                continue
            report_file = Path(report_path)
            if report_file.exists():
                try:
                    with open(report_file, 'r', encoding='utf-8') as f:
                        report_data = json.load(f)
                    merged_results['results'].extend(report_data.get('results', []))
                except:
                    pass
        
        # 保存临时JSON用于生成表格
        temp_json = date_output_dir / "temp_merged_results.json"
        with open(temp_json, 'w', encoding='utf-8') as f:
            json.dump(merged_results, f, indent=2, ensure_ascii=False)
        
        csv_path = date_output_dir / "date_time_statistics.csv"
        excel_path = date_output_dir / "date_time_statistics.xlsx"
        generate_time_table(str(temp_json), str(csv_path), str(excel_path))
        
        # 删除临时文件
        temp_json.unlink()
        
        print(f"   ✅ 日期汇总报告: {summary_json}")
        print(f"   ✅ 日期时间统计: {csv_path}, {excel_path}")
    except Exception as e:
        print(f"   ⚠️  生成日期汇总表格时出错: {e}")

def main():
    global args
    args = parse_args()
    
    # 从基础目录提取日期名称
    base_path = Path(args.base_dir)
    date_name = None
    if 'extracted_frames' in base_path.parts:
        # 如果路径包含extracted_frames，提取父目录作为日期
        date_name = base_path.parent.name
    
    print("="*70)
    print("📊 RF-DETR 批量分析脚本")
    print("="*70)
    print(f"   基础目录: {args.base_dir}")
    if date_name:
        print(f"   日期: {date_name}")
    print(f"   模型: {args.checkpoint}")
    print(f"   阈值: {args.threshold}")
    print(f"   输出目录: {args.output_dir}")
    if args.dir_pattern:
        print(f"   目录过滤: {args.dir_pattern}")
    if args.dry_run:
        print(f"   ⚠️  预览模式（不会实际执行）")
    print("="*70)
    
    # 检查checkpoint是否存在
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ 错误: 模型检查点不存在: {args.checkpoint}")
        print(f"💡 提示: 请确保模型文件存在")
        return
    
    # 查找所有子目录
    print(f"\n🔍 查找包含图片的子目录...")
    subdirs = find_subdirectories(args.base_dir, args.dir_pattern)
    
    if not subdirs:
        print(f"❌ 未找到包含图片的子目录")
        return
    
    # 确定输出目录结构
    if date_name:
        date_output_dir = Path(args.output_dir) / date_name
        print(f"✅ 找到 {len(subdirs)} 个子目录（将保存到: {date_output_dir}/）:")
    else:
        date_output_dir = None
        print(f"✅ 找到 {len(subdirs)} 个子目录:")
    
    for i, subdir in enumerate(subdirs, 1):
        # 统计图片数量
        image_count = 0
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_count += len(list(subdir.glob(f"*{ext}"))) + len(list(subdir.glob(f"*{ext.upper()}")))
        
        if date_name:
            # 提取子目录名（不包含日期前缀）
            subdir_name = subdir.name
            output_path = date_output_dir / subdir_name if date_output_dir else Path(args.output_dir) / f"analysis_{extract_folder_name_from_path(subdir)}"
        else:
            output_name = extract_folder_name_from_path(subdir)
            output_path = Path(args.output_dir) / f"analysis_{output_name}"
        
        status = ""
        if (output_path / "test_results.json").exists():
            status = " (已存在)"
        
        print(f"   [{i}/{len(subdirs)}] {subdir.name}")
        print(f"       图片数: {image_count}")
        if date_name:
            print(f"       输出: {date_name}/{subdir_name}{status}")
        else:
            print(f"       输出: analysis_{output_name}{status}")
    
    if args.dry_run:
        print(f"\n{'='*70}")
        print(f"📋 预览模式 - 不会实际执行分析")
        print(f"{'='*70}")
        print(f"\n💡 要执行分析，请移除 --dry_run 参数")
        return
    
    # 确认执行
    print(f"\n{'='*70}")
    print(f"⚠️  准备开始分析 {len(subdirs)} 个子目录")
    print(f"{'='*70}")
    print(f"\n将生成以下报告:")
    for subdir in subdirs:
        output_name = extract_folder_name_from_path(subdir)
        print(f"   - output/analysis_{output_name}/")
    
    # 创建输出目录
    if date_name:
        output_base_dir = date_output_dir
        output_base_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_base_dir = Path(args.output_dir)
        output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # 分析每个子目录
    print(f"\n{'='*70}")
    print(f"🚀 开始批量分析")
    print(f"{'='*70}")
    
    reports = []
    for i, subdir in enumerate(subdirs, 1):
        print(f"\n[{i}/{len(subdirs)}] 处理: {subdir.name}")
        
        if args.skip_existing:
            if date_name:
                report_json = date_output_dir / subdir.name / "test_results.json"
            else:
                output_name = extract_folder_name_from_path(subdir)
                report_json = output_base_dir / f"analysis_{output_name}" / "test_results.json"
            
            if report_json.exists():
                print(f"   ⏭️  跳过（报告已存在）")
                reports.append(str(report_json))
                continue
        
        report = analyze_directory(
            subdir, args.checkpoint, args.threshold, output_base_dir, date_name
        )
        if report:
            reports.append(report)
    
    # 如果是在日期目录下，生成日期汇总报告
    if date_name and reports:
        generate_date_summary(date_name, date_output_dir, reports)
    
    # 生成汇总报告
    print(f"\n{'='*70}")
    print(f"📋 生成汇总报告")
    print(f"{'='*70}")
    
    summary = {
        'analysis_time': datetime.now().isoformat(),
        'base_directory': args.base_dir,
        'checkpoint': args.checkpoint,
        'threshold': args.threshold,
        'total_directories': len(subdirs),
        'successful_analyses': len(reports),
        'reports': reports
    }
    
    summary_json = output_base_dir / "batch_analysis_summary.json"
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 批量分析完成!")
    print(f"   总目录数: {len(subdirs)}")
    print(f"   成功分析数: {len(reports)}")
    print(f"   汇总报告: {summary_json}")
    print(f"\n💡 查看报告:")
    for report in reports:
        print(f"   - {report}")

if __name__ == "__main__":
    main()
