#!/usr/bin/env python
"""
批量分析所有日期目录脚本 - 分析F盘下所有日期目录的extracted_frames子目录
"""

import argparse
import json
import re
from pathlib import Path
from datetime import datetime
import sys
import subprocess

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="批量分析所有日期目录脚本 - 分析F盘下所有日期目录",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 分析所有日期目录
  python batch_analyze_all_dates.py --base_dir /mnt/f

  # 使用微调模型
  python batch_analyze_all_dates.py \\
      --base_dir /mnt/f \\
      --checkpoint checkpoints/finetuned/default_model/best/checkpoint_best_ema.pth

  # 只分析特定日期
  python batch_analyze_all_dates.py \\
      --base_dir /mnt/f \\
      --date_pattern "2025.09.*"
        """
    )
    
    parser.add_argument("--base_dir", type=str, default="/mnt/f",
                       help="基础目录路径（默认: /mnt/f）")
    parser.add_argument("--checkpoint", type=str, 
                       default="checkpoints/finetuned/default_model/best/checkpoint_best_ema.pth",
                       help="模型检查点路径（默认: 使用微调模型）")
    parser.add_argument("--threshold", type=float, default=0.3,
                       help="置信度阈值（默认: 0.3）")
    parser.add_argument("--output_dir", type=str, default="output",
                       help="输出目录（默认: output）")
    parser.add_argument("--date_pattern", type=str, default=None,
                       help="日期模式过滤（例如: '2025.09.*'，可选）")
    parser.add_argument("--skip_existing", action="store_true",
                       help="跳过已存在的报告")
    parser.add_argument("--dry_run", action="store_true",
                       help="仅显示计划，不执行分析")
    
    return parser.parse_args()

def find_date_directories(base_dir, date_pattern=None):
    """查找所有日期格式的目录"""
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"❌ 错误: 基础目录不存在: {base_dir}")
        return []
    
    date_dirs = []
    for item in base_path.iterdir():
        if item.is_dir():
            dir_name = item.name
            # 匹配日期格式：2025.06.19, 24.12.15, 25.8.13等
            if re.match(r'^\d{2,4}[._-]\d{1,2}[._-]\d{1,2}$', dir_name):
                # 检查是否有extracted_frames子目录
                extracted_frames = item / "extracted_frames"
                if extracted_frames.exists() and extracted_frames.is_dir():
                    # 如果指定了日期模式，进行过滤
                    if date_pattern:
                        if not re.match(date_pattern.replace('*', '.*'), dir_name):
                            continue
                    date_dirs.append(extracted_frames)
    
    return sorted(date_dirs)

def count_subdirectories_and_images(date_dir):
    """统计日期目录下的子目录数和图片数"""
    subdirs = []
    total_images = 0
    
    for item in date_dir.iterdir():
        if item.is_dir():
            # 统计图片
            image_count = 0
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_count += len(list(item.glob(f"*{ext}"))) + len(list(item.glob(f"*{ext.upper()}")))
            
            if image_count > 0:
                subdirs.append({
                    'path': item,
                    'name': item.name,
                    'image_count': image_count
                })
                total_images += image_count
    
    return subdirs, total_images

def main():
    args = parse_args()
    
    print("="*70)
    print("📊 RF-DETR 批量分析所有日期目录")
    print("="*70)
    print(f"   基础目录: {args.base_dir}")
    print(f"   模型: {args.checkpoint}")
    print(f"   阈值: {args.threshold}")
    print(f"   输出目录: {args.output_dir}")
    if args.date_pattern:
        print(f"   日期过滤: {args.date_pattern}")
    if args.dry_run:
        print(f"   ⚠️  预览模式（不会实际执行）")
    print("="*70)
    
    # 检查checkpoint是否存在
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ 错误: 模型检查点不存在: {args.checkpoint}")
        return
    
    # 查找所有日期目录
    print(f"\n🔍 查找日期目录...")
    date_dirs = find_date_directories(args.base_dir, args.date_pattern)
    
    if not date_dirs:
        print(f"❌ 未找到日期格式的目录")
        return
    
    print(f"✅ 找到 {len(date_dirs)} 个日期目录")
    
    # 统计所有信息
    all_tasks = []
    total_subdirs = 0
    total_images = 0
    
    for date_dir in date_dirs:
        date_name = date_dir.parent.name
        subdirs, image_count = count_subdirectories_and_images(date_dir)
        
        all_tasks.append({
            'date': date_name,
            'date_dir': date_dir,
            'subdirs': subdirs,
            'total_images': image_count
        })
        
        total_subdirs += len(subdirs)
        total_images += image_count
    
    # 显示计划
    print(f"\n{'='*70}")
    print(f"📋 分析计划")
    print(f"{'='*70}")
    print(f"   总日期目录数: {len(date_dirs)}")
    print(f"   总子目录数: {total_subdirs}")
    print(f"   总图片数: {total_images:,}")
    print(f"\n详细列表:")
    
    for i, task in enumerate(all_tasks, 1):
        print(f"\n[{i}] {task['date']}")
        print(f"    路径: {task['date_dir']}")
        print(f"    子目录数: {len(task['subdirs'])}")
        print(f"    总图片数: {task['total_images']:,}")
        for subdir in task['subdirs']:
            print(f"      - {subdir['name']}: {subdir['image_count']} 张")
    
    if args.dry_run:
        print(f"\n{'='*70}")
        print(f"📋 预览模式 - 不会实际执行分析")
        print(f"{'='*70}")
        print(f"\n💡 要执行分析，请移除 --dry_run 参数")
        return
    
    # 确认执行
    print(f"\n{'='*70}")
    print(f"⚠️  准备开始分析")
    print(f"{'='*70}")
    print(f"   将分析 {len(date_dirs)} 个日期目录")
    print(f"   共 {total_subdirs} 个子目录")
    print(f"   共 {total_images:,} 张图片")
    
    # 使用batch_analyze_by_date.py分析每个日期目录
    print(f"\n{'='*70}")
    print(f"🚀 开始批量分析")
    print(f"{'='*70}")
    
    results = []
    skipped_count = 0
    
    for i, task in enumerate(all_tasks, 1):
        date_name = task['date']
        date_dir = task['date_dir']
        
        print(f"\n[{i}/{len(all_tasks)}] 处理日期: {date_name}")
        print(f"   子目录数: {len(task['subdirs'])}")
        print(f"   图片数: {task['total_images']:,}")
        
        # 检查是否已存在完整的日期目录报告
        if args.skip_existing:
            date_output_dir = Path(args.output_dir) / date_name
            all_subdirs_completed = True
            missing_reports = []
            
            # 使用与batch_analyze_by_date.py相同的逻辑来构建路径
            sys.path.insert(0, str(Path(__file__).parent))
            from batch_analyze_by_date import extract_folder_name_from_path
            
            # 检查所有子目录是否都有报告
            for subdir in task['subdirs']:
                subdir_path = subdir['path']
                subdir_name = subdir['name']
                
                # 使用与analyze_directory相同的逻辑
                folder_name = extract_folder_name_from_path(subdir_path)
                # 根据analyze_directory的逻辑，如果提供了date_name，输出路径是：
                # output_base_dir / date_name / folder_name.replace(f"{date_name}_", "")
                output_subdir_name = folder_name.replace(f"{date_name}_", "")
                
                # 构建报告文件路径
                # 注意：实际输出路径是嵌套的日期目录格式: date_name/date_name/子目录名/test_results.json
                # 这是因为extract_folder_name_from_path会添加日期前缀，但实际输出时又嵌套了一层
                report_json_nested = date_output_dir / date_name / output_subdir_name / "test_results.json"
                report_json_standard = date_output_dir / output_subdir_name / "test_results.json"
                report_json_direct = date_output_dir / date_name / subdir_name / "test_results.json"
                report_json_simple = date_output_dir / subdir_name / "test_results.json"
                
                if not report_json_nested.exists() and not report_json_standard.exists() and not report_json_direct.exists() and not report_json_simple.exists():
                    all_subdirs_completed = False
                    missing_reports.append(subdir_name)
                    # 调试信息（仅第一个缺失的报告）
                    if len(missing_reports) == 1:
                        print(f"   🔍 调试信息:")
                        print(f"      检查路径1 (嵌套): {report_json_nested}")
                        print(f"      检查路径2 (标准): {report_json_standard}")
                        print(f"      检查路径3 (直接): {report_json_direct}")
                        print(f"      检查路径4 (简单): {report_json_simple}")
                        print(f"      folder_name: {folder_name}")
                        print(f"      output_subdir_name: {output_subdir_name}")
            
            if all_subdirs_completed and len(task['subdirs']) > 0:
                print(f"   ⏭️  跳过（所有 {len(task['subdirs'])} 个子目录报告已存在）")
                skipped_count += 1
                results.append({
                    'date': date_name,
                    'status': 'skipped',
                    'subdirs_count': len(task['subdirs']),
                    'images_count': task['total_images'],
                    'reason': 'all_reports_exist'
                })
                continue
            elif missing_reports:
                print(f"   ⚠️  部分子目录报告缺失 ({len(missing_reports)}/{len(task['subdirs'])}): {', '.join(missing_reports[:3])}{'...' if len(missing_reports) > 3 else ''}")
                print(f"   将继续处理缺失的报告...")
        
        # 调用batch_analyze_by_date.py
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "batch_analyze_by_date.py"),
            "--base_dir", str(date_dir),
            "--checkpoint", args.checkpoint,
            "--threshold", str(args.threshold),
            "--output_dir", args.output_dir
        ]
        
        if args.skip_existing:
            cmd.append("--skip_existing")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"   ✅ 完成")
            results.append({
                'date': date_name,
                'status': 'success',
                'subdirs_count': len(task['subdirs']),
                'images_count': task['total_images']
            })
        except subprocess.CalledProcessError as e:
            print(f"   ❌ 失败: {e}")
            print(f"   错误输出: {e.stderr[:500] if e.stderr else '无错误信息'}")
            results.append({
                'date': date_name,
                'status': 'failed',
                'error': str(e)
            })
    
    # 生成汇总报告
    print(f"\n{'='*70}")
    print(f"📋 生成汇总报告")
    print(f"{'='*70}")
    
    summary = {
        'analysis_time': datetime.now().isoformat(),
        'base_directory': args.base_dir,
        'checkpoint': args.checkpoint,
        'threshold': args.threshold,
        'total_date_directories': len(date_dirs),
        'total_subdirectories': total_subdirs,
        'total_images': total_images,
        'results': results
    }
    
    summary_json = Path(args.output_dir) / "all_dates_analysis_summary.json"
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 批量分析完成!")
    print(f"   总日期目录数: {len(date_dirs)}")
    print(f"   成功: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"   跳过: {sum(1 for r in results if r['status'] == 'skipped')}")
    print(f"   失败: {sum(1 for r in results if r['status'] == 'failed')}")
    print(f"   汇总报告: {summary_json}")

if __name__ == "__main__":
    main()


