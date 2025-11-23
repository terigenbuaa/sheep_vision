#!/usr/bin/env python
"""
数据集整理脚本 - 将 Label Studio 导出的 COCO 格式数据整理成 RF-DETR 要求的格式
"""

import json
import os
import shutil
from pathlib import Path
from collections import defaultdict
import random

def load_coco_json(json_path):
    """加载 COCO JSON 文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_coco_json(data, json_path):
    """保存 COCO JSON 文件"""
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def fix_image_paths(coco_data, images_dir):
    """
    修复 COCO JSON 中的图片路径
    将服务器路径或绝对路径转换为相对路径（仅文件名）
    """
    images_dir = Path(images_dir)
    
    # 创建图片文件名到完整路径的映射
    image_files = {}
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        for img_file in images_dir.glob(ext):
            image_files[img_file.name] = img_file
    
    # 修复图片路径
    fixed_images = []
    missing_images = []
    
    for img in coco_data.get('images', []):
        original_path = img['file_name']
        
        # 提取文件名
        filename = os.path.basename(original_path)
        # 处理 URL 编码的文件名（如 %E5%89%AF%E6%9C%AC.jpg）
        try:
            from urllib.parse import unquote
            filename = unquote(filename)
        except:
            pass
        
        # 查找对应的图片文件
        if filename in image_files:
            # 更新为相对路径（仅文件名）
            img['file_name'] = filename
            fixed_images.append(img)
        else:
            # 尝试直接匹配（不区分大小写）
            found = False
            for img_file_name, img_file_path in image_files.items():
                if img_file_name.lower() == filename.lower():
                    img['file_name'] = img_file_name
                    fixed_images.append(img)
                    found = True
                    break
            
            if not found:
                missing_images.append(filename)
                print(f"⚠️  警告: 找不到图片文件 {filename}")
    
    coco_data['images'] = fixed_images
    
    # 更新标注中的 image_id 引用
    valid_image_ids = {img['id'] for img in fixed_images}
    coco_data['annotations'] = [
        ann for ann in coco_data.get('annotations', [])
        if ann.get('image_id') in valid_image_ids
    ]
    
    return coco_data, len(missing_images)

def split_dataset(coco_data, train_ratio=0.7, valid_ratio=0.2, test_ratio=0.1, seed=42):
    """
    划分数据集为 train/valid/test
    """
    random.seed(seed)
    
    # 获取所有图片 ID
    image_ids = [img['id'] for img in coco_data['images']]
    random.shuffle(image_ids)
    
    # 计算划分点
    total = len(image_ids)
    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * valid_ratio)
    
    train_ids = set(image_ids[:train_end])
    valid_ids = set(image_ids[train_end:valid_end])
    test_ids = set(image_ids[valid_end:])
    
    # 创建划分后的数据
    splits = {
        'train': {'images': [], 'annotations': []},
        'valid': {'images': [], 'annotations': []},
        'test': {'images': [], 'annotations': []}
    }
    
    # 划分图片
    for img in coco_data['images']:
        img_id = img['id']
        if img_id in train_ids:
            splits['train']['images'].append(img)
        elif img_id in valid_ids:
            splits['valid']['images'].append(img)
        elif img_id in test_ids:
            splits['test']['images'].append(img)
    
    # 划分标注
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id in train_ids:
            splits['train']['annotations'].append(ann)
        elif img_id in valid_ids:
            splits['valid']['annotations'].append(ann)
        elif img_id in test_ids:
            splits['test']['annotations'].append(ann)
    
    # 保持 categories 和 info
    for split_name in splits:
        splits[split_name]['categories'] = coco_data.get('categories', [])
        splits[split_name]['info'] = coco_data.get('info', {})
    
    return splits

def prepare_dataset(
    input_json_path,
    images_dir,
    output_dir="dataset",
    train_ratio=0.7,
    valid_ratio=0.2,
    test_ratio=0.1,
    copy_images=True,
    train_only=False
):
    """
    准备 RF-DETR 训练数据集
    
    Args:
        input_json_path: Label Studio 导出的 COCO JSON 文件路径
        images_dir: 图片文件所在目录
        output_dir: 输出数据集目录
        train_ratio: 训练集比例
        valid_ratio: 验证集比例
        test_ratio: 测试集比例
        copy_images: 是否复制图片文件（True）或使用符号链接（False）
    """
    print("="*60)
    print("📦 数据集整理脚本")
    print("="*60)
    
    # 检查输入文件
    if not os.path.exists(input_json_path):
        print(f"❌ 错误: 找不到 JSON 文件: {input_json_path}")
        return False
    
    images_dir = Path(images_dir)
    if not images_dir.exists():
        print(f"❌ 错误: 找不到图片目录: {images_dir}")
        return False
    
    # 加载 COCO JSON
    print(f"\n📖 加载 COCO JSON: {input_json_path}")
    coco_data = load_coco_json(input_json_path)
    
    print(f"   图片数量: {len(coco_data.get('images', []))}")
    print(f"   标注数量: {len(coco_data.get('annotations', []))}")
    print(f"   类别数量: {len(coco_data.get('categories', []))}")
    
    # 修复图片路径
    print(f"\n🔧 修复图片路径...")
    coco_data, missing_count = fix_image_paths(coco_data, images_dir)
    
    if missing_count > 0:
        print(f"⚠️  警告: {missing_count} 张图片未找到")
    
    print(f"✅ 修复完成，有效图片: {len(coco_data['images'])}")
    
    # 划分数据集
    if train_only:
        print(f"\n📊 所有数据用于训练（不划分验证集和测试集）...")
        splits = {
            'train': {
                'images': coco_data['images'],
                'annotations': coco_data['annotations'],
                'categories': coco_data.get('categories', []),
                'info': coco_data.get('info', {})
            },
            'valid': {
                'images': [],
                'annotations': [],
                'categories': coco_data.get('categories', []),
                'info': coco_data.get('info', {})
            },
            'test': {
                'images': [],
                'annotations': [],
                'categories': coco_data.get('categories', []),
                'info': coco_data.get('info', {})
            }
        }
    else:
        print(f"\n📊 划分数据集 (训练:{train_ratio:.0%}, 验证:{valid_ratio:.0%}, 测试:{test_ratio:.0%})...")
        splits = split_dataset(coco_data, train_ratio, valid_ratio, test_ratio)
    
    for split_name, split_data in splits.items():
        print(f"   {split_name}: {len(split_data['images'])} 张图片, {len(split_data['annotations'])} 个标注")
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 保存划分后的数据
    print(f"\n💾 保存数据集到: {output_dir}")
    for split_name, split_data in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(exist_ok=True)
        
        # 保存 JSON
        json_path = split_dir / "_annotations.coco.json"
        save_coco_json(split_data, json_path)
        print(f"   ✅ {split_name}/_annotations.coco.json")
        
        # 复制或链接图片
        if copy_images:
            images_output_dir = split_dir
        else:
            images_output_dir = split_dir / "images"
            images_output_dir.mkdir(exist_ok=True)
        
        copied = 0
        for img in split_data['images']:
            filename = img['file_name']
            src = images_dir / filename
            
            if src.exists():
                if copy_images:
                    dst = images_output_dir / filename
                    shutil.copy2(src, dst)
                else:
                    dst = images_output_dir / filename
                    if not dst.exists():
                        os.symlink(src.resolve(), dst)
                copied += 1
        
        print(f"   ✅ {split_name}: 复制了 {copied} 张图片")
    
    print("\n" + "="*60)
    print("✅ 数据集准备完成！")
    print(f"📁 数据集目录: {output_dir}")
    print("="*60)
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="准备 RF-DETR 训练数据集")
    parser.add_argument("--input_json", type=str, required=True,
                       help="Label Studio 导出的 COCO JSON 文件路径")
    parser.add_argument("--images_dir", type=str, required=True,
                       help="图片文件所在目录")
    parser.add_argument("--output_dir", type=str, default="dataset",
                       help="输出数据集目录（默认: dataset）")
    parser.add_argument("--train_ratio", type=float, default=0.7,
                       help="训练集比例（默认: 0.7）")
    parser.add_argument("--valid_ratio", type=float, default=0.2,
                       help="验证集比例（默认: 0.2）")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                       help="测试集比例（默认: 0.1）")
    parser.add_argument("--no_copy", action="store_true",
                       help="不复制图片，使用符号链接（默认: 复制）")
    parser.add_argument("--train_only", action="store_true",
                       help="所有数据用于训练，不划分验证集和测试集")
    
    args = parser.parse_args()
    
    # 验证比例（如果使用 train_only 则跳过）
    if not args.train_only:
        if abs(args.train_ratio + args.valid_ratio + args.test_ratio - 1.0) > 1e-6:
            print("❌ 错误: train_ratio + valid_ratio + test_ratio 必须等于 1.0")
            exit(1)
    
    success = prepare_dataset(
        args.input_json,
        args.images_dir,
        args.output_dir,
        args.train_ratio,
        args.valid_ratio,
        args.test_ratio,
        copy_images=not args.no_copy,
        train_only=args.train_only
    )
    
    exit(0 if success else 1)

