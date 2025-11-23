#!/usr/bin/env python
"""
解压并合并多个Roboflow数据集zip文件，统一整理成RF-DETR训练格式
"""
import zipfile
import json
import shutil
from pathlib import Path
from collections import defaultdict
import random
import tempfile

def extract_zip(zip_path, extract_to):
    """解压zip文件"""
    print(f"   📦 解压: {Path(zip_path).name}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        return True
    except Exception as e:
        print(f"   ❌ 解压失败: {e}")
        return False

def find_coco_json(directory):
    """在目录中查找COCO格式的JSON文件"""
    directory = Path(directory)
    
    # 常见的JSON文件名
    possible_names = [
        "_annotations.coco.json",
        "annotations.coco.json",
        "_annotations.json",
        "annotations.json",
        "coco.json",
    ]
    
    # 先查找常见名称
    for name in possible_names:
        json_path = directory / name
        if json_path.exists():
            return json_path
    
    # 查找所有JSON文件
    json_files = list(directory.glob("*.json"))
    if len(json_files) == 1:
        return json_files[0]
    elif len(json_files) > 1:
        # 如果有多个，优先选择包含"annotation"或"coco"的
        for json_file in json_files:
            if "annotation" in json_file.name.lower() or "coco" in json_file.name.lower():
                return json_file
        return json_files[0]  # 返回第一个
    
    return None

def load_coco_json(json_path):
    """加载COCO JSON文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def merge_coco_datasets(coco_datasets, start_image_id=1, start_ann_id=1):
    """
    合并多个COCO格式的数据集
    
    Args:
        coco_datasets: COCO数据集的列表
        start_image_id: 起始图片ID
        start_ann_id: 起始标注ID
    
    Returns:
        合并后的COCO数据集
    """
    merged = {
        'images': [],
        'annotations': [],
        'categories': [],
        'info': {}
    }
    
    # 合并类别（去重）
    category_map = {}  # name -> id映射
    next_cat_id = 1
    
    current_image_id = start_image_id
    current_ann_id = start_ann_id
    
    for idx, coco_data in enumerate(coco_datasets):
        print(f"   📊 处理数据集 {idx + 1}/{len(coco_datasets)}")
        
        # 处理类别
        for cat in coco_data.get('categories', []):
            cat_name = cat['name']
            if cat_name not in category_map:
                category_map[cat_name] = next_cat_id
                merged['categories'].append({
                    'id': next_cat_id,
                    'name': cat_name,
                    'supercategory': cat.get('supercategory', '')
                })
                next_cat_id += 1
        
        # 创建旧类别ID到新类别ID的映射
        old_to_new_cat_id = {}
        for cat in coco_data.get('categories', []):
            old_id = cat['id']
            new_id = category_map[cat['name']]
            old_to_new_cat_id[old_id] = new_id
        
        # 处理图片
        old_to_new_image_id = {}
        for img in coco_data.get('images', []):
            old_id = img['id']
            new_id = current_image_id
            old_to_new_image_id[old_id] = new_id
            
            merged['images'].append({
                'id': new_id,
                'file_name': img['file_name'],
                'width': img.get('width', 0),
                'height': img.get('height', 0)
            })
            current_image_id += 1
        
        # 处理标注
        for ann in coco_data.get('annotations', []):
            old_image_id = ann['image_id']
            old_cat_id = ann['category_id']
            
            if old_image_id in old_to_new_image_id and old_cat_id in old_to_new_cat_id:
                merged['annotations'].append({
                    'id': current_ann_id,
                    'image_id': old_to_new_image_id[old_image_id],
                    'category_id': old_to_new_cat_id[old_cat_id],
                    'bbox': ann.get('bbox', []),
                    'area': ann.get('area', 0),
                    'iscrowd': ann.get('iscrowd', 0),
                    'segmentation': ann.get('segmentation', [])
                })
                current_ann_id += 1
    
    return merged

def split_dataset(coco_data, train_ratio=0.7, valid_ratio=0.2, test_ratio=0.1, seed=42):
    """划分数据集为train/valid/test"""
    random.seed(seed)
    
    image_ids = [img['id'] for img in coco_data['images']]
    random.shuffle(image_ids)
    
    total = len(image_ids)
    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * valid_ratio)
    
    train_ids = set(image_ids[:train_end])
    valid_ids = set(image_ids[train_end:valid_end])
    test_ids = set(image_ids[valid_end:])
    
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
    
    # 保持categories和info
    for split_name in splits:
        splits[split_name]['categories'] = coco_data.get('categories', [])
        splits[split_name]['info'] = coco_data.get('info', {})
    
    return splits

def main():
    print("=" * 70)
    print("📦 解压并合并数据集")
    print("=" * 70)
    
    dataset_dir = Path("dataset")
    zip_files = list(dataset_dir.glob("project-*.zip"))
    
    if len(zip_files) == 0:
        print("❌ 错误: 在dataset目录中没有找到project-*.zip文件")
        return
    
    print(f"\n📋 找到 {len(zip_files)} 个数据集zip文件:")
    for zip_file in zip_files:
        print(f"   - {zip_file.name}")
    
    # 创建临时目录用于解压
    temp_dir = Path(tempfile.mkdtemp(prefix="rfdetr_merge_"))
    print(f"\n📁 临时解压目录: {temp_dir}")
    
    # 解压所有zip文件
    print(f"\n🔓 开始解压...")
    extracted_dirs = []
    coco_datasets = []
    all_images_dir = temp_dir / "all_images"
    all_images_dir.mkdir(exist_ok=True)
    
    for zip_file in zip_files:
        extract_dir = temp_dir / zip_file.stem
        extract_dir.mkdir(exist_ok=True)
        
        if extract_zip(zip_file, extract_dir):
            extracted_dirs.append(extract_dir)
            
            # 查找COCO JSON文件
            json_path = find_coco_json(extract_dir)
            if json_path:
                print(f"   ✅ 找到标注文件: {json_path.name}")
                
                # 加载COCO数据
                coco_data = load_coco_json(json_path)
                
                # 查找图片目录
                images_dir = None
                for possible_dir in ['train', 'valid', 'test', 'images', '']:
                    test_dir = extract_dir / possible_dir
                    if test_dir.exists() and test_dir.is_dir():
                        # 检查是否有图片文件
                        if list(test_dir.glob("*.jpg")) or list(test_dir.glob("*.png")):
                            images_dir = test_dir
                            break
                
                if not images_dir:
                    # 在整个目录中查找图片
                    for img_file in extract_dir.rglob("*.jpg"):
                        images_dir = img_file.parent
                        break
                    if not images_dir:
                        for img_file in extract_dir.rglob("*.png"):
                            images_dir = img_file.parent
                            break
                
                if images_dir:
                    print(f"   📸 图片目录: {images_dir.relative_to(extract_dir)}")
                    
                    # 复制图片到统一目录（避免文件名冲突）
                    valid_images = []
                    for img_info in coco_data.get('images', []):
                        # 尝试多个可能的路径
                        possible_srcs = [
                            images_dir / img_info['file_name'],
                            images_dir / Path(img_info['file_name']).name,  # 只使用文件名
                            extract_dir / img_info['file_name'],
                            extract_dir / Path(img_info['file_name']).name,
                        ]
                        
                        src = None
                        for possible_src in possible_srcs:
                            if possible_src.exists():
                                src = possible_src
                                break
                        
                        if src and src.exists():
                            # 使用项目名作为前缀避免冲突
                            prefix = zip_file.stem[:20]  # 使用前20个字符
                            original_name = Path(img_info['file_name']).name
                            new_name = f"{prefix}_{original_name}"
                            dst = all_images_dir / new_name
                            shutil.copy2(src, dst)
                            # 更新JSON中的文件名
                            img_info['file_name'] = new_name
                            valid_images.append(img_info)
                        else:
                            # 尝试在整个extract_dir中查找
                            filename = Path(img_info['file_name']).name
                            found_file = None
                            for img_file in extract_dir.rglob(filename):
                                found_file = img_file
                                break
                            
                            if found_file:
                                prefix = zip_file.stem[:20]
                                new_name = f"{prefix}_{filename}"
                                dst = all_images_dir / new_name
                                shutil.copy2(found_file, dst)
                                img_info['file_name'] = new_name
                                valid_images.append(img_info)
                    
                    # 只保留有效图片的数据
                    if valid_images:
                        coco_data['images'] = valid_images
                        # 更新标注，只保留有效图片的标注
                        valid_image_ids = {img['id'] for img in valid_images}
                        coco_data['annotations'] = [
                            ann for ann in coco_data.get('annotations', [])
                            if ann.get('image_id') in valid_image_ids
                        ]
                        coco_datasets.append(coco_data)
                        print(f"   ✅ 处理了 {len(valid_images)} 张图片")
                    else:
                        print(f"   ⚠️  警告: 未找到任何有效图片")
                else:
                    print(f"   ⚠️  警告: 未找到图片目录")
            else:
                print(f"   ⚠️  警告: 未找到COCO标注文件")
    
    if len(coco_datasets) == 0:
        print("\n❌ 错误: 没有成功加载任何数据集")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return
    
    print(f"\n✅ 成功加载 {len(coco_datasets)} 个数据集")
    
    # 合并数据集
    print(f"\n🔄 合并数据集...")
    merged_coco = merge_coco_datasets(coco_datasets)
    
    print(f"\n📊 合并后的统计:")
    print(f"   图片总数: {len(merged_coco['images'])}")
    print(f"   标注总数: {len(merged_coco['annotations'])}")
    print(f"   类别数量: {len(merged_coco['categories'])}")
    print(f"   类别列表: {[cat['name'] for cat in merged_coco['categories']]}")
    
    # 划分数据集
    print(f"\n📊 划分数据集 (训练:70%, 验证:20%, 测试:10%)...")
    splits = split_dataset(merged_coco, train_ratio=0.7, valid_ratio=0.2, test_ratio=0.1)
    
    for split_name, split_data in splits.items():
        print(f"   {split_name}: {len(split_data['images'])} 张图片, {len(split_data['annotations'])} 个标注")
    
    # 保存到dataset目录
    output_dir = dataset_dir
    print(f"\n💾 保存数据集到: {output_dir}")
    
    for split_name, split_data in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(exist_ok=True)
        
        # 保存JSON
        json_path = split_dir / "_annotations.coco.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        print(f"   ✅ {split_name}/_annotations.coco.json")
        
        # 复制图片（从all_images_dir复制到对应的split目录）
        copied = 0
        for img in split_data['images']:
            filename = img['file_name']
            src = all_images_dir / filename
            if src.exists():
                dst = split_dir / filename
                if not dst.exists():  # 避免重复复制
                    shutil.copy2(src, dst)
                copied += 1
            else:
                print(f"   ⚠️  警告: 图片文件不存在: {filename}")
        
        print(f"   ✅ {split_name}: 复制了 {copied}/{len(split_data['images'])} 张图片")
    
    # 清理临时目录
    print(f"\n🧹 清理临时文件...")
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("\n" + "=" * 70)
    print("✅ 数据集合并完成！")
    print(f"📁 数据集目录: {output_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()
