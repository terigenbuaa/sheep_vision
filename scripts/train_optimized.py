#!/usr/bin/env python
"""
RF-DETR 优化训练脚本 - 针对损失偏高问题
"""

from rfdetr import RFDETRBase
import torch
import os
from pathlib import Path

# ========== 配置参数 ==========
DATASET_DIR = "dataset"  # 数据集根目录（包含 train/valid/test 文件夹）
OUTPUT_DIR = "output_optimized"     # 模型输出目录（使用新目录避免覆盖）
EPOCHS = 100              # 训练轮数
BATCH_SIZE = 8            # 批次大小
GRAD_ACCUM_STEPS = 2      # 梯度累积步数
LEARNING_RATE = 1e-4      # 学习率
LR_ENCODER = 1.5e-4       # 编码器学习率

# ========== 学习率调度器优化 ==========
LR_DROP = 80              # 学习率下降的epoch（原值: 100，太晚了！）
WARMUP_EPOCHS = 1.0       # 预热轮数（原值: 0.0，建议启用预热）

# ========== 损失优化参数 ==========
# 降低框回归损失权重，从5降到3，以平衡分类和回归
BBOX_LOSS_COEF = 3        # 原值: 5 (降低框回归损失权重)
CLS_LOSS_COEF = 2         # 保持不变
GIOU_LOSS_COEF = 2        # 保持不变

# ========== Matcher成本权重优化 ==========
# 与损失权重保持一致，降低框回归的匹配成本
SET_COST_BBOX = 3         # 原值: 5 (降低框回归匹配成本)
SET_COST_CLASS = 2        # 保持不变
SET_COST_GIOU = 2         # 保持不变

# ========== 特征金字塔优化 ==========
# 使用多尺度特征以提升小物体检测能力
PROJECTOR_SCALE = ["P3", "P4", "P5"]  # 原值: ["P4"] (使用多尺度)

# ========== 检查数据集 ==========
def check_dataset(dataset_dir):
    """检查数据集结构"""
    dataset_dir = Path(dataset_dir)
    
    if not dataset_dir.exists():
        print(f"❌ 错误: 数据集目录 '{dataset_dir}' 不存在！")
        return False
    
    train_dir = dataset_dir / "train"
    valid_dir = dataset_dir / "valid"
    
    if not train_dir.exists():
        print(f"❌ 错误: 训练集目录 '{train_dir}' 不存在！")
        return False
    
    train_json = train_dir / "_annotations.coco.json"
    
    if not train_json.exists():
        print(f"❌ 错误: 训练集标注文件 '{train_json}' 不存在！")
        print("   请确保文件名为 '_annotations.coco.json'")
        return False
    
    # 检查图片文件
    train_images = list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.png"))
    
    if len(train_images) == 0:
        print(f"⚠️  警告: 训练集目录中没有找到图片文件")
    
    # 验证集是可选的（如果禁用评估）
    valid_images = []
    if valid_dir.exists():
        valid_images = list(valid_dir.glob("*.jpg")) + list(valid_dir.glob("*.png"))
    
    print(f"✅ 数据集检查通过！")
    print(f"   训练集: {len(train_images)} 张图片")
    if len(valid_images) > 0:
        print(f"   验证集: {len(valid_images)} 张图片")
    else:
        print(f"   验证集: 无（所有数据用于训练）")
    
    return True

# ========== 主函数 ==========
def main():
    print("="*60)
    print("🚀 RF-DETR 优化训练脚本（降低损失配置）")
    print("="*60)
    
    # 检查数据集
    if not check_dataset(DATASET_DIR):
        print("\n💡 提示: 如果数据集格式不正确，请先运行:")
        print("   python prepare_dataset.py --input_json <json文件> --images_dir <图片目录>")
        exit(1)
    
    # 检查 GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"\n✅ 使用 GPU: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("\n⚠️  使用 CPU（训练会很慢，建议使用 GPU）")
    
    # 初始化模型（使用多尺度特征金字塔）
    print("\n🚀 初始化 RF-DETR 模型（优化配置）...")
    try:
        model = RFDETRBase(projector_scale=PROJECTOR_SCALE)
        print("✅ 模型初始化成功")
        print(f"   特征金字塔: {PROJECTOR_SCALE}")
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # 显示训练配置
    print("\n" + "="*60)
    print("📊 训练配置（优化版）:")
    print(f"   数据集路径: {DATASET_DIR}")
    print(f"   输出目录: {OUTPUT_DIR}")
    print(f"   训练轮数: {EPOCHS}")
    print(f"   批次大小: {BATCH_SIZE} × {GRAD_ACCUM_STEPS} = {BATCH_SIZE * GRAD_ACCUM_STEPS}")
    print(f"   学习率: {LEARNING_RATE}")
    print(f"   编码器学习率: {LR_ENCODER}")
    print(f"   设备: {device}")
    print("\n📉 损失权重配置（优化）:")
    print(f"   分类损失权重: {CLS_LOSS_COEF}")
    print(f"   框回归损失权重: {BBOX_LOSS_COEF} (原值: 5)")
    print(f"   GIoU损失权重: {GIOU_LOSS_COEF}")
    print("\n🎯 Matcher成本权重（优化）:")
    print(f"   分类成本: {SET_COST_CLASS}")
    print(f"   框回归成本: {SET_COST_BBOX} (原值: 5)")
    print(f"   GIoU成本: {SET_COST_GIOU}")
    print("\n🔍 特征金字塔:")
    print(f"   使用层级: {PROJECTOR_SCALE}")
    print("\n📈 学习率调度器（优化）:")
    print(f"   学习率下降epoch: {LR_DROP} (原值: 100，太晚了！)")
    print(f"   预热轮数: {WARMUP_EPOCHS} (原值: 0.0)")
    print(f"   学习率变化: Epoch 1-{LR_DROP} = {LEARNING_RATE}, Epoch {LR_DROP+1}-{EPOCHS} = {LEARNING_RATE*0.1}")
    print("="*60 + "\n")
    
    # 开始训练
    try:
        model.train(
            dataset_dir=DATASET_DIR,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            grad_accum_steps=GRAD_ACCUM_STEPS,
            lr=LEARNING_RATE,
            lr_encoder=LR_ENCODER,
            output_dir=OUTPUT_DIR,
            device=device,
            eval=False,                    # eval=True会只评估不训练！
            tensorboard=True,              # 启用 TensorBoard
            early_stopping=True,           # 启用早停
            early_stopping_patience=10,    # 早停耐心值
            checkpoint_interval=10,         # 每10个epoch保存一次
            use_ema=True,                  # 启用EMA
            
            # ========== 损失权重优化 ==========
            cls_loss_coef=CLS_LOSS_COEF,
            bbox_loss_coef=BBOX_LOSS_COEF,  # 从5降低到3
            giou_loss_coef=GIOU_LOSS_COEF,
            
            # ========== Matcher成本权重优化 ==========
            set_cost_class=SET_COST_CLASS,
            set_cost_bbox=SET_COST_BBOX,    # 从5降低到3
            set_cost_giou=SET_COST_GIOU,
            
            # ========== 学习率调度器优化 ==========
            lr_drop=LR_DROP,                # 在第80个epoch下降学习率（原值: 100）
            warmup_epochs=WARMUP_EPOCHS,    # 启用1个epoch预热（原值: 0.0）
        )
        
        print("\n" + "="*60)
        print("✅ 训练完成！")
        print(f"📁 模型保存在: {OUTPUT_DIR}/")
        print(f"   最佳模型: {OUTPUT_DIR}/checkpoint_best_total.pth")
        print("\n💡 使用训练好的模型:")
        print(f"   from rfdetr import RFDETRBase")
        print(f"   model = RFDETRBase(pretrain_weights='{OUTPUT_DIR}/checkpoint_best_total.pth')")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
        print(f"   检查点已保存到: {OUTPUT_DIR}/checkpoint.pth")
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()

