#!/usr/bin/env python
"""
RF-DETR 训练脚本
"""

from rfdetr import RFDETRBase
import torch
import os
import argparse
from pathlib import Path

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="RF-DETR 训练脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认参数训练
  python train.py

  # 指定数据集和输出目录
  python train.py --dataset_dir dataset --output_dir output

  # 自定义训练参数
  python train.py --epochs 50 --batch_size 16 --lr 2e-4

  # 使用预训练模型继续训练
  python train.py --resume checkpoints/finetuned/default_model/checkpoints/latest.pth
        """
    )
    
    # 数据集和输出
    parser.add_argument("--dataset_dir", type=str, default="dataset",
                       help="数据集根目录（包含 train/valid/test 文件夹，默认: dataset）")
    parser.add_argument("--output_dir", type=str, default="output",
                       help="模型输出目录（默认: output）")
    parser.add_argument("--resume", type=str, default=None,
                       help="恢复训练的检查点路径（可选）")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=100,
                       help="训练轮数（默认: 100）")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="批次大小（默认: 8）")
    parser.add_argument("--grad_accum_steps", type=int, default=2,
                       help="梯度累积步数（默认: 2，总batch = batch_size × grad_accum_steps）")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="学习率（默认: 1e-4）")
    parser.add_argument("--lr_encoder", type=float, default=1.5e-4,
                       help="编码器学习率（默认: 1.5e-4）")
    
    # 训练选项
    parser.add_argument("--no_tensorboard", action="store_true",
                       help="禁用 TensorBoard 日志")
    parser.add_argument("--no_early_stopping", action="store_true",
                       help="禁用早停机制")
    parser.add_argument("--early_stopping_patience", type=int, default=10,
                       help="早停耐心值（默认: 10）")
    parser.add_argument("--checkpoint_interval", type=int, default=10,
                       help="检查点保存间隔（epoch数，默认: 10）")
    parser.add_argument("--no_ema", action="store_true",
                       help="禁用 EMA（指数移动平均）")
    parser.add_argument("--eval", action="store_true",
                       help="仅评估模式（不训练）")
    
    return parser.parse_args()

# ========== 配置参数（保留默认值，可通过命令行覆盖） ==========

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
    args = parse_args()
    
    # 使用命令行参数或默认值
    DATASET_DIR = args.dataset_dir
    OUTPUT_DIR = args.output_dir
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    GRAD_ACCUM_STEPS = args.grad_accum_steps
    LEARNING_RATE = args.lr
    LR_ENCODER = args.lr_encoder
    
    print("="*60)
    print("🚀 RF-DETR 训练脚本")
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
    
    # 初始化模型
    print("\n🚀 初始化 RF-DETR 模型...")
    try:
        model = RFDETRBase()
        print("✅ 模型初始化成功")
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # 显示训练配置
    print("\n" + "="*60)
    print("📊 训练配置:")
    print(f"   数据集路径: {DATASET_DIR}")
    print(f"   输出目录: {OUTPUT_DIR}")
    print(f"   训练轮数: {EPOCHS}")
    print(f"   批次大小: {BATCH_SIZE} × {GRAD_ACCUM_STEPS} = {BATCH_SIZE * GRAD_ACCUM_STEPS}")
    print(f"   学习率: {LEARNING_RATE}")
    print(f"   编码器学习率: {LR_ENCODER}")
    print(f"   设备: {device}")
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
            eval=args.eval,                          # 评估模式
            tensorboard=not args.no_tensorboard,      # TensorBoard
            early_stopping=not args.no_early_stopping, # 早停
            early_stopping_patience=args.early_stopping_patience,
            checkpoint_interval=args.checkpoint_interval,
            use_ema=not args.no_ema,                 # EMA
            resume=args.resume,                       # 恢复训练
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

