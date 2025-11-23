# RF-DETR 快速上手指南（中文）

> 📖 **完整文档**: 查看 [QUICK_START.md](QUICK_START.md) 获取详细的中文快速上手指南

## 🚀 5分钟快速开始

### 1. 安装

```bash
pip install rfdetr
```

### 2. 快速测试

```bash
# 使用预训练模型快速测试
python scripts/quick_test.py
```

### 3. 训练自己的模型

```bash
# 准备数据集
python scripts/prepare_dataset.py --input_json annotations.json --images_dir images/

# 开始训练
python scripts/train.py

# 评估模型
python scripts/evaluate_model.py
```

## 📚 更多资源

- **快速上手指南**: [QUICK_START.md](QUICK_START.md) - 完整的中文快速入门教程
- **脚本使用说明**: [docs/guides/SCRIPTS_USAGE.md](docs/guides/SCRIPTS_USAGE.md) - 所有脚本的命令行参数
- **项目结构说明**: [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - 目录结构详解
- **微调模型管理**: [checkpoints/finetuned/README.md](checkpoints/finetuned/README.md) - 如何保存和管理多个模型

## 💡 常用命令

```bash
# 查看帮助
python scripts/train.py --help

# 训练模型
python scripts/train.py --epochs 100 --batch_size 8

# 测试图片
python scripts/test_model.py --image data/test.jpg

# 保存微调模型
./scripts/save_finetuned_model.sh my_model
```

---

**详细说明请查看 [QUICK_START.md](QUICK_START.md)**

