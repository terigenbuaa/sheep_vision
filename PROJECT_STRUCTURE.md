# 项目目录结构说明

本文档说明项目的目录组织结构，便于后续扩展和管理。

## 目录结构

```
rf-detr/
├── README.md                    # 项目主README
├── QUICK_START.md              # 快速上手指南（推荐新手阅读）
├── CONTRIBUTING.md              # 贡献指南
├── PROJECT_STRUCTURE.md         # 本文件：目录结构说明
│
├── scripts/                     # 脚本文件目录
│   ├── train.py                 # 训练脚本
│   ├── train_optimized.py       # 优化训练脚本
│   ├── evaluate_model.py        # 模型评估脚本
│   ├── analyze_video.py         # 视频分析脚本
│   ├── test_model.py            # 模型测试脚本
│   ├── test_dataset.py          # 数据集测试脚本
│   ├── prepare_dataset.py       # 数据集准备脚本
│   ├── merge_datasets.py        # 数据集合并脚本
│   ├── generate_time_table.py   # 时间表生成脚本
│   ├── quick_test.py            # 快速测试脚本
│   ├── start_training.sh        # 训练启动脚本
│   ├── batch_analyze_folders.sh # 批量分析脚本
│   ├── quick_start.sh           # 快速启动脚本
│   └── mount_f_drive.sh         # 挂载脚本
│
├── checkpoints/                 # 模型检查点目录
│   ├── pretrained/              # 预训练模型
│   │   └── rf-detr-base.pth     # 基础预训练模型
│   └── finetuned/               # 微调后的模型（支持多个模型）
│       ├── README.md            # 微调模型管理说明
│       ├── default_model/       # 默认微调模型（当前模型）
│       │   ├── README.md        # 模型说明文档
│       │   ├── best/            # 最佳模型
│       │   │   ├── checkpoint_best_ema.pth
│       │   │   ├── checkpoint_best_regular.pth
│       │   │   └── checkpoint_best_total.pth
│       │   └── checkpoints/     # 检查点文件
│       │       └── latest.pth
│       ├── model_v2/            # 示例：第二个模型（可按需添加）
│       ├── task_detection/      # 示例：特定任务的模型
│       └── 2025-11-22/          # 示例：按日期组织的模型
│
├── data/                        # 数据文件目录
│   ├── test/                    # 测试数据
│   │   ├── images/             # 测试图片
│   │   └── videos/             # 测试视频
│   ├── train/                   # 训练数据（可选）
│   │   ├── images/             # 训练图片
│   │   └── videos/             # 训练视频
│   ├── raw/                     # 原始数据（备份）
│   └── processed/               # 处理后的数据
│
├── dataset/                     # 训练数据集（COCO格式）
│   ├── train/
│   ├── valid/
│   └── test/
│
├── output/                      # 训练输出和分析结果
│   ├── checkpoint*.pth          # 训练检查点
│   ├── eval/                    # 评估结果
│   └── analysis_*/              # 分析结果目录
│
├── docs/                        # 文档目录
│   ├── guides/                  # 使用指南
│   │   ├── SCRIPTS_USAGE.md     # 脚本使用指南（命令行参数说明）
│   │   ├── ANALYSIS_COMMANDS.md
│   │   ├── DATASET_TRAINING_GUIDE.md
│   │   ├── TEST_AND_EVALUATE.md
│   │   ├── TRAINING_METRICS_EXPLAINED.md
│   │   ├── TRAINING_TROUBLESHOOTING.md
│   │   ├── README_WSL_WINDOWS.md
│   │   └── IMPROVEMENT_ROADMAP.md
│   └── ...                      # 其他文档（API参考等）
│
├── rfdetr/                      # 核心代码包
├── tests/                       # 测试代码
└── ...                          # 其他配置文件

```

## 目录用途说明

### scripts/
存放所有可执行的Python和Shell脚本。便于统一管理和调用。

### checkpoints/
- `pretrained/`: 存放预训练模型权重文件
- `finetuned/`: 存放微调后的模型（便于后续扩展多个微调模型）

### data/
- `raw/`: 存放原始数据文件（视频、图片等）
- `processed/`: 存放处理后的数据文件

### output/
训练和评估的输出目录，包括：
- 训练过程中的检查点文件
- 评估结果
- 分析结果（视频分析、图像分析等）

### docs/guides/
存放项目使用指南和教程文档。

## 快速开始

**新手入门**：请先阅读 [QUICK_START.md](QUICK_START.md) 快速上手指南。

## 使用建议

1. **添加新的微调模型**：
   - **方法1（推荐）**：使用脚本自动保存
     ```bash
     ./scripts/save_finetuned_model.sh <模型名称>
     ```
   - **方法2**：手动创建目录结构
     ```bash
     mkdir -p checkpoints/finetuned/<模型名称>/{best,checkpoints}
     # 然后复制模型文件并创建README.md
     ```
   - 命名建议：`model_v2`, `task_detection`, `2025-11-22` 等
   - 参考 `checkpoints/finetuned/README.md` 了解详细说明

2. **添加新的数据**：
   - 原始数据放入 `data/raw/`，处理后的数据放入 `data/processed/`
   - 训练数据集放入 `dataset/` 目录（COCO格式）

3. **添加新的脚本**：将脚本文件放入 `scripts/` 目录

4. **添加新的文档**：将指南类文档放入 `docs/guides/` 目录

5. **使用脚本**：
   - 所有脚本都支持命令行参数，使用 `--help` 查看帮助
   - 详细使用说明请参考 `docs/guides/SCRIPTS_USAGE.md`

5. **使用微调模型**：
   - 推荐使用 `checkpoints/finetuned/best/checkpoint_best_ema.pth`
   - 查看 `checkpoints/finetuned/README.md` 了解详细性能指标

## 注意事项

- 运行脚本时，注意路径可能需要调整（从根目录运行）
- `output/` 目录中的文件可能会被覆盖，重要结果请及时备份
- 大文件（模型、数据）建议使用 `.gitignore` 排除，避免提交到版本控制

