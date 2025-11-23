# 脚本更新日志

## 2025-11-22 - 添加命令行参数支持

### 更新内容

所有主要脚本已更新为支持完整的命令行参数，使用 `argparse` 模块提供更好的用户体验。

### 更新的脚本

1. **train.py** - 训练脚本
   - ✅ 添加完整的命令行参数支持
   - ✅ 支持所有训练参数配置
   - ✅ 支持恢复训练 (`--resume`)
   - ✅ 支持禁用/启用各种功能（TensorBoard、早停、EMA等）

2. **evaluate_model.py** - 模型评估脚本
   - ✅ 添加命令行参数支持
   - ✅ 支持指定检查点、数据集、分割和阈值

3. **test_model.py** - 单张图片测试脚本
   - ✅ 添加命令行参数支持
   - ✅ 支持指定图片、检查点、阈值和输出路径
   - ✅ 支持显示真实标注（通过 `--dataset_dir`）

4. **test_dataset.py** - 批量测试脚本
   - ✅ 添加命令行参数支持
   - ✅ 支持指定数据集目录、检查点和输出目录

5. **analyze_video.py** - 视频分析脚本
   - ✅ 添加命令行参数支持
   - ✅ 支持指定视频、检查点、帧间隔和输出目录

6. **generate_time_table.py** - 时间统计表格生成脚本
   - ✅ 添加命令行参数支持
   - ✅ 支持指定输入JSON和输出文件路径

### 新增文档

- **docs/guides/SCRIPTS_USAGE.md** - 完整的脚本使用指南
  - 所有脚本的参数说明
  - 使用示例
  - 快速参考
  - 常见问题

### 使用方法

所有脚本现在都支持：

1. **查看帮助**：
   ```bash
   python scripts/<script_name>.py --help
   ```

2. **使用命令行参数**：
   ```bash
   python scripts/train.py --epochs 50 --batch_size 16
   python scripts/evaluate_model.py --split all --threshold 0.5
   ```

3. **向后兼容**：
   - 所有脚本保持默认值，可以不传参数直接运行
   - 原有的 `sys.argv` 用法已被替换为 `argparse`

### 改进

- ✅ 统一的参数命名规范
- ✅ 详细的帮助信息（`--help`）
- ✅ 参数验证和错误提示
- ✅ 使用示例（在帮助信息中）
- ✅ 更好的用户体验

### 文档更新

- ✅ `PROJECT_STRUCTURE.md` - 添加脚本使用说明引用
- ✅ `docs/guides/SCRIPTS_USAGE.md` - 完整的使用指南

### 测试

所有脚本的 `--help` 功能已测试通过：
- ✅ train.py
- ✅ evaluate_model.py
- ✅ test_model.py
- ✅ test_dataset.py
- ✅ analyze_video.py
- ✅ generate_time_table.py

---

## 下一步

- [ ] 考虑添加配置文件支持（YAML/JSON）
- [ ] 添加日志级别控制
- [ ] 添加进度条显示
- [ ] 添加更多验证和错误处理

