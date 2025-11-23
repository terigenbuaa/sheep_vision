# WSL中访问Windows文件系统

## 访问Windows驱动器

在WSL中，Windows驱动器会自动挂载到 `/mnt/` 目录下：

- C盘 → `/mnt/c/`
- D盘 → `/mnt/d/`
- E盘 → `/mnt/e/`
- F盘 → `/mnt/f/` (可能需要手动挂载)

## 挂载F盘

如果F盘没有自动挂载，可以手动挂载：

```bash
# 方法1: 使用提供的脚本
./mount_f_drive.sh

# 方法2: 手动挂载
sudo mkdir -p /mnt/f
sudo mount -t drvfs F: /mnt/f
```

## 路径转换

Windows路径 → WSL路径：

```
F:\25.8.13\extracted_frames\12-全景_frames
↓
/mnt/f/25.8.13/extracted_frames/12-全景_frames
```

## 使用批量测试脚本

```bash
# 测试Windows路径下的图片
python test_dataset.py /mnt/f/25.8.13/extracted_frames/12-全景_frames

# 或者使用相对路径（如果在项目目录中）
python test_dataset.py /mnt/f/25.8.13/extracted_frames/12-全景_frames
```

## 注意事项

1. **路径中的中文**: WSL支持中文路径，但确保终端编码正确
2. **权限问题**: Windows文件在WSL中可能有权限限制
3. **性能**: 跨文件系统访问可能比Linux原生文件系统慢
4. **自动挂载**: 重启WSL后，已挂载的驱动器可能需要重新挂载

## 检查挂载状态

```bash
# 查看所有挂载的Windows驱动器
ls -la /mnt/

# 检查F盘是否挂载
ls /mnt/f/ 2>&1
```

