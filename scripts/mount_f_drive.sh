#!/bin/bash
# 挂载Windows F盘到WSL

echo "正在挂载Windows F盘到 /mnt/f ..."

# 创建挂载点
sudo mkdir -p /mnt/f

# 挂载F盘
sudo mount -t drvfs F: /mnt/f

# 检查是否挂载成功
if [ -d "/mnt/f" ]; then
    echo "✅ F盘挂载成功！"
    echo "路径: /mnt/f"
    ls -la /mnt/f/ | head -5
else
    echo "❌ F盘挂载失败"
    echo "请确保："
    echo "1. F盘在Windows中存在"
    echo "2. 有管理员权限"
    echo "3. 或者重启WSL后自动挂载"
fi

