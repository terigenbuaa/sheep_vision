"""
Pytest configuration and fixtures for RF-DETR tests
"""
import pytest
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import tempfile
import shutil

@pytest.fixture
def device():
    """返回可用的设备（CPU或CUDA）"""
    return "cuda" if torch.cuda.is_available() else "cpu"

@pytest.fixture
def temp_dir():
    """创建临时目录用于测试"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)

@pytest.fixture
def sample_image():
    """创建示例图片用于测试"""
    # 创建一个简单的RGB图片
    img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return Image.fromarray(img_array)

@pytest.fixture
def sample_image_path(temp_dir, sample_image):
    """保存示例图片到临时路径"""
    img_path = temp_dir / "test_image.jpg"
    sample_image.save(img_path)
    return img_path

@pytest.fixture
def pretrained_model_available():
    """检查预训练模型是否可用"""
    try:
        from rfdetr import RFDETRBase
        model = RFDETRBase()
        return True
    except Exception:
        return False

@pytest.fixture
def checkpoint_path():
    """返回checkpoint路径（如果存在）"""
    checkpoint = Path("output/checkpoint_best_total.pth")
    if checkpoint.exists():
        return str(checkpoint)
    return None
