"""
工具函数测试
"""
import pytest
from pathlib import Path

def test_coco_classes_import():
    """测试COCO类别导入"""
    from rfdetr.util.coco_classes import COCO_CLASSES
    assert COCO_CLASSES is not None
    assert isinstance(COCO_CLASSES, (dict, list))
    assert len(COCO_CLASSES) > 0

def test_config_imports():
    """测试配置模块导入"""
    from rfdetr.config import (
        RFDETRBaseConfig,
        RFDETRNanoConfig,
        RFDETRSmallConfig,
        RFDETRMediumConfig,
        ModelConfig
    )
    assert RFDETRBaseConfig is not None
    assert RFDETRNanoConfig is not None
    assert RFDETRSmallConfig is not None
    assert RFDETRMediumConfig is not None
    assert ModelConfig is not None
