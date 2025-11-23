"""
基础模型测试 - 测试模型初始化和基本功能
"""
import pytest
import torch
import numpy as np
from PIL import Image
from pathlib import Path

def test_model_import():
    """测试模型导入"""
    from rfdetr import RFDETRBase, RFDETRNano, RFDETRSmall, RFDETRMedium
    assert RFDETRBase is not None
    assert RFDETRNano is not None
    assert RFDETRSmall is not None
    assert RFDETRMedium is not None

def test_model_initialization():
    """测试模型初始化"""
    from rfdetr import RFDETRBase
    
    model = RFDETRBase()
    assert model is not None
    assert hasattr(model, 'model')
    assert hasattr(model, 'predict')

def test_model_with_pretrained_weights():
    """测试使用预训练权重初始化模型"""
    from rfdetr import RFDETRBase
    
    # 使用默认预训练权重
    model = RFDETRBase()
    assert model is not None

@pytest.mark.skipif(not Path("rf-detr-base.pth").exists(), reason="需要预训练权重文件")
def test_model_with_local_checkpoint():
    """测试使用本地checkpoint初始化模型"""
    from rfdetr import RFDETRBase
    
    checkpoint_path = "rf-detr-base.pth"
    if Path(checkpoint_path).exists():
        model = RFDETRBase(pretrain_weights=checkpoint_path)
        assert model is not None

def test_model_class_names():
    """测试模型类别名称"""
    from rfdetr import RFDETRBase
    
    model = RFDETRBase()
    class_names = model.class_names
    assert class_names is not None
    assert isinstance(class_names, dict) or isinstance(class_names, list)

def test_model_predict_single_image(sample_image):
    """测试单张图片预测"""
    from rfdetr import RFDETRBase
    
    model = RFDETRBase()
    detections = model.predict(sample_image, threshold=0.5)
    
    assert detections is not None
    assert hasattr(detections, 'xyxy')  # supervision Detections对象
    assert hasattr(detections, 'confidence')
    assert hasattr(detections, 'class_id')

def test_model_predict_image_path(sample_image_path):
    """测试使用图片路径进行预测"""
    from rfdetr import RFDETRBase
    
    model = RFDETRBase()
    detections = model.predict(str(sample_image_path), threshold=0.5)
    
    assert detections is not None
    assert hasattr(detections, 'xyxy')

def test_model_predict_numpy_array(sample_image):
    """测试使用numpy数组进行预测"""
    from rfdetr import RFDETRBase
    import numpy as np
    
    model = RFDETRBase()
    img_array = np.array(sample_image)
    detections = model.predict(img_array, threshold=0.5)
    
    assert detections is not None

def test_model_predict_batch(sample_image):
    """测试批量预测"""
    from rfdetr import RFDETRBase
    
    model = RFDETRBase()
    images = [sample_image, sample_image]
    detections_list = model.predict(images, threshold=0.5)
    
    assert isinstance(detections_list, list)
    assert len(detections_list) == 2
    for detections in detections_list:
        assert detections is not None

def test_model_predict_threshold():
    """测试不同阈值的影响"""
    from rfdetr import RFDETRBase
    import numpy as np
    from PIL import Image
    
    model = RFDETRBase()
    img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    image = Image.fromarray(img_array)
    
    # 低阈值应该检测到更多目标
    detections_low = model.predict(image, threshold=0.1)
    detections_high = model.predict(image, threshold=0.9)
    
    assert len(detections_low) >= len(detections_high)

def test_model_optimize_for_inference():
    """测试推理优化"""
    from rfdetr import RFDETRBase
    
    model = RFDETRBase()
    model.optimize_for_inference(compile=False)
    
    assert model._is_optimized_for_inference == True
    assert model.model.inference_model is not None
