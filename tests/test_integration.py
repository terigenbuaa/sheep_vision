"""
集成测试 - 测试完整的工作流程
"""
import pytest
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np

@pytest.fixture
def test_images_dir(temp_dir):
    """创建包含测试图片的目录"""
    # 创建几张测试图片
    for i in range(3):
        img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(temp_dir / f"test_{i}.jpg")
    return temp_dir

def test_end_to_end_inference(test_images_dir, temp_dir):
    """端到端推理测试"""
    from rfdetr import RFDETRBase
    import supervision as sv
    
    # 初始化模型
    model = RFDETRBase()
    
    # 处理目录中的所有图片
    image_files = list(test_images_dir.glob("*.jpg"))
    assert len(image_files) > 0
    
    results = []
    for img_path in image_files:
        image = Image.open(img_path).convert('RGB')
        detections = model.predict(image, threshold=0.5)
        
        assert detections is not None
        results.append({
            'image': img_path.name,
            'detections': len(detections)
        })
    
    assert len(results) == len(image_files)

def test_model_with_different_input_formats():
    """测试模型接受不同输入格式"""
    from rfdetr import RFDETRBase
    import numpy as np
    from PIL import Image
    
    model = RFDETRBase()
    
    # 创建测试图片
    img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 测试PIL Image
    pil_image = Image.fromarray(img_array)
    detections1 = model.predict(pil_image, threshold=0.5)
    assert detections1 is not None
    
    # 测试numpy array
    detections2 = model.predict(img_array, threshold=0.5)
    assert detections2 is not None
    
    # 测试文件路径（需要先保存）
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        pil_image.save(tmp.name)
        detections3 = model.predict(tmp.name, threshold=0.5)
        assert detections3 is not None
        Path(tmp.name).unlink()

def test_batch_processing():
    """测试批量处理"""
    from rfdetr import RFDETRBase
    import numpy as np
    from PIL import Image
    
    model = RFDETRBase()
    
    # 创建多张图片
    images = []
    for i in range(3):
        img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        images.append(Image.fromarray(img_array))
    
    # 批量预测
    detections_list = model.predict(images, threshold=0.5)
    
    assert isinstance(detections_list, list)
    assert len(detections_list) == len(images)
    
    for detections in detections_list:
        assert detections is not None

def test_visualization_pipeline():
    """测试可视化流程"""
    from rfdetr import RFDETRBase
    import supervision as sv
    import numpy as np
    from PIL import Image
    
    model = RFDETRBase()
    
    # 创建测试图片
    img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    image = Image.fromarray(img_array)
    
    # 预测
    detections = model.predict(image, threshold=0.5)
    
    # 可视化
    annotated_image = image.copy()
    annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
    
    assert annotated_image is not None
    assert annotated_image.size == image.size
