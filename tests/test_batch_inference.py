"""
批量推理脚本测试
"""
import pytest
import json
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np

def test_batch_inference_imports():
    """测试批量推理脚本的导入"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from batch_inference import (
        load_trained_model,
        process_image,
        save_annotated_image
    )
    assert load_trained_model is not None
    assert process_image is not None
    assert save_annotated_image is not None

@pytest.mark.skipif(not Path("output/checkpoint_best_total.pth").exists(), 
                    reason="需要训练好的checkpoint")
def test_load_trained_model():
    """测试加载训练好的模型"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from batch_inference import load_trained_model
    
    checkpoint_path = "output/checkpoint_best_total.pth"
    if Path(checkpoint_path).exists():
        model = load_trained_model(checkpoint_path)
        assert model is not None
        assert hasattr(model, 'predict')

def test_process_image(sample_image_path):
    """测试处理单张图片"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from batch_inference import process_image
    from rfdetr import RFDETRBase
    
    model = RFDETRBase()
    result = process_image(model, sample_image_path, threshold=0.3)
    
    assert result is not None
    assert 'success' in result
    assert 'image_name' in result
    assert 'image_path' in result
    
    if result['success']:
        assert 'class_counts' in result
        assert 'total_detections' in result
        assert isinstance(result['class_counts'], dict)
        assert isinstance(result['total_detections'], int)

def test_process_image_error_handling(temp_dir):
    """测试错误处理"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from batch_inference import process_image
    from rfdetr import RFDETRBase
    
    model = RFDETRBase()
    # 使用不存在的图片路径
    fake_path = temp_dir / "nonexistent.jpg"
    result = process_image(model, fake_path, threshold=0.3)
    
    assert result is not None
    assert result['success'] == False
    assert 'error' in result

def test_save_annotated_image(sample_image_path, temp_dir):
    """测试保存标注图片"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from batch_inference import save_annotated_image
    from rfdetr import RFDETRBase
    
    model = RFDETRBase()
    output_path = temp_dir / "annotated_test.jpg"
    
    success = save_annotated_image(model, sample_image_path, output_path, threshold=0.3)
    
    assert success == True
    assert output_path.exists()

def test_batch_inference_main_with_test_images():
    """测试使用当前目录的测试图片运行批量推理"""
    import sys
    import subprocess
    from pathlib import Path
    
    # 检查是否有测试图片
    test_images = list(Path(".").glob("*.jpg")) + list(Path(".").glob("*.png"))
    
    if not test_images:
        pytest.skip("当前目录没有测试图片")
    
    # 检查是否有checkpoint
    checkpoint = Path("output/checkpoint_best_total.pth")
    if not checkpoint.exists():
        pytest.skip("需要训练好的checkpoint")
    
    # 运行批量推理脚本（使用当前目录）
    script_path = Path(__file__).parent.parent / "batch_inference.py"
    result = subprocess.run(
        ["python", str(script_path), "--input_dir", ".", "--output_dir", "output/test_batch_results"],
        capture_output=True,
        text=True,
        timeout=300  # 5分钟超时
    )
    
    # 检查输出目录是否创建
    output_dir = Path("output/test_batch_results")
    if output_dir.exists():
        # 检查是否有结果文件
        json_path = output_dir / "results.json"
        if json_path.exists():
            with open(json_path, 'r') as f:
                results = json.load(f)
                assert 'summary' in results
                assert 'results' in results
