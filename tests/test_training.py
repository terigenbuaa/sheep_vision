"""
训练相关测试
"""
import pytest
from pathlib import Path

def test_train_script_exists():
    """测试训练脚本存在"""
    train_script = Path("train.py")
    assert train_script.exists()

def test_train_script_imports():
    """测试训练脚本可以导入"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    # 检查是否可以导入主要模块
    from rfdetr import RFDETRBase
    assert RFDETRBase is not None

@pytest.mark.skipif(not Path("dataset/train").exists(), 
                    reason="需要训练数据集")
def test_dataset_structure():
    """测试数据集结构"""
    dataset_dir = Path("dataset")
    
    if dataset_dir.exists():
        train_dir = dataset_dir / "train"
        valid_dir = dataset_dir / "valid"
        
        # 检查训练集
        if train_dir.exists():
            train_json = train_dir / "_annotations.coco.json"
            assert train_json.exists(), "训练集应该包含 _annotations.coco.json"
            
            # 检查是否有图片文件
            images = list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.png"))
            assert len(images) > 0, "训练集应该包含图片文件"

def test_model_train_method():
    """测试模型的train方法存在"""
    from rfdetr import RFDETRBase
    
    model = RFDETRBase()
    assert hasattr(model, 'train')
    assert callable(model.train)
