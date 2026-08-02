"""
conftest.py —— pytest共享配置

把 app/ 目录加入 sys.path，让测试可以直接 `import detector`，
不需要把 deployment/ 变成一个正式的Python包。
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# tests/ 和 app/ 是兄弟目录，这里把 app/ 加进路径
APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))


@pytest.fixture
def fake_weights(tmp_path):
    """
    创建一个假的.pt文件，只是为了让 os.path.exists() 检查通过。
    文件内容不重要，因为真正的模型加载会被mock掉，不会真的去解析这个文件。
    """
    weight_file = tmp_path / "best.pt"
    weight_file.write_text("fake weights, content doesn't matter")
    return str(weight_file)


@pytest.fixture
def mock_yolo_class(monkeypatch):
    """
    Mock掉 detector.py 里导入的 YOLO 类本身。

    对应fastapi-rag项目里"在模块边界mock"的同一思路：
    不去真的加载几十MB的模型权重，只是替换掉 YOLO(...) 这个调用，
    换成一个可以自由控制返回值的假对象，让测试跑得快、不依赖真实环境。
    """
    mock_class = MagicMock()
    monkeypatch.setattr("detector.YOLO", mock_class)
    return mock_class
