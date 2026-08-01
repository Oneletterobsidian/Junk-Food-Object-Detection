"""
test_main.py —— FastAPI服务层的测试

同样mock掉真正的Detector/YOLO，只测试"接口层"的行为：
路由对不对、状态码对不对、返回的JSON结构对不对。
"""

import io
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture
def client(monkeypatch):
    """
    创建一个TestClient，同时把main.py里的Detector类替换成假的，
    这样lifespan启动时不会真的去加载模型。
    """
    mock_detector_instance = MagicMock()
    mock_detector_class = MagicMock(return_value=mock_detector_instance)
    monkeypatch.setattr("main.Detector", mock_detector_class)

    import main
    with TestClient(main.app) as test_client:
        # 把mock实例挂到test_client上，方便测试函数里直接改predict()的返回值
        test_client.mock_detector = mock_detector_instance
        yield test_client


def _make_fake_image_bytes():
    """生成一张最小的内存图片，用于模拟文件上传，不需要真实图片文件"""
    image = Image.new("RGB", (10, 10), color="white")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer


def test_health_check_returns_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_detect_image_returns_detections(client):
    client.mock_detector.predict.return_value = [
        {"class_id": 26, "class_name": "French Fry", "confidence": 0.58, "bbox": [1, 2, 3, 4]}
    ]

    response = client.post(
        "/detect/image",
        files={"file": ("test.jpg", _make_fake_image_bytes(), "image/jpeg")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["detections"][0]["class_name"] == "French Fry"
    assert body["filename"] == "test.jpg"


def test_detect_image_no_detections_returns_empty_list(client):
    client.mock_detector.predict.return_value = []

    response = client.post(
        "/detect/image",
        files={"file": ("empty.jpg", _make_fake_image_bytes(), "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json()["count"] == 0
    assert response.json()["detections"] == []


def test_detect_image_invalid_file_returns_400(client):
    # 传一段乱七八糟的字节，模拟"不是图片"的情况
    bad_file = io.BytesIO(b"this is not an image")

    response = client.post(
        "/detect/image",
        files={"file": ("bad.txt", bad_file, "text/plain")},
    )

    assert response.status_code == 400


def test_detect_image_passes_custom_confidence(client):
    client.mock_detector.predict.return_value = []

    client.post(
        "/detect/image?conf=0.5",
        files={"file": ("test.jpg", _make_fake_image_bytes(), "image/jpeg")},
    )

    # 确认conf参数被正确传给了Detector.predict()
    call_kwargs = client.mock_detector.predict.call_args.kwargs
    assert call_kwargs["conf"] == 0.5
