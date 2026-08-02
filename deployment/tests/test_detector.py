"""
test_detector.py —— Detector类的单元测试

覆盖三块逻辑：
1. 模型文件不存在时的报错行为
2. predict() 是否用正确的参数调用底层模型
3. _parse_results() 结果解析、类别名映射是否正确（包括fallback情况）

不依赖真实的 best.pt 权重文件，YOLO类被mock掉，测试跑得快、不挑环境。
"""

import pytest
from detector import Detector, CLASS_NAMES


# ── 测试用的假 ultralytics Results 对象 ──────────────────────

class FakeXYXY:
    """模拟 box.xyxy[0]，真实场景下这是个tensor，这里只需要 .tolist()"""
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


class FakeBox:
    """模拟 ultralytics 单个检测框对象"""
    def __init__(self, cls_id, conf, bbox):
        self.cls = [cls_id]
        self.conf = [conf]
        self.xyxy = [FakeXYXY(bbox)]


class FakeResult:
    """模拟 model.predict() 返回的 Results 对象（取 results[0] 用）"""
    def __init__(self, boxes):
        self.boxes = boxes


# ── 1. 初始化相关 ────────────────────────────────────────────

def test_missing_model_file_raises_error(tmp_path):
    """模型文件不存在时，应该在加载前就报错，而不是让ultralytics抛出更难懂的异常"""
    nonexistent_path = str(tmp_path / "does_not_exist.pt")

    with pytest.raises(FileNotFoundError, match="找不到模型权重文件"):
        Detector(model_path=nonexistent_path)


def test_detector_loads_when_file_exists(mock_yolo_class, fake_weights):
    """文件存在时，应该正常初始化，并且调用了(mock的)YOLO类去加载"""
    detector = Detector(model_path=fake_weights)

    mock_yolo_class.assert_called_once_with(fake_weights)
    assert detector.model_path == fake_weights
    assert detector.class_names == CLASS_NAMES


# ── 2. predict() 参数传递是否正确 ────────────────────────────

def test_predict_calls_model_with_expected_args(mock_yolo_class, fake_weights):
    """确认predict()把正确的参数传给了底层model.predict()，尤其是device='cpu'这个关键设定"""
    mock_model_instance = mock_yolo_class.return_value
    # 让mock的model.predict()返回一个"没有检测到任何框"的空结果，避免这里牵扯解析逻辑
    mock_model_instance.predict.return_value = [FakeResult(boxes=[])]

    detector = Detector(model_path=fake_weights)
    detector.predict("some_image.jpg", conf=0.4)

    mock_model_instance.predict.assert_called_once_with(
        source="some_image.jpg",
        conf=0.4,
        device="cpu",
        verbose=False,
    )


def test_predict_uses_default_confidence(mock_yolo_class, fake_weights):
    """不传conf参数时，应该用默认值0.25"""
    mock_model_instance = mock_yolo_class.return_value
    mock_model_instance.predict.return_value = [FakeResult(boxes=[])]

    detector = Detector(model_path=fake_weights)
    detector.predict("some_image.jpg")

    called_kwargs = mock_model_instance.predict.call_args.kwargs
    assert called_kwargs["conf"] == 0.25


# ── 3. 结果解析 + 类别名映射 ──────────────────────────────────

def test_parse_results_known_class(mock_yolo_class, fake_weights):
    """已知类别号(26 -> French Fry)应该正确映射成名字，数值也要对得上"""
    detector = Detector(model_path=fake_weights)

    fake_result = FakeResult(boxes=[
        FakeBox(cls_id=26, conf=0.5818, bbox=[152.04, 76.43, 749.78, 354.29]),
    ])

    parsed = detector._parse_results([fake_result])

    assert len(parsed) == 1
    assert parsed[0]["class_id"] == 26
    assert parsed[0]["class_name"] == "French Fry"
    assert parsed[0]["confidence"] == 0.5818
    assert parsed[0]["bbox"] == [152.04, 76.43, 749.78, 354.29]


def test_parse_results_unknown_class_falls_back_to_number(mock_yolo_class, fake_weights):
    """CLASS_NAMES字典里没覆盖到的类别号，应该fallback成字符串数字，而不是报错"""
    detector = Detector(model_path=fake_weights)

    fake_result = FakeResult(boxes=[
        FakeBox(cls_id=99, conf=0.7, bbox=[0, 0, 10, 10]),
    ])

    parsed = detector._parse_results([fake_result])

    assert parsed[0]["class_name"] == "99"


def test_parse_results_no_detections_returns_empty_list(mock_yolo_class, fake_weights):
    """图片里什么都没检测到时，应该返回空列表，而不是报错或返回None"""
    detector = Detector(model_path=fake_weights)

    fake_result = FakeResult(boxes=[])
    parsed = detector._parse_results([fake_result])

    assert parsed == []


def test_parse_results_multiple_detections(mock_yolo_class, fake_weights):
    """一张图检测到多个目标时，应该按顺序全部解析出来"""
    detector = Detector(model_path=fake_weights)

    fake_result = FakeResult(boxes=[
        FakeBox(cls_id=1, conf=0.9, bbox=[0, 0, 10, 10]),    # Burger
        FakeBox(cls_id=26, conf=0.6, bbox=[20, 20, 30, 30]),  # French Fry
    ])

    parsed = detector._parse_results([fake_result])

    assert len(parsed) == 2
    assert parsed[0]["class_name"] == "Burger"
    assert parsed[1]["class_name"] == "French Fry"
