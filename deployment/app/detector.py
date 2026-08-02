"""
detector.py —— 推理引擎层核心模块

职责：只做"加载模型 + 预测 + 结果解析"，不训练、不评估。
对应架构笔记里的"推理引擎层"，被 FastAPI 服务层和实时视频demo共同复用。
"""

import os
from ultralytics import YOLO

# ── 配置：环境变量 + 生产默认值（跟fastapi-rag项目同样的DI模式）──
# 本地开发/测试时可以用不同路径，Docker容器里用默认值，不用改代码
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/model/best.pt")

# 注意：模型自带的 model.names 在训练时就存成了纯数字字符串("0","1","2"...)，
# 不是真正的食物名字（推测是 notebook_1_yolov8.py 里那段fallback逻辑导致的：
# 手写类别名单数量跟实际类别数nc对不上时，会自动退化成用数字代替）。
# 这份映射来自demo.py里已经验证过的NAMES字典，是真实可信的类别名。
# 注意：这份字典不是完整的35类，只覆盖demo.py里出现过的类别号，
# 未出现的类别号会fallback显示成数字（见下方_parse_results）
CLASS_NAMES = {
    0: 'Biryani', 1: 'Burger', 2: 'Unknown', 3: 'Hotdog', 4: 'Chips',
    15: 'Pizza', 20: 'Kabab', 21: 'Mac and Cheese', 22: 'Meatloaf',
    23: 'Muffin', 24: 'Nachos', 25: 'Cookies', 26: 'French Fry',
    27: 'Ice Cream', 28: 'Pizza', 29: 'Processed Cheese',
    30: 'Cheesecake', 31: 'Crispy Chicken', 32: 'Sandwich',
    33: 'Noodles', 34: 'Waffles'
}


class Detector:
    """
    单例式推理封装：模型只在实例化时加载一次，之后反复调用 predict()。
    FastAPI 服务里应该在应用启动时创建一次全局实例，不要每次请求都 new 一个。
    """

    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"找不到模型权重文件: {model_path}\n"
                f"请检查 MODEL_PATH 环境变量，或确认 best.pt 是否放在了正确路径"
            )
        # ultralytics 的 YOLO 类在没有GPU时会自动退回CPU，
        # 但显式指定更清晰，也避免它偷偷尝试调用不存在的CUDA
        self.model = YOLO(model_path)
        self.model_path = model_path
        # 用demo.py里验证过的CLASS_NAMES字典纠正类别名，而不是模型自带的(有问题的)model.names
        self.class_names = CLASS_NAMES

    def predict(self, image, conf: float = 0.25) -> list[dict]:
        """
        对单张图片做推理，返回结构化的检测框列表。

        参数:
            image: 可以是文件路径(str)、numpy数组、PIL.Image，
                   ultralytics内部会自动处理这几种输入类型
            conf: 置信度阈值，低于这个值的检测框会被过滤掉

        返回:
            [
                {"class_id": 6, "class_name": "french_fries",
                 "confidence": 0.87, "bbox": [x1, y1, x2, y2]},
                ...
            ]
        """
        results = self.model.predict(
            source=image,
            conf=conf,
            device="cpu",   # 显式指定CPU，跟当前硬件环境保持一致
            verbose=False
        )
        return self._parse_results(results)

    def _parse_results(self, results) -> list[dict]:
        """把ultralytics返回的Results对象，整理成干净的可JSON序列化字典"""
        detections = []
        # predict()对单张图返回长度为1的list，取第一个元素
        boxes = results[0].boxes

        for box in boxes:
            class_id = int(box.cls[0])
            class_name = self.class_names.get(class_id, str(class_id))
            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(float(box.conf[0]), 4),
                "bbox": [round(v, 2) for v in box.xyxy[0].tolist()],
            })
        return detections


# ── 简单的本地测试入口，方便在写pytest之前先手动跑通 ──
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python detector.py <图片路径>")
        sys.exit(1)

    detector = Detector()
    results = detector.predict(sys.argv[1])
    print(f"检测到 {len(results)} 个目标:")
    for r in results:
        print(f"  {r['class_name']} (conf={r['confidence']}) bbox={r['bbox']}")
