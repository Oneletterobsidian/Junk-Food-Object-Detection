"""
export_onnx.py —— 把训练好的YOLOv8 .pt权重，导出成ONNX格式(FP32)

这一步只是格式转换，不涉及训练/反向传播。
导出后的.onnx文件会跟原始.pt文件放在同一个目录下。
"""

import os
import sys

from ultralytics import YOLO

# 默认路径可以通过命令行参数覆盖: python export_onnx.py <pt路径>
DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "training/yolov8/best.pt")


def export_to_onnx(pt_path: str, imgsz: int = 640) -> str:
    """
    导出ONNX模型。

    参数:
        pt_path: 原始.pt权重文件路径
        imgsz: 输入图片尺寸，跟训练时保持一致(640x640)

    返回:
        导出后的.onnx文件路径
    """
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"找不到模型权重文件: {pt_path}")

    print(f"加载模型: {pt_path}")
    model = YOLO(pt_path)

    print(f"导出ONNX (imgsz={imgsz})...")
    # ultralytics内置的export方法，自动处理opset版本、动态输入等细节
    onnx_path = model.export(format="onnx", imgsz=imgsz, opset=12)

    print(f"导出完成: {onnx_path}")
    return onnx_path


if __name__ == "__main__":
    pt_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_PATH
    export_to_onnx(pt_path)
