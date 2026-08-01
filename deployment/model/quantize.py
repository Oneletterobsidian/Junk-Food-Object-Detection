"""
quantize.py —— 把ONNX(FP32)模型，做静态量化(INT8)

需要一批"校准图片"来统计激活值分布，跟训练集是否一致不重要，
重要的是图片内容跟实际使用场景类似(都是自然图片、都是food类图片更好，
但用网上随便找的食物图也足够用来演示这个流程)。
"""

import os
import sys

import cv2
import numpy as np
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)

DEFAULT_ONNX_PATH = os.environ.get("ONNX_PATH", "training/yolov8/best.onnx")
DEFAULT_CALIBRATION_DIR = os.environ.get(
    "CALIBRATION_DIR", "deployment/model/calibration_images"
)


def preprocess_image(image_path: str, imgsz: int = 640) -> np.ndarray:
    """
    把一张图片处理成YOLO模型期望的输入格式:
    resize到imgsz x imgsz -> BGR转RGB -> 归一化到[0,1] -> HWC转CHW -> 加batch维度
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")

    image = cv2.resize(image, (imgsz, imgsz))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
    image = np.expand_dims(image, axis=0)   # 加batch维度 -> (1, 3, H, W)
    return image


class FolderCalibrationDataReader(CalibrationDataReader):
    """
    从指定文件夹里读取图片，逐张喂给量化工具做"摸底"。
    图片数量建议至少5-20张，越有代表性效果越好，但不需要跟训练集完全一致。
    """

    def __init__(self, calibration_dir: str, input_name: str, imgsz: int = 640):
        if not os.path.isdir(calibration_dir):
            raise FileNotFoundError(
                f"找不到校准图片目录: {calibration_dir}\n"
                f"请在这个路径下放几张食物图片(jpg/png)后再运行"
            )

        valid_ext = (".jpg", ".jpeg", ".png")
        self.image_paths = [
            os.path.join(calibration_dir, f)
            for f in sorted(os.listdir(calibration_dir))
            if f.lower().endswith(valid_ext)
        ]
        if not self.image_paths:
            raise ValueError(f"目录 {calibration_dir} 里没有找到任何图片文件")

        print(f"校准数据集: {len(self.image_paths)} 张图片")
        self.input_name = input_name
        self.imgsz = imgsz
        self._iterator = iter(self.image_paths)

    def get_next(self):
        path = next(self._iterator, None)
        if path is None:
            return None
        return {self.input_name: preprocess_image(path, self.imgsz)}


def quantize_model(onnx_path: str, calibration_dir: str) -> str:
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(
            f"找不到ONNX模型: {onnx_path}，请先运行 export_onnx.py 生成它"
        )

    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    output_path = onnx_path.replace(".onnx", "_int8.onnx")

    print(f"开始量化: {onnx_path} -> {output_path}")
    calibration_reader = FolderCalibrationDataReader(calibration_dir, input_name)

    quantize_static(
        model_input=onnx_path,
        model_output=output_path,
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,  # x64 CPU上推荐用QDQ格式，避免QOperator格式导致的性能下降
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
    )

    print(f"量化完成: {output_path}")

    # 顺便打印一下体积对比，这是最直观的量化效果
    original_size = os.path.getsize(onnx_path) / (1024 * 1024)
    quantized_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"原始体积: {original_size:.1f} MB")
    print(f"量化后体积: {quantized_size:.1f} MB")
    print(f"体积压缩: {(1 - quantized_size / original_size) * 100:.1f}%")

    return output_path


if __name__ == "__main__":
    onnx_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ONNX_PATH
    calibration_dir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_CALIBRATION_DIR
    quantize_model(onnx_path, calibration_dir)
