"""
compare_models.py —— 对比三个版本模型的推理速度(FPS)

三个版本:
  1. PyTorch FP32 (原始 .pt 权重)
  2. ONNX FP32 (导出后，未量化)
  3. ONNX INT8 (量化后)

用ultralytics的YOLO类统一加载这三种格式——它会自动识别文件格式(.pt/.onnx)，
内部自动选用对应的推理后端，不需要为ONNX单独手写推理代码，
这样三个版本的测试逻辑完全一致，对比才有意义。

注意: mAP精度对比需要完整的验证集(该数据集存放在Kaggle上，本地暂无)，
本脚本只做FPS(速度)对比。如果之后能拿到本地验证集，可以用
model.val(data="data.yaml", device="cpu") 补充精度对比。
"""

import os
import sys
import time

from ultralytics import YOLO

MODELS = {
    "PyTorch FP32": "training/yolov8/best.pt",
    "ONNX FP32": "training/yolov8/best.onnx",
    "ONNX INT8": "training/yolov8/best_int8.onnx",
}

DEFAULT_CALIBRATION_DIR = "deployment/model/calibration_images"


def find_test_image() -> str:
    """默认用校准图片文件夹里的第一张图做速度测试"""
    if not os.path.isdir(DEFAULT_CALIBRATION_DIR):
        raise FileNotFoundError(
            f"找不到测试图片目录: {DEFAULT_CALIBRATION_DIR}，"
            f"请指定一张图片路径作为命令行参数"
        )
    valid_ext = (".jpg", ".jpeg", ".png")
    for f in sorted(os.listdir(DEFAULT_CALIBRATION_DIR)):
        if f.lower().endswith(valid_ext):
            return os.path.join(DEFAULT_CALIBRATION_DIR, f)
    raise FileNotFoundError(f"{DEFAULT_CALIBRATION_DIR} 里没有找到图片")


def benchmark_fps(model_path: str, test_image: str, n_warmup: int = 5, n_runs: int = 30) -> float:
    """
    对单个模型跑速度测试。
    先warmup几次(排除模型第一次加载/JIT编译的额外开销)，再正式计时n_runs次取平均。
    """
    model = YOLO(model_path)

    for _ in range(n_warmup):
        model.predict(test_image, device="cpu", verbose=False)

    start = time.time()
    for _ in range(n_runs):
        model.predict(test_image, device="cpu", verbose=False)
    elapsed = time.time() - start

    return n_runs / elapsed


def run_comparison(test_image: str):
    print(f"测试图片: {test_image}\n")

    results = []
    for name, path in MODELS.items():
        if not os.path.exists(path):
            print(f"跳过 [{name}]: 找不到文件 {path}")
            continue

        print(f"测试中: {name} ...")
        fps = benchmark_fps(path, test_image)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        results.append((name, fps, size_mb))
        print(f"  -> {fps:.1f} FPS, 文件大小 {size_mb:.1f} MB\n")

    # 汇总表格
    print("=" * 50)
    print(f"{'模型版本':<15}{'FPS':>10}{'体积(MB)':>15}")
    print("-" * 50)
    for name, fps, size_mb in results:
        print(f"{name:<15}{fps:>10.1f}{size_mb:>15.1f}")
    print("=" * 50)

    return results


if __name__ == "__main__":
    test_image = sys.argv[1] if len(sys.argv) > 1 else find_test_image()
    run_comparison(test_image)
