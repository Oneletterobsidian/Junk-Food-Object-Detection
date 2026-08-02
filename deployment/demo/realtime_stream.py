"""
realtime_stream.py —— 实时视频流检测demo

逐帧读取视频(摄像头或本地视频文件)，用Detector画出检测框，
把结果保存成一个新的视频文件(不用cv2.imshow弹窗，避免依赖图形界面库，
跟当前环境用的opencv-python-headless保持兼容)。

用法:
    python realtime_stream.py <输入源> <输出路径>

    输入源可以是:
        - 数字(如 0)：本地摄像头编号
        - 视频文件路径(如 test_video.mp4)

示例:
    python realtime_stream.py 0 demo_output.mp4          # 用摄像头
    python realtime_stream.py my_video.mp4 demo_output.mp4  # 用视频文件
"""

import os
import sys
import time

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
from detector import Detector  # noqa: E402


def draw_detections(frame, detections):
    """在一帧图像上画出所有检测框和标签"""
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        label = f"{det['class_name']} {det['confidence']:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 在框上方画一个实心背景条，让文字更清楚
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - text_h - 8), (x1 + text_w, y1), (0, 255, 0), -1)
        cv2.putText(
            frame, label, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
        )
    return frame


def run_stream(source, output_path: str, conf: float = 0.25, max_frames: int = None):
    """
    参数:
        source: 摄像头编号(int)或视频文件路径(str)
        output_path: 输出视频保存路径
        conf: 检测置信度阈值
        max_frames: 最多处理多少帧(None表示处理到视频结束/手动中断)
    """
    detector = Detector()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频源: {source}")

    fps_in = cap.get(cv2.CAP_PROP_FPS)
    if fps_in is None or fps_in <= 0:
        # 部分摄像头驱动查询帧率会返回-1或0(查询失败)，这里做个兜底
        fps_in = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path = os.path.abspath(output_path)
    writer = cv2.VideoWriter(output_path, fourcc, fps_in, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            f"无法创建输出视频文件: {output_path}\n"
            f"可能是mp4v编码器在当前环境不可用，可以尝试把输出文件后缀改成 .avi 再试一次"
        )

    frame_count = 0
    total_infer_time = 0.0

    print(f"开始处理视频源: {source}")
    print(f"分辨率: {width}x{height}, 输入FPS: {fps_in:.1f}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            start = time.time()
            detections = detector.predict(frame, conf=conf)
            total_infer_time += time.time() - start

            annotated = draw_detections(frame, detections)
            writer.write(annotated)

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"已处理 {frame_count} 帧...")

            if max_frames is not None and frame_count >= max_frames:
                break
    finally:
        cap.release()
        writer.release()

    avg_fps = frame_count / total_infer_time if total_infer_time > 0 else 0
    print(f"\n处理完成: 共 {frame_count} 帧")
    print(f"平均推理速度: {avg_fps:.1f} FPS")
    print(f"结果已保存: {output_path}")
    if frame_count == 0:
        print("警告: 没有处理任何帧，请检查视频源是否正常打开")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python realtime_stream.py <输入源(摄像头编号或视频路径)> <输出路径>")
        sys.exit(1)

    raw_source = sys.argv[1]
    # 摄像头编号是数字，视频路径是字符串，这里做个简单判断
    source = int(raw_source) if raw_source.isdigit() else raw_source
    output_path = sys.argv[2]

    run_stream(source, output_path)
