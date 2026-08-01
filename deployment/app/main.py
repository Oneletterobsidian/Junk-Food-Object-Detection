"""
main.py —— FastAPI服务层

职责：接收HTTP请求（上传图片），调用Detector做推理，返回JSON结果。
不包含推理逻辑本身（那是detector.py的职责），这一层只负责"接口"。
"""

import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from detector import MODEL_PATH, Detector

# 全局变量持有唯一的Detector实例（单例）
# 用lifespan而不是在import时就创建，是为了让FastAPI能完整控制"启动/关闭"的时机，
# 也方便以后写测试时用依赖注入的方式替换掉它
detector_instance: Detector | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动时加载一次模型，关闭时释放。FastAPI生命周期钩子的标准写法。"""
    global detector_instance
    detector_instance = Detector()
    yield
    detector_instance = None


app = FastAPI(
    title="Junk Food Detection API",
    description="基于YOLOv8的垃圾食品检测服务",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    """健康检查接口，方便部署时确认服务是否正常启动、模型是否加载成功"""
    return {
        "status": "ok" if detector_instance is not None else "model_not_loaded",
        "model_path": MODEL_PATH,
    }


@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...), conf: float = 0.25):
    """
    上传一张图片，返回检测到的目标列表。

    参数:
        file: 上传的图片文件（jpg/png等常见格式）
        conf: 置信度阈值，默认0.25，跟detector.py里的默认值保持一致
    """
    if detector_instance is None:
        # 理论上不应该发生（lifespan保证启动时就加载），
        # 但防御性地处理一下，避免模型加载失败时返回更难懂的错误
        raise HTTPException(status_code=503, detail="模型尚未加载完成，请稍后重试")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="无法解析上传的图片文件，请确认格式正确")

    detections = detector_instance.predict(image, conf=conf)

    return {
        "filename": file.filename,
        "count": len(detections),
        "detections": detections,
    }
