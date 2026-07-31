# ============================================================
# Junk Food Detection - 部署服务 Dockerfile
# 纯CPU环境（跟本地开发环境保持一致，不涉及任何CUDA/GPU依赖）
# ============================================================

FROM python:3.11-slim

WORKDIR /app

# ── 第一步：先只拷贝 requirements.txt 并安装依赖 ──
# 这样做是为了利用Docker的层缓存：只要requirements.txt不变，
# 以后改代码重新build时，这一层（安装依赖，最耗时的部分）可以直接复用缓存，
# 不用每次都重新下载几百MB的依赖
COPY requirements.txt .

# torch/torchvision的CPU版本不在默认PyPI源上，
# 需要额外指定 --extra-index-url 才能找到 +cpu 这个版本
#
# 注意：ultralytics自己依赖普通版opencv-python，会跟requirements.txt里
# 指定的opencv-python-headless一起被装进来，两者共用cv2这个模块路径，
# 会互相覆盖/污染彼此的文件（单纯卸载其中一个会把另一个的文件也带坏）。
# 所以这里先把两个都卸载干净，再单独重新装一次headless版，确保cv2模块完整、干净。
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt \
    && pip uninstall -y opencv-python opencv-python-headless \
    && pip install --no-cache-dir opencv-python-headless

# ── 第二步：拷贝应用代码 ──
# 注意：模型权重(best.pt)不在这里拷贝，而是通过docker run -v 挂载进来，
# 保持镜像轻量，且权重更新不需要重新build镜像
COPY deployment/app ./app
COPY deployment/tests ./tests

# ── 环境变量默认值 ──
# 对应容器内的模型路径，跟docker-compose.yml里的挂载路径要对应上
ENV MODEL_PATH=/app/model/best.pt
ENV PYTHONUNBUFFERED=1

# ── 默认命令：先跑一遍pytest做健康检查 ──
# 后续FastAPI服务(main.py)写好之后，这里会换成:
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["pytest", "tests/", "-v"]
