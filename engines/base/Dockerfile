# syntax=docker/dockerfile:1
# CUDA 12.8、Python、PyTorch。不含模型代码和权重。
# 在交付台仓库根目录执行：
# docker build -f engines/base/Dockerfile --build-context common=engines/common -t local/base-cuda128-torch:latest engines/base

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    WEIGHTS_PATH=/weights/checkpoint \
    PORT=8080

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip ca-certificates \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --break-system-packages \
        torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128

WORKDIR /opt/engine
COPY --from=common runtime_loader.py /opt/engine/runtime_loader.py
COPY serve.py /opt/engine/serve.py

EXPOSE 8080
CMD ["python", "serve.py"]
