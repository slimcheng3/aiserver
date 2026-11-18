# syntax=docker/dockerfile:1
# SAM 3 运行时。本文件和 serve.py 由交付台维护。
# 模型代码 clone 到仓库根目录的 tmp/sam3，不预置、不提交。第一次构建发现还没有代码时，
# 先创建该目录，再 clone https://github.com/facebookresearch/sam3 。权重不打进镜像。
#
# 在交付台仓库根目录执行：
# docker build -f engines/sam3/Dockerfile --build-context common=engines/common --build-context sam3=tmp/sam3 -t local/sam3-engine:latest engines/sam3

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
        torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128 \
    && python -m pip install --no-cache-dir --break-system-packages "numpy>=1.26,<2"

COPY --from=sam3 / /opt/sam3
RUN python -m pip install --no-cache-dir --break-system-packages /opt/sam3

WORKDIR /opt/engine
COPY --from=common runtime_loader.py /opt/engine/runtime_loader.py
COPY serve.py /opt/engine/serve.py

EXPOSE 8080
CMD ["python", "serve.py"]
