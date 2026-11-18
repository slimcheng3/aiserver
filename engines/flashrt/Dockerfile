# syntax=docker/dockerfile:1
# FlashRT 运行 π0.5。计算核来自本机的 FlashRT 目录，不通过 requirements.txt 安装。
# docker build -f engines/flashrt/Dockerfile --build-context common=engines/common --build-context flashrt=D:\playgound\flashrt -t local/flashrt-engine:latest engines/flashrt

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    WEIGHTS_PATH=/weights/checkpoint \
    PORT=8000

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip ca-certificates \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --break-system-packages \
        torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128

COPY --from=flashrt / /opt/flashrt
RUN python -m pip install --no-cache-dir --break-system-packages /opt/flashrt

WORKDIR /opt/engine
COPY --from=common runtime_loader.py /opt/engine/runtime_loader.py
COPY serve.py /opt/engine/serve.py

EXPOSE 8000
CMD ["python", "serve.py"]
