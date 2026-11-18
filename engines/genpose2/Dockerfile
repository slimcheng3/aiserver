# syntax=docker/dockerfile:1
# GenPose2 运行时。模型代码来自本机 clone，权重不打进镜像。
# docker build -f engines/genpose2/Dockerfile --build-context common=engines/common --build-context genpose2=D:\playgound\genpose2 -t local/genpose2-engine:latest engines/genpose2

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    WEIGHTS_PATH=/weights/checkpoint \
    PORT=8081

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip ca-certificates \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --break-system-packages \
        torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128 \
    && python -m pip install --no-cache-dir --break-system-packages "numpy>=1.26,<2"

COPY --from=genpose2 / /opt/genpose2
RUN python -m pip install --no-cache-dir --break-system-packages /opt/genpose2

WORKDIR /opt/engine
COPY --from=common runtime_loader.py /opt/engine/runtime_loader.py
COPY serve.py /opt/engine/serve.py

EXPOSE 8081
CMD ["python", "serve.py"]
