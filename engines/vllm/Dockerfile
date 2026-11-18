# syntax=docker/dockerfile:1
# vLLM 自带计算核，不通过用户的 requirements.txt 安装。权重不打进镜像。
# docker build -f engines/vllm/Dockerfile --build-context common=engines/common -t local/vllm-engine:latest engines/vllm

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    WEIGHTS_PATH=/weights/checkpoint \
    PORT=8082 \
    HF_HUB_OFFLINE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip ca-certificates \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --break-system-packages vllm

WORKDIR /opt/engine
COPY --from=common runtime_loader.py /opt/engine/runtime_loader.py
COPY serve.py /opt/engine/serve.py

EXPOSE 8082
CMD ["python", "serve.py"]
