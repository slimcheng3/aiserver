# syntax=docker/dockerfile:1
# 官方 vLLM OpenAI 服务。权重不打进镜像。
# docker build -f engines/vllm/Dockerfile -t local/vllm-engine:latest engines/vllm
#
# 官方镜像 ENTRYPOINT 已是 python -m vllm.entrypoints.openai.api_server。
# 可用 --build-arg BASE_IMAGE=... 换成 Docker Hub 或其它镜像源。
# 默认走 DaoCloud，避免 docker.io / docker.1ms.run 的 429。

ARG BASE_IMAGE=docker.m.daocloud.io/vllm/vllm-openai:v0.11.0
FROM ${BASE_IMAGE}

ENV HF_HUB_OFFLINE=1 \
    WEIGHTS_PATH=/weights/checkpoint

EXPOSE 8000
CMD ["--model", "/weights/checkpoint", "--host", "0.0.0.0", "--port", "8000", "--gpu-memory-utilization", "0.5"]
