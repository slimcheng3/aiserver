# syntax=docker/dockerfile:1
# FlashRT 运行时。源码和 CUTLASS 在 upstream/。按本机 GPU 编核（4090=89, Orin=87, 5090=120）。
# 默认构建架构 = 这台机器要跑的架构。权重不打进镜像。
#
# docker build -f engines/flashrt/Dockerfile --build-context common=engines/common \
#   --build-arg GPU_ARCH=89 -t local/flashrt-engine:latest engines/flashrt

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.10-py3
FROM ${BASE_IMAGE}

ARG GPU_ARCH=89
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    WEIGHTS_PATH=/weights/checkpoint \
    PORT=8000 \
    GPU_ARCH=${GPU_ARCH} \
    PYTHONPATH=/opt/flashrt

RUN apt-get update \
    && apt-get install -y --no-install-recommends ccache git cmake ninja-build curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pybind11 ninja

WORKDIR /opt/flashrt
COPY upstream /opt/flashrt

# CUTLASS 头文件已 vendored 时跳过。直连 GitHub 在国内常 SSL 超时，优先用发行包镜像。
RUN if [ ! -f third_party/cutlass/include/cutlass/cutlass.h ]; then \
        mkdir -p third_party && \
        (curl -fL --retry 5 --retry-delay 2 --connect-timeout 20 --max-time 180 \
            -o /tmp/cutlass.tgz \
            https://ghfast.top/https://github.com/NVIDIA/cutlass/archive/refs/tags/v4.4.2.tar.gz \
         || curl -fL --retry 5 --retry-delay 2 --connect-timeout 20 --max-time 180 \
            -o /tmp/cutlass.tgz \
            https://github.com/NVIDIA/cutlass/archive/refs/tags/v4.4.2.tar.gz) \
        && tar -xzf /tmp/cutlass.tgz -C third_party \
        && rm -rf third_party/cutlass \
        && mv third_party/cutlass-4.4.2 third_party/cutlass \
        && rm -f /tmp/cutlass.tgz \
        && find third_party/cutlass -mindepth 1 -maxdepth 1 ! -name include ! -name tools ! -name examples ! -name LICENSE.txt -exec rm -rf {} + \
        && find third_party/cutlass/tools -mindepth 1 -maxdepth 1 ! -name util -exec rm -rf {} + \
        && find third_party/cutlass/tools/util -mindepth 1 -maxdepth 1 ! -name include -exec rm -rf {} + \
        && find third_party/cutlass/examples -mindepth 1 -maxdepth 1 ! -name 77_blackwell_fmha -exec rm -rf {} + \
        && find third_party/cutlass/examples/77_blackwell_fmha -type f \( -name '*.cu' -o -name 'CMakeLists.txt' \) -delete; \
    fi \
    && test -f third_party/cutlass/include/cutlass/cutlass.h

# 本机构建：只编当前 arch 的 SASS。关掉默认会编进来的 Motus/LingBot 等无关目标。
RUN cmake -B build -S . \
        -DGPU_ARCH=${GPU_ARCH} \
        -DFA2_ARCH_NATIVE_ONLY=ON \
        -DFA2_HDIMS="96;256" \
        -DFLASHRT_SLIM_BUILD=ON \
        -DFLASHRT_ENABLE_MOTUS=OFF \
        -DFLASHRT_ENABLE_LINGBOT=OFF \
        -DFLASHRT_ENABLE_AUDIO_CODEBOOK=OFF \
        -DFLASHRT_ENABLE_QWEN3_FP8_PREFILL=OFF \
    && cmake --build build -j"$(nproc)" \
    && rm -rf build/CMakeFiles

RUN python3 -m pip install --no-cache-dir --no-build-isolation -e ".[torch]" \
    && python3 -m pip install --no-cache-dir "msgpack>=1.0.5" "websockets>=12" \
        "jax[cpu]" "ml_dtypes" "orbax-checkpoint" "flax"

RUN python3 -c "import os, flash_rt; from flash_rt import flash_rt_kernels; \
print('flash_rt', flash_rt.__version__, 'sm'+os.environ.get('GPU_ARCH',''))"

# OpenPI 权重不带 PaliGemma 分词器。容器以 root 运行，缓存放在 /root。
COPY paligemma_tokenizer.model /root/.cache/flash_rt/paligemma_tokenizer.model

WORKDIR /opt/engine
COPY --from=common runtime_loader.py /opt/engine/runtime_loader.py
COPY serve.py /opt/engine/serve.py

EXPOSE 8000
CMD ["python3", "serve.py"]
