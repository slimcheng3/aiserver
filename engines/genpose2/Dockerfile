# syntax=docker/dockerfile:1
# GenPose2 运行时。推理代码在 upstream/。Score/Energy/Scale 三个 pth 不进镜像。
# DINOv2 ViT-S/14 是固定骨干，打进 TORCH_HOME，启动时不访问 GitHub。
#
# docker build -f engines/genpose2/Dockerfile --build-context common=engines/common -t local/genpose2-engine:latest engines/genpose2

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS ext

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip python3-dev git ca-certificates ninja-build \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --break-system-packages \
        torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128

COPY upstream/networks/pts_encoder/pointnet2_utils/pointnet2 /tmp/pointnet2
WORKDIR /tmp/pointnet2
RUN python setup.py build_ext --inplace

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    WEIGHTS_PATH=/weights/checkpoint \
    PORT=8081 \
    PYTHONPATH=/opt/genpose2 \
    TORCH_HOME=/opt/torch-hub \
    OPENCV_IO_ENABLE_OPENEXR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip ca-certificates libgl1 libglib2.0-0 libgomp1 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --break-system-packages \
        torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128 \
    && python -m pip install --no-cache-dir --break-system-packages \
        "numpy>=1.26,<2" opencv-python-headless==4.10.0.84 scipy==1.14.1 scikit-learn tqdm pillow cutoop tensorboardX

COPY --from=ext /tmp/pointnet2/*.so /usr/local/lib/python3.12/dist-packages/
COPY upstream /opt/genpose2
COPY torch-hub /opt/torch-hub/hub

WORKDIR /opt/engine
COPY --from=common runtime_loader.py /opt/engine/runtime_loader.py
COPY serve.py /opt/engine/serve.py

EXPOSE 8081
CMD ["python", "serve.py"]
