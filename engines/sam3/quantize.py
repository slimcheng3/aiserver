"""把 SAM 3 的 checkpoint 写成 fp16、bf16 或 int8。

fp16 和 bf16 只改浮点精度，键保持原样。
int8 把二维权重收成 int8，并记下每一行的比例，服务用来替换线性层。
这个文件由量化作业挂进 SAM 3 镜像执行，不进镜像构建。
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

SRC = os.environ["WEIGHTS_PATH"]
DST = os.environ["QUANT_OUTPUT"]
QUANT = os.environ.get("QUANT", "")
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


def convert(value, dtype: torch.dtype):
    if isinstance(value, torch.Tensor):
        if value.is_floating_point():
            return value.to(dtype)
        return value
    if isinstance(value, dict):
        return {key: convert(item, dtype) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(convert(item, dtype) for item in value)
    if isinstance(value, list):
        return [convert(item, dtype) for item in value]
    return value


def image_state(checkpoint: dict) -> dict:
    body = checkpoint
    if "model" in body and isinstance(body["model"], dict):
        body = body["model"]
    picked = {key.replace("detector.", ""): value for key, value in body.items() if isinstance(key, str) and "detector" in key}
    if picked:
        return picked
    return {key: value for key, value in body.items() if isinstance(value, torch.Tensor)}


def quantize_int8(state: dict) -> tuple[dict, dict, int]:
    saved: dict = {}
    scales: dict = {}
    count = 0
    for key, value in state.items():
        if isinstance(value, torch.Tensor) and value.is_floating_point() and value.ndim == 2:
            weight = value.float()
            limit = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
            scale = limit / 127
            saved[key] = torch.clamp(torch.round(weight / scale), -128, 127).to(torch.int8)
            scales[key] = scale.half()
            count += 1
        elif isinstance(value, torch.Tensor) and value.is_floating_point():
            saved[key] = value.half()
        else:
            saved[key] = value
    return saved, scales, count


def main() -> None:
    src = Path(SRC)
    if not src.is_file():
        raise SystemExit(f"{src} 不是 SAM 3 的 checkpoint 文件")
    print(f"读取 {src}", flush=True)
    checkpoint = torch.load(src, map_location="cpu", weights_only=True)
    dst = Path(DST)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if QUANT == "int8":
        if not isinstance(checkpoint, dict):
            raise SystemExit("int8 量化需要字典形式的 checkpoint")
        state, scales, count = quantize_int8(image_state(checkpoint))
        if count == 0:
            raise SystemExit("checkpoint 里没有可以收成 int8 的二维权重")
        print(f"收成 int8 的线性层权重 {count} 个", flush=True)
        torch.save(
            {"aiserver_quant": torch.tensor([8], dtype=torch.int32), "state": state, "scales": scales},
            dst,
        )
    else:
        dtype = DTYPES.get(QUANT)
        if dtype is None:
            raise SystemExit(f"SAM 3 只做 fp16、bf16 和 int8，收到 {QUANT}")
        print(f"写成 {QUANT}", flush=True)
        torch.save(convert(checkpoint, dtype), dst)
    print(f"写入 {dst}", flush=True)
    print("量化完成", flush=True)


if __name__ == "__main__":
    main()
