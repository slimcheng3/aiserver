"""FlashRT 上的 π0.5。对外是 OpenPI 的 WebSocket 策略协议。

连上后服务端先发一条 msgpack metadata。之后每条请求是 msgpack 观测，
响应是 msgpack 动作字典，并带上 server_timing。出错时先发一段纯文本
traceback，再用 1011 关掉这条连接。GET /healthz 返回 OpenPI 的纯文本
OK；GET /health 仍返回 JSON，给交付台的健康检查用。
"""

from __future__ import annotations

import asyncio
import functools
import http
import os
import sys
import time
import traceback
from pathlib import Path

import msgpack
import numpy as np
import websockets.asyncio.server as ws_server
import websockets.exceptions
import websockets.frames

WEIGHTS_PATH = os.environ.get("WEIGHTS_PATH", "/weights/checkpoint")
PORT = int(os.environ.get("PORT", "8000"))
METADATA: dict = {}


def pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)


def parse_image(image) -> np.ndarray:
    """Match openpi's image parse: float [0, 1] to uint8, CHW to HWC."""
    array = np.asarray(image)
    if np.issubdtype(array.dtype, np.floating):
        array = (255 * array).astype(np.uint8)
    if array.ndim == 3 and array.shape[0] == 3:
        array = np.moveaxis(array, 0, -1)
    return np.ascontiguousarray(array)


def to_model_observation(obs: dict) -> dict:
    """Map an OpenPI env observation onto the keys FlashRT's infer reads."""
    out = dict(obs)
    aliases = (
        ("observation/image", "image"),
        ("observation/exterior_image_1_left", "image"),
        ("observation/wrist_image", "wrist_image"),
        ("observation/wrist_image_left", "wrist_image"),
        ("observation/wrist_image_right", "wrist_image_right"),
    )
    for source, dest in aliases:
        if dest not in out and source in obs:
            out[dest] = obs[source]
    for key in ("image", "wrist_image", "wrist_image_right"):
        if key in out:
            out[key] = parse_image(out[key])
    if "images" in out:
        out["images"] = [parse_image(image) for image in out["images"]]
    return out


def _state_of(obs: dict):
    """状态向量，写进 π0.5 的 prompt。

    LIBERO 直接给 8 维 observation/state。DROID 把关节角和夹爪拆开，
    这里按 OpenPI 的顺序拼成关节角在前、夹爪在后。
    """
    for key in ("state", "observation/state", "proprio"):
        if key in obs and obs[key] is not None:
            return np.asarray(obs[key])
    joints = obs.get("observation/joint_position")
    if joints is None:
        return None
    joints = np.asarray(joints).reshape(-1)
    gripper = obs.get("observation/gripper_position")
    if gripper is None:
        return joints
    return np.concatenate([joints, np.asarray(gripper).reshape(-1)])


def _packable(value):
    if isinstance(value, dict):
        return {key: _packable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_packable(item) for item in value]
    if isinstance(value, (np.ndarray, np.generic)):
        return value
    detach = getattr(value, "detach", None)
    if callable(detach):
        return detach().cpu().numpy()
    return value


def checkpoint_framework(weights_path: str) -> str:
    """PyTorch 目录走 torch，OpenPI 的 JAX/Orbax 目录走 jax。

    jax 前端只换权重读取，后面的推理和 torch 前端是同一条 RTX 流水线。
    """
    root = Path(weights_path)
    if (root / "model.safetensors").is_file():
        return "torch"
    if (root / "params").is_dir() or (root / "_METADATA").is_file() or (root / "manifest.ocdbt").is_file():
        return "jax"
    raise FileNotFoundError(
        f"认不出 {weights_path} 的权重格式。"
        "PyTorch 目录需要 model.safetensors，OpenPI 的 JAX 目录需要 params/。"
    )


def load(weights_path: str):
    if not os.path.isdir(weights_path):
        raise FileNotFoundError(f"π0.5 的权重需要是 checkpoint 目录：{weights_path}")
    import flash_rt

    hardware = os.environ.get("FLASHRT_HARDWARE", "auto")
    framework = checkpoint_framework(weights_path)
    return flash_rt.load_model(weights_path, framework=framework, config="pi05", hardware=hardware)


def infer(model, obs: dict) -> dict:
    if not isinstance(obs, dict):
        raise ValueError("观测需要是字典")
    observation = to_model_observation(obs)
    prompt = obs.get("prompt", obs.get("text"))
    state = _state_of(obs)
    # VLAModel.predict 会在第一次调用时用这条观测做校准并建好 CUDA 流。
    # 直接 infer 会跳过这一步，JAX 前端上就会缺 _graph_torch_stream。
    predict = getattr(model, "predict", None)
    if callable(predict) and hasattr(model, "_pipe"):
        kwargs = {}
        if prompt is not None:
            kwargs["prompt"] = str(prompt)
        if state is not None:
            kwargs["state"] = state
        result = predict(observation, **kwargs)
        if not isinstance(result, dict):
            result = {"actions": result}
        return _packable(result)
    setter = getattr(model, "set_prompt", None)
    if callable(setter) and prompt is not None:
        try:
            setter(str(prompt), state=state)
        except TypeError:
            setter(str(prompt))
    runner = getattr(model, "infer", None) or predict
    if not callable(runner):
        raise RuntimeError("FlashRT 模型没有 infer 或 predict")
    try:
        result = runner(observation)
    except TypeError:
        result = runner(
            images=observation.get("images", observation.get("image")),
            prompt=prompt,
            state=state,
        )
    if not isinstance(result, dict):
        result = {"actions": result}
    return _packable(result)


def health_check(connection, request):
    path = request.path.split("?", 1)[0]
    if path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    if path == "/health":
        response = connection.respond(http.HTTPStatus.OK, '{"status":"ok"}')
        del response.headers["Content-Type"]
        response.headers["Content-Type"] = "application/json"
        return response
    return None


async def serve(model, infer_fn, host: str = "0.0.0.0", port: int = PORT):
    lock = asyncio.Lock()

    async def handler(websocket):
        packer = Packer()
        await websocket.send(packer.pack(METADATA))
        prev_total_time = None
        while True:
            try:
                start = time.monotonic()
                message = await websocket.recv()
                if isinstance(message, str):
                    raise ValueError("观测需要是 msgpack 二进制帧")
                obs = unpackb(message)
                infer_start = time.monotonic()
                async with lock:
                    action = await asyncio.to_thread(infer_fn, model, obs)
                infer_time = time.monotonic() - infer_start
                if not isinstance(action, dict):
                    action = {"actions": action}
                action = dict(action)
                action["server_timing"] = {"infer_ms": infer_time * 1000}
                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000
                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start
            except websockets.exceptions.ConnectionClosed:
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    return await ws_server.serve(
        handler,
        host,
        port,
        compression=None,
        max_size=None,
        process_request=health_check,
    )


async def _main() -> None:
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    common = here.parent / "common"
    if (common / "runtime_loader.py").is_file():
        sys.path.insert(0, str(common))
    from runtime_loader import resolve_fns

    load_fn, infer_fn, pipeline = resolve_fns(WEIGHTS_PATH, load, infer)
    if pipeline:
        print(f"使用权重旁的 pipeline.py：{pipeline}", flush=True)
    model = load_fn(WEIGHTS_PATH)
    server = await serve(model, infer_fn, "0.0.0.0", PORT)
    print(f"FlashRT 运行时监听 {PORT}，OpenPI WebSocket，权重 {WEIGHTS_PATH}", flush=True)
    await server.serve_forever()


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
