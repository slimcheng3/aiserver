"""SAM 3 运行时。镜像里的默认 load 和 infer。"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime_loader import resolve_fns

WEIGHTS_PATH = os.environ.get("WEIGHTS_PATH", "/weights/checkpoint")
PORT = int(os.environ.get("PORT", "8080"))


def load(weights_path: str):
    """按挂载进来的 checkpoint 加载 SAM 3，不从网络下载权重。"""
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"找不到 SAM 3 权重：{weights_path}")

    import torch
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    dtype_name = os.environ.get("WEIGHT_DTYPE", "")
    if dtype_name == "int8":
        blob = torch.load(weights_path, map_location="cpu", weights_only=True)
        return _load_int8(blob)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam3_image_model(
        checkpoint_path=weights_path,
        load_from_HF=False,
        device=device,
    )
    if dtype_name == "fp16":
        _cast_keeping_complex(model, torch.float16)
    elif dtype_name == "bf16":
        _cast_keeping_complex(model, torch.bfloat16)
    return Sam3Processor(model, device=device)


def _cast_keeping_complex(model, dtype):
    """只把浮点权重转成目标精度。旋转位置编码是复数，转成实数会丢掉虚部。"""
    import torch

    kept = []
    for module in model.modules():
        for name, buf in list(module.named_buffers(recurse=False)):
            if torch.is_tensor(buf) and buf.is_complex():
                kept.append((module, name, buf.detach().clone()))
    model.to(dtype=dtype)
    for module, name, buf in kept:
        module.register_buffer(name, buf)
    if kept:
        print(f"旋转位置编码保持复数 {len(kept)} 处", flush=True)
    return model


def _load_int8(blob: dict):
    import torch
    from torch import nn
    from torch.nn import functional as F
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    marker = blob.get("aiserver_quant") if isinstance(blob, dict) else None
    if not torch.is_tensor(marker) or int(marker.reshape(-1)[0]) != 8:
        raise RuntimeError("这份文件不是 int8 checkpoint")
    state = blob.get("state")
    scales = blob.get("scales")
    if not isinstance(state, dict) or not isinstance(scales, dict):
        raise RuntimeError("int8 checkpoint 缺少 state 或 scales")

    class _Int8Linear(nn.Module):
        def __init__(self, weight: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None):
            super().__init__()
            self.register_buffer("weight", weight)
            self.register_buffer("scale", scale)
            if bias is None:
                self.bias = None
            else:
                self.register_buffer("bias", bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            shape = x.shape
            flat = x.reshape(-1, shape[-1])
            if flat.is_cuda:
                out = _int8_gemm(flat, self.weight, self.scale)
            else:
                weight = self.weight.to(flat.dtype) * self.scale.to(flat.dtype)
                out = F.linear(flat, weight)
            if self.bias is not None:
                out = out + self.bias.to(dtype=out.dtype)
            return out.reshape(*shape[:-1], out.shape[-1])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam3_image_model(checkpoint_path=None, load_from_HF=False, device="cpu")
    replaced = _install_int8(model, state, scales, _Int8Linear)
    if replaced == 0:
        raise RuntimeError("int8 checkpoint 里没有能替换的线性层")
    print(f"int8 线性层 {replaced} 个", flush=True)
    _cast_keeping_complex(model, torch.float16)
    if device == "cuda":
        model = model.cuda()
    _index_int8(model)
    _patch_int8_ops()
    model.eval()
    return Sam3Processor(model, device=device)


_PROJ_WEIGHTS = {"in_proj_weight", "q_proj_weight", "k_proj_weight", "v_proj_weight"}
_INT8_SCALES: dict = {}
_ORIGINAL_LINEAR = None


def _install_int8(model, state: dict, scales: dict, layer_cls) -> int:
    import torch
    from torch import nn

    restored = {}
    linears = []
    projections = []
    for key, scale in scales.items():
        module_name, _, attr = key.rpartition(".")
        try:
            module = model.get_submodule(module_name) if module_name else model
        except AttributeError:
            module = None
        weight = state.get(key)
        if module is None or not torch.is_tensor(weight):
            continue
        if attr == "weight" and isinstance(module, nn.Linear):
            linears.append(key)
        elif attr in _PROJ_WEIGHTS and isinstance(getattr(module, attr, None), nn.Parameter):
            projections.append((module, attr, weight, scale))
        else:
            restored[key] = (weight.float() * scale.float()).half()

    float_state = {key: value for key, value in state.items() if key not in scales}
    float_state.update(restored)
    model.load_state_dict(float_state, strict=False)
    for key in linears:
        linear_name = key[: -len(".weight")]
        parent_name, _, child = linear_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, layer_cls(state[key], scales[key], state.get(linear_name + ".bias")))
    for module, attr, weight, scale in projections:
        module.register_parameter(attr, nn.Parameter(weight.contiguous(), requires_grad=False))
        module.register_buffer(attr + "_scale", scale.contiguous())
    return len(linears) + len(projections)


def _index_int8(model) -> None:
    import torch

    _INT8_SCALES.clear()
    for module in model.modules():
        weight = getattr(module, "weight", None)
        scale = getattr(module, "scale", None)
        if torch.is_tensor(weight) and weight.dtype == torch.int8 and torch.is_tensor(scale):
            _INT8_SCALES[weight.untyped_storage().data_ptr()] = scale
        for name, param in module.named_parameters(recurse=False):
            extra = getattr(module, name + "_scale", None)
            if param.dtype == torch.int8 and torch.is_tensor(extra):
                _INT8_SCALES[param.untyped_storage().data_ptr()] = extra


def _apply_int8(x, weight, scale, bias, activation):
    import torch
    from torch import nn
    from torch.nn import functional as F

    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    if scale.shape[0] != weight.shape[0]:
        # q/k/v 会把一整块 in_proj_weight 切成三段，比例要跟着这一段走。
        row = weight.storage_offset() // weight.stride(0)
        scale = scale[row : row + weight.shape[0]]
    if flat.is_cuda:
        out = _int8_gemm(flat, weight, scale)
    else:
        out = _ORIGINAL_LINEAR(flat, (weight.float() * scale.float()).to(dtype=flat.dtype))
    if bias is not None:
        out = out + bias.to(dtype=out.dtype)
    if activation in (F.relu, nn.ReLU):
        out = F.relu(out)
    elif activation in (F.gelu, nn.GELU):
        out = F.gelu(out, approximate="tanh")
    elif activation is not None:
        raise RuntimeError(f"int8 线性层不认识这个激活：{activation}")
    return out.reshape(*shape[:-1], out.shape[-1])


def _patch_int8_ops() -> None:
    global _ORIGINAL_LINEAR
    import torch
    from torch.nn import functional as F
    import sam3.model.vitdet as vitdet

    if _ORIGINAL_LINEAR is None:
        _ORIGINAL_LINEAR = F.linear

        def linear(inp, weight, bias=None):
            if weight.dtype == torch.int8:
                scale = _INT8_SCALES.get(weight.untyped_storage().data_ptr())
                if scale is None:
                    raise RuntimeError("int8 权重没有对应的比例")
                return _apply_int8(inp, weight, scale, bias, None)
            return _ORIGINAL_LINEAR(inp, weight, bias)

        F.linear = linear

    if getattr(vitdet.addmm_act, "_aiserver_int8", False):
        return
    original_addmm = vitdet.addmm_act

    def addmm_act(activation, linear, mat1):
        weight = getattr(linear, "weight", None)
        scale = getattr(linear, "scale", None)
        if torch.is_tensor(weight) and weight.dtype == torch.int8 and torch.is_tensor(scale):
            return _apply_int8(mat1, weight, scale, getattr(linear, "bias", None), activation)
        return original_addmm(activation, linear, mat1)

    addmm_act._aiserver_int8 = True
    vitdet.addmm_act = addmm_act


def _int8_gemm(x, weight, scale):
    import torch
    from torch.nn import functional as F

    rows, cols = x.shape
    outputs = weight.shape[0]
    limit = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    row_scale = limit / 127
    quantized = torch.clamp(torch.round(x / row_scale), -128, 127).to(torch.int8)
    # 这张卡上的 int8 乘要求行数大于 16，收缩维和输出维都是 8 的倍数。
    row_pad = 17 - rows if rows <= 16 else 0
    col_pad = (-cols) % 8
    out_pad = (-outputs) % 8
    packed = weight
    if col_pad:
        quantized = F.pad(quantized, (0, col_pad))
        packed = F.pad(packed, (0, col_pad))
    if out_pad:
        packed = F.pad(packed, (0, 0, 0, out_pad))
    if row_pad:
        quantized = F.pad(quantized, (0, 0, 0, row_pad))
    product = torch._int_mm(quantized.contiguous(), packed.t().contiguous())
    product = product[:rows, :outputs]
    # 整数累加和会到几十万，先乘比例再转回半精度，避免半精度直接溢出成 inf。
    return (product.float() * row_scale.float() * scale.float().view(1, -1)).to(dtype=x.dtype)


def infer(model, request: dict) -> dict:
    """对一张图和一个文本概念做分割。model 是 load 返回的 Sam3Processor。"""
    text = str(request.get("text") or "").strip()
    image_b64 = request.get("image")
    if not text or not image_b64:
        raise ValueError("需要 image（base64）和 text")
    if request.get("video"):
        raise ValueError("这份 serve.py 只接受图像和文本")

    threshold = request.get("threshold")
    if threshold is not None:
        model.confidence_threshold = float(threshold)

    image = _decode_image(image_b64)
    import torch

    # ViT MLP 的 fused addmm 会把中间结果打成 bfloat16；没有 autocast 时，
    # 下一层 Linear 仍是 float32，会报 mat1/mat2 dtype 不一致。
    # 量化后的 checkpoint 已经是 fp16 或 bf16，autocast 跟着权重走。
    if torch.cuda.is_available():
        dtype_name = os.environ.get("WEIGHT_DTYPE", "")
        autocast_dtype = torch.float16 if dtype_name in {"fp16", "int8"} else torch.bfloat16
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            state = model.set_text_prompt(prompt=text, state=model.set_image(image))
    else:
        state = model.set_text_prompt(prompt=text, state=model.set_image(image))
    return _pack(state)


def _decode_image(image_b64: str):
    from PIL import Image

    payload = image_b64.split(",", 1)[1] if "," in image_b64 else image_b64
    try:
        raw = base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise ValueError("image 不是合法的 base64") from exc
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _pack(state: dict) -> dict:
    from PIL import Image

    masks = state.get("masks")
    boxes = state.get("boxes")
    scores = state.get("scores")
    if masks is None or boxes is None or scores is None or masks.numel() == 0:
        return {"scores": [], "boxes": [], "masks": []}

    masks = masks.detach()
    while masks.ndim > 3:
        masks = masks.squeeze(1)
    encoded = []
    for mask in masks:
        image = Image.fromarray(mask.cpu().numpy().astype("uint8") * 255, mode="L")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
    return {
        "scores": scores.detach().float().cpu().tolist(),
        "boxes": boxes.detach().float().cpu().tolist(),
        "masks": encoded,
    }


class Handler(BaseHTTPRequestHandler):
    processor = None
    infer_fn = infer

    def do_GET(self):
        if self.path.rstrip("/") == "/health":
            self._json(200, {"status": "ok"})
            return
        self._json(404, {"detail": "未知路径"})

    def do_POST(self):
        if self.path.rstrip("/") != "/infer":
            self._json(404, {"detail": "未知路径"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length).decode("utf-8"))
            self._json(200, Handler.infer_fn(self.processor, request))
        except ValueError as exc:
            self._json(400, {"detail": str(exc)})
        except Exception as exc:
            traceback.print_exc()
            self._json(500, {"detail": str(exc)})

    def log_message(self, fmt, *args):
        print(f"[sam3] {self.address_string()} {fmt % args}", flush=True)

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    load_fn, infer_fn, pipeline = resolve_fns(WEIGHTS_PATH, load, infer)
    if pipeline:
        print(f"使用权重旁的 pipeline.py：{pipeline}", flush=True)
    Handler.infer_fn = infer_fn
    Handler.processor = load_fn(WEIGHTS_PATH)
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"SAM 3 运行时监听 {PORT}，权重 {WEIGHTS_PATH}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
