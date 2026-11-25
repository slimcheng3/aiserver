"""SAM 3 运行时。镜像里的默认 load 和 infer。"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam3_image_model(
        checkpoint_path=weights_path,
        load_from_HF=False,
        device=device,
    )
    return Sam3Processor(model, device=device)


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
    if torch.cuda.is_available():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
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
