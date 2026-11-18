"""vLLM 运行时。精度由权重目录里的量化配置决定，交付台不在这里改精度。"""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime_loader import resolve_fns

WEIGHTS_PATH = os.environ.get("WEIGHTS_PATH", "/weights/checkpoint")
PORT = int(os.environ.get("PORT", "8082"))


def load(weights_path: str):
    if not os.path.isdir(weights_path):
        raise FileNotFoundError(f"Qwen 权重需要是模型目录：{weights_path}")
    from vllm import LLM

    return LLM(model=weights_path)


def infer(model, request: dict) -> dict:
    text = str(request.get("text") or request.get("prompt") or "").strip()
    if not text:
        raise ValueError("需要 text")
    from vllm import SamplingParams

    params = SamplingParams(
        max_tokens=int(request.get("maxTokens") or 256),
        temperature=float(request.get("temperature") or 0),
    )
    outputs = model.generate([text], params)
    return {"text": outputs[0].outputs[0].text}


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
            self._json(200, self.infer_fn(self.processor, request))
        except ValueError as exc:
            self._json(400, {"detail": str(exc)})
        except Exception as exc:
            self._json(500, {"detail": str(exc)})

    def log_message(self, fmt, *args):
        print(f"[vllm] {self.address_string()} {fmt % args}", flush=True)

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
    print(f"vLLM 运行时监听 {PORT}，权重 {WEIGHTS_PATH}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
