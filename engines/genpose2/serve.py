"""GenPose2 运行时。镜像里的默认 load 和 infer。"""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime_loader import resolve_fns

WEIGHTS_PATH = os.environ.get("WEIGHTS_PATH", "/weights/checkpoint")
PORT = int(os.environ.get("PORT", "8081"))


def load(weights_path: str):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"找不到 GenPose2 权重：{weights_path}")
    import genpose2

    loader = getattr(genpose2, "load", None)
    if not callable(loader):
        raise RuntimeError("镜像里的 GenPose2 包没有 load(weights_path)")
    return loader(weights_path)


def infer(model, request: dict) -> dict:
    if not isinstance(request, dict):
        raise ValueError("请求需要是 JSON 对象")
    runner = getattr(model, "infer", None)
    if callable(runner):
        return runner(request)
    import genpose2

    fn = getattr(genpose2, "infer", None)
    if not callable(fn):
        raise RuntimeError("GenPose2 没有 infer(model, request)")
    return fn(model, request)


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
        print(f"[genpose2] {self.address_string()} {fmt % args}", flush=True)

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
    print(f"GenPose2 运行时监听 {PORT}，权重 {WEIGHTS_PATH}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
