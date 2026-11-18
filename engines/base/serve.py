"""基础镜像的占位入口。不含模型推理，只回答健康检查。"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("PORT", "8080"))


def load(weights_path: str):
    raise RuntimeError("基础镜像只提供 CUDA、Python 和 PyTorch，不加载模型")


def infer(model, request: dict) -> dict:
    raise RuntimeError("基础镜像没有推理实现")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.rstrip("/") == "/health":
            self._json(200, {"status": "ok"})
            return
        self._json(404, {"detail": "未知路径"})

    def do_POST(self):
        self._json(501, {"detail": "基础镜像不提供推理。请用上传的 serve.py 构建新运行时。"})

    def log_message(self, fmt, *args):
        print(f"[base] {self.address_string()} {fmt % args}", flush=True)

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"基础镜像监听 {PORT}，没有模型", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
