"""自定义运行时的参考样例。

上传后，交付台把它存成 serve_user.py，再用生成的入口调用这里的 main。
启动前，如果权重旁边有 pipeline.py，入口会换掉本文件的 load 和 infer。

约定：
- load(weights_path) 读取部署时挂进来的权重，返回模型对象。不要在这里下载权重。
- infer(model, request) 接收 JSON 对象，返回 JSON 对象。
- main() 按基础镜像对外服务：HTTP，端口来自环境变量 PORT（默认 8080），GET /health。
- 推理入口是 POST /infer。
- 在 main 和请求处理里直接调用 load、infer 这两个名字。不要先存成局部变量，否则权重旁的 pipeline.py 换不掉它们。
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

WEIGHTS_PATH = os.environ.get("WEIGHTS_PATH", "/weights/checkpoint")
PORT = int(os.environ.get("PORT", "8080"))


def load(weights_path: str):
    """按挂载进来的路径加载模型。把下面的占位换成你的加载代码。"""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"找不到权重：{weights_path}")
    # 基础镜像里已经有 PyTorch。例如：
    # import torch
    # return torch.load(weights_path, map_location="cpu", weights_only=True)
    return {"weights_path": weights_path}


def infer(model, request: dict) -> dict:
    """做一次推理。request 是 POST /infer 的 JSON。返回值会原样写成 JSON。"""
    if not isinstance(request, dict):
        raise ValueError("请求体需要是 JSON 对象")
    return {
        "weights_path": model["weights_path"],
        "echo": request,
    }


class Handler(BaseHTTPRequestHandler):
    model = None

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
            raw = self.rfile.read(length).decode("utf-8")
            request = json.loads(raw) if raw else {}
            self._json(200, infer(self.model, request))
        except ValueError as exc:
            self._json(400, {"detail": str(exc)})
        except Exception as exc:
            self._json(500, {"detail": str(exc)})

    def log_message(self, fmt, *args):
        print(f"[runtime] {self.address_string()} {fmt % args}", flush=True)

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    Handler.model = load(WEIGHTS_PATH)
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"自定义运行时监听 {PORT}，权重 {WEIGHTS_PATH}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
