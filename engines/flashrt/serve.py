"""FlashRT 上的 π0.5。WebSocket 收发 JSON，健康检查仍是 HTTP GET /health。"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime_loader import resolve_fns

WEIGHTS_PATH = os.environ.get("WEIGHTS_PATH", "/weights/checkpoint")
PORT = int(os.environ.get("PORT", "8000"))
_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def load(weights_path: str):
    if not os.path.isdir(weights_path):
        raise FileNotFoundError(f"π0.5 的权重需要是 checkpoint 目录：{weights_path}")
    import flashrt

    loader = getattr(flashrt, "load", None)
    if not callable(loader):
        raise RuntimeError("镜像里的 FlashRT 没有 load(weights_path)")
    return loader(weights_path)


def infer(model, request: dict) -> dict:
    if not isinstance(request, dict):
        raise ValueError("请求需要是 JSON 对象")
    runner = getattr(model, "infer", None)
    if callable(runner):
        return runner(request)
    import flashrt

    fn = getattr(flashrt, "infer", None)
    if not callable(fn):
        raise RuntimeError("FlashRT 没有 infer(model, request)")
    return fn(model, request)


def accept_key(key: str) -> str:
    digest = hashlib.sha1((key + _GUID).encode("ascii")).digest()
    return base64.b64encode(digest).decode("ascii")


class _Reader:
    def __init__(self, conn: socket.socket, pending: bytes = b"") -> None:
        self.conn = conn
        self.buf = bytearray(pending)

    def read(self, size: int) -> bytes:
        while len(self.buf) < size:
            piece = self.conn.recv(max(4096, size - len(self.buf)))
            if not piece:
                raise ConnectionError("连接已关闭")
            self.buf.extend(piece)
        out = bytes(self.buf[:size])
        del self.buf[:size]
        return out


def read_text(reader: _Reader) -> str | None:
    header = reader.read(2)
    opcode = header[0] & 0x0F
    if opcode == 0x8:
        return None
    length = header[1] & 0x7F
    if length == 126:
        length = struct.unpack("!H", reader.read(2))[0]
    elif length == 127:
        length = struct.unpack("!Q", reader.read(8))[0]
    mask = reader.read(4) if header[1] & 0x80 else b""
    payload = reader.read(length)
    if mask:
        payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
    if opcode != 0x1:
        return ""
    return payload.decode("utf-8")


def send_text(conn: socket.socket, text: str) -> None:
    data = text.encode("utf-8")
    if len(data) < 126:
        header = bytes([0x81, len(data)])
    elif len(data) < 65536:
        header = bytes([0x81, 126]) + struct.pack("!H", len(data))
    else:
        header = bytes([0x81, 127]) + struct.pack("!Q", len(data))
    conn.sendall(header + data)


def handle(conn: socket.socket, model, infer_fn) -> None:
    raw = b""
    while b"\r\n\r\n" not in raw:
        piece = conn.recv(4096)
        if not piece:
            return
        raw += piece
        if len(raw) > 65536:
            return
    head, _, rest = raw.partition(b"\r\n\r\n")
    text = head.decode("latin1")
    first = text.split("\r\n", 1)[0]
    if first.startswith("GET /health"):
        body = b'{"status":"ok"}'
        conn.sendall(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
            + str(len(body)).encode()
            + b"\r\nConnection: close\r\n\r\n"
            + body
        )
        return
    headers = {}
    for line in text.split("\r\n")[1:]:
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()
    key = headers.get("sec-websocket-key", "")
    if not key:
        conn.sendall(b"HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n")
        return
    accept = accept_key(key)
    conn.sendall(
        (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
        ).encode("ascii")
    )
    reader = _Reader(conn, rest)
    while True:
        try:
            message = read_text(reader)
        except ConnectionError:
            return
        if message is None:
            return
        try:
            request = json.loads(message) if message else {}
            result = infer_fn(model, request)
            send_text(conn, json.dumps(result, ensure_ascii=False))
        except Exception as exc:
            send_text(conn, json.dumps({"detail": str(exc)}, ensure_ascii=False))


def main() -> None:
    load_fn, infer_fn, pipeline = resolve_fns(WEIGHTS_PATH, load, infer)
    if pipeline:
        print(f"使用权重旁的 pipeline.py：{pipeline}", flush=True)
    model = load_fn(WEIGHTS_PATH)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", PORT))
    server.listen()
    print(f"FlashRT 运行时监听 {PORT}，权重 {WEIGHTS_PATH}", flush=True)
    while True:
        conn, _addr = server.accept()
        try:
            handle(conn, model, infer_fn)
        finally:
            conn.close()


if __name__ == "__main__":
    main()
