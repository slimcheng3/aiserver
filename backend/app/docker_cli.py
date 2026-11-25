"""对本机 Docker 的薄封装。测试可以换掉 run / stream / save。"""

from __future__ import annotations

import subprocess
import threading
from collections.abc import Callable, Iterator
from pathlib import Path


def run(args: list[str], cwd: Path | None = None) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            ["docker", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("这台机体没有 docker") from exc
    output = (completed.stdout or "") + (completed.stderr or "")
    return completed.returncode, output.strip()


def _visible(chunk: str) -> str:
    text = chunk.replace("\r\n", "\n").rstrip("\n")
    if "\r" in text:
        text = text.rsplit("\r", 1)[-1]
    return text.rstrip()


def stream(args: list[str], on_line: Callable[[str], None], cwd: Path | None = None) -> int:
    try:
        proc = subprocess.Popen(
            ["docker", *args],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("这台机体没有 docker") from exc
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = _visible(raw)
        if line:
            on_line(line)
    return int(proc.wait())


def save(image: str) -> Iterator[bytes]:
    """把本机镜像打成 docker save 的 tar，按块吐出。中途断开时停掉 docker。"""
    try:
        proc = subprocess.Popen(
            ["docker", "save", image],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("这台机体没有 docker") from exc
    assert proc.stdout is not None
    assert proc.stderr is not None
    errors: list[bytes] = []

    def drain() -> None:
        errors.append(proc.stderr.read() if proc.stderr is not None else b"")

    thread = threading.Thread(target=drain, daemon=True)
    thread.start()
    try:
        while True:
            chunk = proc.stdout.read(1024 * 1024)
            if not chunk:
                break
            yield chunk
        code = proc.wait()
        thread.join(timeout=5)
        if code != 0:
            text = b"".join(errors).decode("utf-8", "replace").strip()
            raise RuntimeError(text or "docker save 失败")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        thread.join(timeout=2)
