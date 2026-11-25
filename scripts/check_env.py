#!/usr/bin/env python3
"""启动交付台之前检查本机环境。有失败项时退出码为 1。"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIN_PYTHON = (3, 11)
MIN_NODE = 18


def run(argv: list[str], timeout: float = 8) -> subprocess.CompletedProcess[str] | None:
    if shutil.which(argv[0]) is None:
        return None
    try:
        return subprocess.run(
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def text_of(completed: subprocess.CompletedProcess[str] | None) -> str:
    if completed is None:
        return ""
    return ((completed.stdout or "") + (completed.stderr or "")).strip()


def first_line(completed: subprocess.CompletedProcess[str] | None) -> str:
    text = text_of(completed)
    return text.splitlines()[0].strip() if text else ""


class Report:
    def __init__(self) -> None:
        self.failed = 0
        self.warned = 0

    def ok(self, message: str) -> None:
        print(f"通过  {message}")

    def warn(self, message: str) -> None:
        self.warned += 1
        print(f"注意  {message}")

    def fail(self, message: str) -> None:
        self.failed += 1
        print(f"失败  {message}")


def version_tuple(raw: str) -> tuple[int, ...]:
    numbers: list[int] = []
    current = ""
    for char in raw:
        if char.isdigit():
            current += char
        elif current:
            numbers.append(int(current))
            current = ""
            if len(numbers) >= 3:
                break
    if current and len(numbers) < 3:
        numbers.append(int(current))
    return tuple(numbers)


def prefer_local_node() -> None:
    if shutil.which("node"):
        return
    tools = ROOT / ".tools"
    if not tools.is_dir():
        return
    for folder in sorted(tools.glob("node-*/bin")):
        node = folder / "node"
        if node.is_file() and os.access(node, os.X_OK):
            os.environ["PATH"] = str(folder) + os.pathsep + os.environ.get("PATH", "")
            return


def check_python(report: Report, ready: bool) -> None:
    current = sys.version_info[:3]
    shown = ".".join(str(part) for part in current)
    if current >= MIN_PYTHON:
        report.ok(f"Python {shown}")
    else:
        report.fail(f"Python {shown} 太旧，需要 {MIN_PYTHON[0]}.{MIN_PYTHON[1]} 或更新")
    venv_python = ROOT / ".venv" / "bin" / "python"
    if venv_python.is_file():
        report.ok(f"虚拟环境已在 {venv_python}")
    elif ready:
        report.fail("还没有 .venv。先执行 python3 -m venv .venv，再安装 backend/requirements.txt")
    else:
        report.warn("还没有 .venv。先执行 python3 -m venv .venv，再安装 backend/requirements.txt")


def check_node(report: Report, ready: bool) -> None:
    completed = run(["node", "--version"])
    if completed is None or completed.returncode != 0:
        report.fail("没有 node，需要 Node.js 18 或更新")
        return
    version = first_line(completed)
    major = version_tuple(version)
    if major and major[0] >= MIN_NODE:
        report.ok(f"Node.js {version}")
    else:
        report.fail(f"Node.js {version or '未知'} 太旧，需要 18 或更新")
    npm = run(["npm", "--version"])
    if npm is None or npm.returncode != 0:
        report.fail("没有 npm")
    else:
        report.ok(f"npm {first_line(npm)}")
    if (ROOT / "frontend" / "node_modules").is_dir():
        report.ok("前端依赖已安装")
    elif ready:
        report.fail("前端依赖还没装。在 frontend 目录执行 npm install")
    else:
        report.warn("前端依赖还没装。在 frontend 目录执行 npm install")


def check_docker(report: Report) -> None:
    if shutil.which("docker") is None:
        report.fail("没有 docker。构建和部署运行时都要它")
        return
    completed = run(["docker", "info", "--format", "{{.ServerVersion}}"], timeout=15)
    if completed is not None and completed.returncode == 0:
        report.ok(f"Docker {first_line(completed)}，当前用户可以调用")
        return
    detail = text_of(completed).lower()
    if "permission denied" in detail or "permission" in detail:
        report.fail("docker 存在，但当前用户没有权限。把用户加入 docker 组并重新登录，或用 sg docker 启动")
        return
    report.fail("docker 存在，但服务没有起来。先启动 Docker 再打开交付台")


def check_accelerator(report: Report) -> None:
    nvidia = run(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"])
    if nvidia is not None and nvidia.returncode == 0 and first_line(nvidia):
        names = [line.strip() for line in text_of(nvidia).splitlines() if line.strip()]
        report.ok("NVIDIA GPU：" + "；".join(names))
        return
    ascend = run(["npu-smi", "info"])
    if ascend is not None and ascend.returncode == 0:
        report.ok("检测到昇腾 NPU")
        return
    report.warn("没有检测到 NVIDIA 或昇腾。交付台可以打开，GPU 推理镜像跑不起来")


def check_ports(report: Report) -> None:
    for port, name in ((8787, "接口"), (5173, "页面")):
        with socket.socket() as sock:
            sock.settimeout(0.3)
            listening = sock.connect_ex(("127.0.0.1", port)) == 0
        if listening:
            report.warn(f"{port} 已有进程在听，{name}可能已经启动")
        else:
            report.ok(f"{port} 空闲，可以启动{name}")


def check_tree(report: Report) -> None:
    required = (
        ROOT / "backend" / "requirements.txt",
        ROOT / "backend" / "app" / "main.py",
        ROOT / "frontend" / "package.json",
    )
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        report.fail("不在交付台目录里，缺少 " + "、".join(missing))
    else:
        report.ok(f"代码目录 {ROOT}")


def main() -> int:
    ready = "--ready" in sys.argv[1:]
    prefer_local_node()
    report = Report()
    print("交付台环境检查")
    check_tree(report)
    check_python(report, ready)
    check_node(report, ready)
    check_docker(report)
    check_accelerator(report)
    check_ports(report)
    if report.failed:
        print(f"先处理 {report.failed} 个失败项，再启动。")
        return 1
    if report.warned:
        print(f"可以启动。还有 {report.warned} 项要注意。")
        return 0
    print("可以启动。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
