"""看这台机体有没有 Docker、NVIDIA 或昇腾，以及哪些镜像已经在本地。"""

from __future__ import annotations

import shutil
import subprocess

from app import docker_cli
from app.runtimes import known_images


def _probe(argv: list[str]) -> dict[str, str | bool]:
    if shutil.which(argv[0]) is None:
        return {"ok": False, "detail": f"没有 {argv[0]}"}
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "detail": str(exc)}
    text = ((completed.stdout or "") + (completed.stderr or "")).strip()
    first = text.splitlines()[0] if text else ""
    return {"ok": completed.returncode == 0, "detail": first}


def environment() -> dict:
    docker = _probe(["docker", "version", "--format", "{{.Server.Version}}"])
    nvidia = _probe(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    ascend = _probe(["npu-smi", "info"])
    images: dict[str, bool] = {}
    for image in known_images():
        if not docker["ok"]:
            images[image] = False
            continue
        code, _output = docker_cli.run(["image", "inspect", image])
        images[image] = code == 0
    if nvidia["ok"]:
        device = "cuda"
    elif ascend["ok"]:
        device = "ascend"
    else:
        device = "none"
    return {"docker": docker, "nvidia": nvidia, "ascend": ascend, "device": device, "images": images}
