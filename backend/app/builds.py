"""镜像构建放在后台。普通构建在本机已有镜像时跳过；重新构建会再打一次。"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from pathlib import Path

_LAYER_LINE = re.compile(r"^(#\d+)\s+(sha256:[0-9a-f]+)\b")

from fastapi import HTTPException

from app.config import data_root, repo_root
from app import docker_cli, git_cli
from app.images import build_args, recipes
from app.runtimes import known_images, user_base_image, user_context
from app.storage import now_iso

_LOCK = threading.Lock()


def _dir() -> Path:
    path = data_root() / "builds"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _path(job_id: str) -> Path:
    return _dir() / f"{job_id}.json"


def _write(job: dict) -> dict:
    job["updatedAt"] = now_iso()
    _path(job["id"]).write_text(json.dumps(job, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return job


def _read(job_id: str) -> dict | None:
    path = _path(job_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_jobs(image: str = "") -> list[dict]:
    jobs = []
    for path in _dir().glob("bld_*.json"):
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if image and job.get("image") != image:
            continue
        jobs.append(job)
    jobs.sort(key=lambda item: item.get("createdAt", ""), reverse=True)
    return jobs


def image_present(image: str) -> bool:
    code, _output = docker_cli.run(["image", "inspect", image])
    return code == 0


def _append(job_id: str, line: str, status: str | None = None) -> None:
    with _LOCK:
        job = _read(job_id)
        if job is None:
            return
        logs = job["logs"]
        match = _LAYER_LINE.match(line)
        prev = _LAYER_LINE.match(logs[-1]) if logs else None
        if match and prev and match.groups() == prev.groups():
            logs[-1] = line
        else:
            logs.append(line)
        if len(job["logs"]) > 4000:
            job["logs"] = job["logs"][-3000:]
        if status:
            job["status"] = status
        _write(job)


def _has_code(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def _clone_destination_ok(path: Path) -> bool:
    root = repo_root().resolve()
    try:
        relative = path.resolve().relative_to(root)
    except ValueError:
        return True
    return bool(relative.parts) and relative.parts[0] == "tmp"


def _prepare_sources(job_id: str, recipe) -> int:
    for name, path in recipe.required.items():
        if _has_code(path):
            _append(job_id, f"{name} 的模型代码已在 {path}")
            continue
        url = recipe.clones.get(name, "")
        if not url:
            _append(job_id, f"找不到{name}的模型代码：{path}")
            _append(job_id, "构建停止")
            return 1
        if path.exists() and not path.is_dir():
            _append(job_id, f"找不到{name}的模型代码：{path}")
            _append(job_id, "构建停止")
            return 1
        if not _clone_destination_ok(path):
            _append(job_id, f"不能把 {name} clone 进交付台源码目录：{path}")
            _append(job_id, "构建停止")
            return 1
        _append(job_id, f"正在 clone {url} 到 {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            code, output = git_cli.run(["clone", "--depth", "1", url, str(path)])
        except RuntimeError as exc:
            _append(job_id, str(exc))
            _append(job_id, "构建停止")
            return 1
        if output:
            for line in output.splitlines():
                _append(job_id, line)
        if code != 0 or not _has_code(path):
            _append(job_id, "clone 失败")
            _append(job_id, "构建停止")
            return 1
    return 0


def _docker_build(job_id: str, args: list[str]) -> int:
    def on_line(line: str) -> None:
        _append(job_id, line)

    stream = getattr(docker_cli, "stream", None)
    if callable(stream):
        try:
            return int(stream(args, on_line))
        except TypeError:
            pass
    code, output = docker_cli.run(args)
    if output:
        for line in output.splitlines():
            _append(job_id, line)
    return code


def _run_build(job_id: str, recipe) -> int:
    if _prepare_sources(job_id, recipe) != 0:
        return 1
    for key, value in recipe.arguments.items():
        _append(job_id, f"构建参数 {key}={value}")
    return _docker_build(job_id, build_args(recipe))


def execute(job_id: str) -> None:
    job = _read(job_id)
    if job is None:
        return
    image = job["image"]
    try:
        if image_present(image) and not job.get("force"):
            _append(job_id, "镜像已在本机，跳过构建", "ready")
            return
        if job.get("force") and image_present(image):
            _append(job_id, "镜像已在本机，按重新构建再打一次。已有容器仍用旧镜像，重新部署后才会换上。")
        user_dir = user_context(image)
        if user_dir is not None:
            parent = user_base_image(user_dir)
            if not image_present(parent):
                base = recipes().get(parent)
                if base is None:
                    _append(
                        job_id,
                        f"本机还没有基础镜像 {parent}。它不在交付台的构建列表里，请先把镜像准备到本机。",
                        "failed",
                    )
                    return
                _append(job_id, f"本机还没有基础镜像 {parent}，先构建它")
                code = _run_build(job_id, base)
                if code != 0:
                    _append(job_id, "基础镜像构建失败", "failed")
                    return
            code = _docker_build(
                job_id,
                ["build", "--progress", "plain", "-f", str(user_dir / "Dockerfile"), "-t", image, str(user_dir)],
            )
        else:
            recipe = recipes().get(image)
            if recipe is None:
                _append(job_id, "没有这份镜像的构建方法", "failed")
                return
            code = _run_build(job_id, recipe)
        _append(job_id, "构建完成" if code == 0 else "构建失败", "ready" if code == 0 else "failed")
    except Exception as exc:
        _append(job_id, str(exc), "failed")


def start_image_pull(image: str, on_ready) -> dict:
    """本机没有这张镜像时从仓库拉取。成功后调用 on_ready。"""
    with _LOCK:
        for job in list_jobs(image):
            if job.get("action") == "pull" and job["status"] in {"queued", "running"}:
                return job
        job = {
            "id": "bld_" + uuid.uuid4().hex[:12],
            "image": image,
            "action": "pull",
            "status": "running",
            "force": False,
            "logs": [f"本机没有 {image}，正在从仓库拉取"],
            "createdAt": now_iso(),
            "updatedAt": now_iso(),
        }
        _write(job)
    if os.environ.get("AISERVER_BUILD_SYNC") == "1":
        _execute_pull(job["id"], on_ready)
        return _read(job["id"]) or job
    threading.Thread(target=_execute_pull, args=(job["id"], on_ready), daemon=True).start()
    return job


def _execute_pull(job_id: str, on_ready) -> None:
    job = _read(job_id)
    if job is None:
        return
    image = job["image"]
    try:
        code = _docker_build(job_id, ["pull", image])
        if code != 0:
            _append(job_id, "拉取失败", "failed")
            return
        if not image_present(image):
            _append(job_id, "拉取结束，但本机仍然没有这张镜像", "failed")
            return
        on_ready()
    except Exception as exc:
        detail = getattr(exc, "detail", None)
        _append(job_id, str(detail or exc), "failed")
        return
    _append(job_id, "拉取完成，已登记", "ready")


def start_build(image: str, force: bool = False) -> dict:
    image = image.strip()
    if not image:
        raise HTTPException(status_code=400, detail="请指定要构建的镜像")
    if image not in recipes() and image not in set(known_images()):
        raise HTTPException(status_code=400, detail="未知镜像")
    with _LOCK:
        for job in list_jobs(image):
            if job["status"] in {"queued", "running"}:
                return job
        job = {
            "id": "bld_" + uuid.uuid4().hex[:12],
            "image": image,
            "status": "running",
            "force": bool(force),
            "logs": [f"准备重新构建 {image}" if force else f"准备构建 {image}"],
            "createdAt": now_iso(),
            "updatedAt": now_iso(),
        }
        _write(job)
    if os.environ.get("AISERVER_BUILD_SYNC") == "1":
        execute(job["id"])
        return _read(job["id"]) or job
    threading.Thread(target=execute, args=(job["id"],), daemon=True).start()
    return job


def _build_process_running(image: str) -> bool:
    needle = image.encode("utf-8")
    for cmdline in Path("/proc").glob("*/cmdline"):
        try:
            raw = cmdline.read_bytes()
        except OSError:
            continue
        if b"docker" in raw and b"build" in raw and needle in raw:
            return True
    return False


def _reap_running(job_id: str) -> None:
    job = _read(job_id)
    if job is None or job.get("status") != "running":
        return
    image = job["image"]
    if not any("服务重启后继续等待" in line for line in job.get("logs", [])):
        _append(job_id, "服务重启后继续等待这次构建。这次 Docker 输出仍要等结束才写入；之后新的构建会逐行显示。")
    while _build_process_running(image):
        time.sleep(2)
        latest = _read(job_id)
        if latest is None or latest.get("status") != "running":
            return
    if image_present(image):
        _append(job_id, "构建完成", "ready")
    else:
        _append(job_id, "构建失败", "failed")


def resume_orphans() -> None:
    for job in list_jobs():
        if job.get("status") != "running":
            continue
        threading.Thread(target=_reap_running, args=(job["id"],), daemon=True).start()
