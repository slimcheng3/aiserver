"""镜像构建放在后台。本机已经有这张镜像时不再构建。"""

from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path

from fastapi import HTTPException

from app.config import data_root
from app import docker_cli
from app.images import build_args, missing_sources, recipes
from app.runtimes import BASE_IMAGE, known_images, user_context
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
        job["logs"].append(line)
        if status:
            job["status"] = status
        _write(job)


def _run_build(job_id: str, recipe) -> int:
    missing = missing_sources(recipe)
    if missing:
        for line in missing:
            _append(job_id, line)
        _append(job_id, "构建停止")
        return 1
    code, output = docker_cli.run(build_args(recipe))
    if output:
        for line in output.splitlines():
            _append(job_id, line)
    return code


def execute(job_id: str) -> None:
    job = _read(job_id)
    if job is None:
        return
    image = job["image"]
    try:
        if image_present(image):
            _append(job_id, "镜像已在本机，跳过构建", "ready")
            return
        user_dir = user_context(image)
        if user_dir is not None:
            if not image_present(BASE_IMAGE):
                _append(job_id, f"本机还没有基础镜像 {BASE_IMAGE}，先构建它")
                base = recipes().get(BASE_IMAGE)
                if base is None:
                    _append(job_id, "没有基础镜像的构建方法", "failed")
                    return
                code = _run_build(job_id, base)
                if code != 0:
                    _append(job_id, "基础镜像构建失败", "failed")
                    return
            code, output = docker_cli.run(["build", "-f", str(user_dir / "Dockerfile"), "-t", image, str(user_dir)])
        else:
            recipe = recipes().get(image)
            if recipe is None:
                _append(job_id, "没有这份镜像的构建方法", "failed")
                return
            code = _run_build(job_id, recipe)
            output = ""
        if output:
            for line in output.splitlines():
                _append(job_id, line)
        _append(job_id, "构建完成" if code == 0 else "构建失败", "ready" if code == 0 else "failed")
    except Exception as exc:
        _append(job_id, str(exc), "failed")


def start_build(image: str) -> dict:
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
            "logs": [f"准备构建 {image}"],
            "createdAt": now_iso(),
            "updatedAt": now_iso(),
        }
        _write(job)
    if os.environ.get("AISERVER_BUILD_SYNC") == "1":
        execute(job["id"])
        return _read(job["id"]) or job
    threading.Thread(target=execute, args=(job["id"],), daemon=True).start()
    return job
