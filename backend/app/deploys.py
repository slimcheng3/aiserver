"""一个模型包一个容器。容器端口映射到宿主机上空闲的端口。"""

from __future__ import annotations

import json
import socket
from pathlib import Path

from fastapi import HTTPException

from app.config import data_root
from app import docker_cli
from app.packaging import get_package
from app.storage import now_iso


def _dir() -> Path:
    path = data_root() / "deploys"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _path(package_id: str) -> Path:
    return _dir() / f"{package_id}.json"


def _write(record: dict) -> dict:
    record["updatedAt"] = now_iso()
    _path(record["packageId"]).write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record


def _read(package_id: str) -> dict | None:
    path = _path(package_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_deploys() -> list[dict]:
    records = []
    for path in _dir().glob("pkg_*.json"):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    records.sort(key=lambda item: item.get("updatedAt", ""), reverse=True)
    return records


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _container(package_id: str) -> str:
    return f"aiserver-{package_id}"


def _endpoint(protocol: str, port: int) -> str:
    scheme = "ws" if protocol == "websocket" else "http"
    return f"{scheme}://127.0.0.1:{port}"


def _host_path(weight: dict) -> Path:
    raw = weight.get("localPath") or ""
    if not raw:
        raise HTTPException(status_code=400, detail=f"{weight.get('label') or weight.get('role')} 没有本机路径")
    path = Path(raw)
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"本机路径不存在：{path}")
    return path


def pipeline_beside(path: Path) -> Path | None:
    folder = path if path.is_dir() else path.parent
    found = folder / "pipeline.py"
    return found if found.is_file() else None


def _mounts(weights: list[dict]) -> tuple[list[tuple[str, str]], str, str | None]:
    primary = next((item for item in weights if item.get("role") != "cad"), weights[0])
    host = _host_path(primary)
    pipeline = pipeline_beside(host)
    mounts: list[tuple[str, str]] = []
    if host.is_dir():
        container_weights = "/weights/checkpoint"
        mounts.append((_docker_path(host), container_weights))
    else:
        container_weights = "/weights/" + host.name
        mounts.append((_docker_path(host.parent), "/weights"))
    used = {primary.get("role")}
    for weight in weights:
        if weight.get("role") in used:
            continue
        extra = _host_path(weight)
        target = "/weights/roles/" + str(weight.get("role"))
        mounts.append((_docker_path(extra if extra.is_dir() else extra.parent), target))
        used.add(weight.get("role"))
    return mounts, container_weights, str(pipeline) if pipeline else None


def _docker_path(path: Path) -> str:
    return path.resolve().as_posix()


def _run_args(record: dict, mounts: list[tuple[str, str]], container_weights: str) -> list[str]:
    args = [
        "run",
        "-d",
        "--name",
        record["container"],
        "-p",
        f"{record['hostPort']}:{record['containerPort']}",
        "-e",
        f"PORT={record['containerPort']}",
        "-e",
        f"WEIGHTS_PATH={container_weights}",
    ]
    for key, value in record.get("env") or {}.items():
        args.extend(["-e", f"{key}={value}"])
    if int(record.get("gpu") or 0) > 0:
        args.extend(["--gpus", "all"])
    for host, container in mounts:
        args.extend(["-v", f"{host}:{container}:ro"])
    args.append(record["image"])
    return args


def deploy(package_id: str, target: str) -> dict:
    if target not in {"local", "edge"}:
        raise HTTPException(status_code=400, detail="部署目标只能是本机或边侧")
    view = get_package(package_id)
    manifest = view.manifest
    engine = manifest["engine"]
    weights = manifest["weights"]
    existing = _read(package_id)
    if existing and existing.get("status") == "running":
        return existing
    if target == "edge":
        if any(weight.get("source") != "s3" for weight in weights):
            raise HTTPException(status_code=400, detail="部署到边侧时，权重都要使用 S3 地址")
        record = {
            "packageId": package_id,
            "name": manifest["metadata"]["name"],
            "target": "edge",
            "status": "prepared",
            "image": engine["image"],
            "protocol": engine["protocol"],
            "containerPort": engine["port"],
            "hostPort": None,
            "endpoint": None,
            "container": "",
            "pipeline": None,
            "command": [],
            "logs": [f"边侧使用已有镜像 {engine['image']}，权重保持 S3 地址。本机没有启动容器。"],
            "gpu": engine.get("resources", {}).get("gpu", 0),
            "env": engine.get("env") or {},
            "createdAt": now_iso(),
        }
        return _write(record)

    if any(weight.get("source") == "s3" for weight in weights):
        raise HTTPException(status_code=400, detail="本机部署使用本机路径。S3 地址用于边侧。")
    code, output = docker_cli.run(["image", "inspect", engine["image"]])
    if code != 0:
        detail = output or "镜像还没构建"
        raise HTTPException(status_code=400, detail=f"镜像还没构建：{engine['image']}。{detail}")
    mounts, container_weights, pipeline = _mounts(weights)
    host_port = _free_port()
    record = {
        "packageId": package_id,
        "name": manifest["metadata"]["name"],
        "target": "local",
        "status": "running",
        "image": engine["image"],
        "protocol": engine["protocol"],
        "containerPort": engine["port"],
        "hostPort": host_port,
        "endpoint": _endpoint(engine["protocol"], host_port),
        "container": _container(package_id),
        "pipeline": pipeline,
        "command": [],
        "logs": [f"使用已有镜像 {engine['image']}，不再构建"],
        "gpu": engine.get("resources", {}).get("gpu", 0),
        "env": engine.get("env") or {},
        "createdAt": now_iso(),
    }
    if pipeline:
        record["logs"].append(f"权重旁有 pipeline.py，容器里优先使用它：{pipeline}")
    args = _run_args(record, mounts, container_weights)
    record["command"] = ["docker", *args]
    docker_cli.run(["rm", "-f", record["container"]])
    code, output = docker_cli.run(args)
    record["logs"].append(output or "容器已启动")
    if code != 0:
        record["status"] = "failed"
        record["endpoint"] = None
        return _write(record)
    state_code, state = docker_cli.run(["inspect", "-f", "{{.State.Running}}", record["container"]])
    if state_code == 0 and state.strip().lower() == "false":
        _log_code, logs = docker_cli.run(["logs", record["container"]])
        record["status"] = "failed"
        record["endpoint"] = None
        if logs:
            record["logs"].append(logs)
    return _write(record)


def stop(package_id: str) -> dict:
    view = get_package(package_id)
    record = _read(package_id)
    if record is None:
        raise HTTPException(status_code=404, detail="这份模型包还没有部署")
    if record.get("container"):
        code, output = docker_cli.run(["rm", "-f", record["container"]])
        record["logs"].append(output or "已停止")
        if code != 0:
            record["status"] = "failed"
            return _write(record)
    record["status"] = "stopped"
    record["endpoint"] = None
    record["logs"].append("已停止。镜像和权重仍留在本机。")
    record["name"] = view.manifest["metadata"]["name"]
    return _write(record)
