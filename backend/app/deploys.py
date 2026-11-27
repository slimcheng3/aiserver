"""一个模型包一个容器。容器端口映射到宿主机端口，可指定或自动分配。"""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from pathlib import Path

from fastapi import HTTPException

from app.config import data_root, repo_root
from app import docker_cli
from app.packaging import get_package
from app.storage import now_iso
from app.usage import attach as attach_usage


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


def _container_running(name: str) -> bool:
    code, output = docker_cli.run(["inspect", "-f", "{{.State.Running}}", name])
    return code == 0 and output.strip().lower() == "true"


def _mark_exited(record: dict) -> dict:
    name = str(record.get("container") or "")
    _, logs = docker_cli.run(["logs", "--tail", "80", name]) if name else (1, "")
    record["status"] = "failed"
    record["endpoint"] = None
    if logs:
        for line in logs.splitlines():
            record["logs"].append(line)
    record["logs"].append("容器已退出")
    return _write(record)


def _sync(record: dict) -> dict:
    if record.get("status") != "running" or record.get("target") != "local":
        return record
    name = str(record.get("container") or "")
    if not name:
        return record
    try:
        alive = _container_running(name)
    except RuntimeError as exc:
        record["status"] = "failed"
        record["endpoint"] = None
        record["logs"].append(str(exc))
        return _write(record)
    if alive:
        return record
    return _mark_exited(record)


def list_deploys() -> list[dict]:
    records = []
    for path in _dir().glob("pkg_*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        records.append(_sync(record))
    records.sort(key=lambda item: item.get("updatedAt", ""), reverse=True)
    attach_usage(records)
    return records


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("0.0.0.0", 0))
        return int(sock.getsockname()[1])


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("0.0.0.0", port))
        except OSError:
            return False
    return True


def _pick_host_port(requested: int | None) -> tuple[int, str]:
    if requested is None:
        port = _free_port()
        return port, f"宿主机端口 {port}（自动分配）"
    if not _port_free(requested):
        raise HTTPException(status_code=400, detail=f"宿主机端口 {requested} 已被占用")
    return requested, f"宿主机端口 {requested}（指定）"


def _container(package_id: str) -> str:
    return f"aiserver-{package_id}"


def _endpoint(protocol: str, port: int) -> str:
    scheme = "ws" if protocol == "websocket" else "http"
    return f"{scheme}://127.0.0.1:{port}"


def _weight_path(weight: dict) -> str:
    path = str(weight.get("path") or weight.get("localPath") or "")
    if path:
        return path
    return str(weight.get("uri") or "")


def _is_remote(weight: dict) -> bool:
    return weight.get("source") == "s3" or _weight_path(weight).startswith("s3://")


def _host_path(weight: dict) -> Path:
    raw = _weight_path(weight)
    if not raw or raw.startswith("s3://"):
        raise HTTPException(status_code=400, detail=f"{weight.get('role') or '权重'} 没有本机路径")
    path = Path(raw)
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"本机路径不存在：{path}")
    return path


def pipeline_beside(path: Path) -> Path | None:
    folder = path if path.is_dir() else path.parent
    found = folder / "pipeline.py"
    return found if found.is_file() else None


def _accelerator_count(engine: dict) -> int:
    resources = engine.get("resources") or {}
    if "accelerators" in resources:
        return int(resources.get("accelerators") or 0)
    return int(resources.get("gpu") or 1)


def _container_weights(engine: dict) -> str:
    path = str(engine.get("weightsPath") or "/weights/checkpoint").strip()
    if not path.startswith("/") or path == "/" or ".." in path.split("/"):
        raise HTTPException(status_code=400, detail="engine.weightsPath 不合法")
    return path


def _mounts(weights: list[dict], weights_path: str) -> tuple[list[tuple[str, str]], str, str | None]:
    primary = next((item for item in weights if item.get("role") != "cad"), weights[0])
    host = _host_path(primary)
    pipeline = pipeline_beside(host)
    mounts: list[tuple[str, str]] = [(_docker_path(host), weights_path)]
    container_weights = weights_path
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


def _weight_dtype(path: Path) -> str | None:
    name = path.name.lower()
    for quant in ("fp16", "bf16", "int8"):
        if name.endswith(f"-{quant}.pt") or name.endswith(f"-{quant}.pth"):
            return quant
    return None


def _run_args(
    record: dict,
    mounts: list[tuple[str, str]],
    container_weights: str,
    start_cmd: list[str] | None = None,
) -> list[str]:
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
    for key, value in (record.get("env") or {}).items():
        args.extend(["-e", f"{key}={value}"])
    if int(record.get("accelerators") or record.get("gpu") or 0) > 0:
        args.extend(["--gpus", "all"])
    if "vllm" in str(record.get("image") or ""):
        args.extend(["--ipc", "host"])
    for host, container in mounts:
        args.extend(["-v", f"{host}:{container}:ro"])
    args.append(record["image"])
    if start_cmd:
        args.extend(start_cmd)
    return args


def deploy(package_id: str, host_port: int | None = None) -> dict:
    view = get_package(package_id)
    manifest = view.manifest
    engine = manifest["engine"]
    weights = manifest["weights"]
    existing = _read(package_id)
    if existing:
        existing = _sync(existing)
    if existing and existing.get("status") == "running":
        return existing
    if any(_is_remote(weight) for weight in weights):
        raise HTTPException(status_code=400, detail="本机部署使用本机路径，不能用 S3 地址")
    code, output = docker_cli.run(["image", "inspect", engine["image"]])
    if code != 0:
        detail = output or "镜像还没构建"
        raise HTTPException(status_code=400, detail=f"镜像还没构建：{engine['image']}。{detail}")
    mounts, container_weights, pipeline = _mounts(weights, _container_weights(engine))
    host_port, port_note = _pick_host_port(host_port)
    record = {
        "packageId": package_id,
        "name": manifest["metadata"]["name"],
        "target": "local",
        "status": "running",
        "image": engine["image"],
        "protocol": engine["protocol"],
        "healthPath": engine.get("healthPath") or "/health",
        "containerPort": engine["port"],
        "hostPort": host_port,
        "endpoint": _endpoint(engine["protocol"], host_port),
        "container": _container(package_id),
        "pipeline": pipeline,
        "command": [],
        "logs": [f"使用已有镜像 {engine['image']}，不再构建", port_note],
        "accelerators": _accelerator_count(engine),
        "env": dict(engine.get("env") or {}),
        "createdAt": now_iso(),
    }
    dtype = _weight_dtype(Path(mounts[0][0])) if mounts else None
    if dtype:
        record["env"]["WEIGHT_DTYPE"] = dtype
        serve = repo_root() / "engines" / "sam3" / "serve.py"
        if serve.is_file() and "sam3" in str(engine.get("image") or ""):
            mounts.append((_docker_path(serve), "/opt/engine/serve.py"))
            record["logs"].append(f"权重是 {dtype}，加载后按这个精度放进显存")
    if pipeline:
        record["logs"].append(f"权重旁有 pipeline.py，容器里优先使用它：{pipeline}")
    args = _run_args(record, mounts, container_weights, list(engine.get("command") or []))
    record["command"] = ["docker", *args]
    docker_cli.run(["rm", "-f", record["container"]])
    code, output = docker_cli.run(args)
    record["logs"].append(output or "容器已启动")
    if code != 0:
        record["status"] = "failed"
        record["endpoint"] = None
        return _write(record)
    written = _write(record)
    return _sync(written)


def _live(package_id: str) -> dict:
    get_package(package_id)
    record = _read(package_id)
    if record:
        record = _sync(record)
    if not record or record.get("status") != "running" or not record.get("endpoint"):
        raise HTTPException(status_code=400, detail="这份模型包还没有在本机运行")
    return record


def _request(method: str, url: str, payload: dict | None, timeout: float) -> tuple[int, dict]:
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            body = json.loads(raw) if raw else {}
            if not isinstance(body, dict):
                body = {"result": body}
            return int(response.status), body
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        detail: object = raw or exc.reason
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and "detail" in parsed:
                detail = parsed["detail"]
            elif isinstance(parsed, dict):
                detail = parsed
        except json.JSONDecodeError:
            pass
        raise HTTPException(status_code=int(exc.code), detail=detail) from exc
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=502, detail=f"连不上容器：{exc.reason}") from exc
    except OSError as exc:
        raise HTTPException(status_code=503, detail=f"服务还在加载权重，稍后再做健康检查或推理：{exc}") from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail="容器返回的不是 JSON") from exc


def probe(package_id: str) -> dict:
    record = _live(package_id)
    path = str(record.get("healthPath") or "/health")
    if not path.startswith("/"):
        path = "/" + path
    _status, body = _request("GET", f"{str(record['endpoint']).rstrip('/')}{path}", None, 10)
    return {"ok": True, "endpoint": record["endpoint"], "body": body}


def call_chat(package_id: str, prompt: str, max_tokens: int = 256, temperature: float = 0) -> dict:
    record = _live(package_id)
    if "vllm" not in str(record.get("image") or ""):
        raise HTTPException(status_code=400, detail="这个运行时不是 OpenAI 接口，请用页面上的 /infer 测试")
    base = str(record["endpoint"]).rstrip("/")
    _status, models = _request("GET", f"{base}/v1/models", None, 10)
    rows = models.get("data") if isinstance(models.get("data"), list) else []
    model_id = rows[0].get("id") if rows and isinstance(rows[0], dict) else None
    if not model_id:
        raise HTTPException(status_code=502, detail="容器没有返回可用模型")
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    _status, body = _request("POST", f"{base}/v1/chat/completions", payload, 180)
    return body


def call_infer(package_id: str, payload: dict) -> dict:
    record = _live(package_id)
    if "vllm" in str(record.get("image") or ""):
        raise HTTPException(
            status_code=400,
            detail="这个运行时是官方 OpenAI 接口，请直连 /v1/chat/completions，不要走 /infer",
        )
    if str(record.get("protocol") or "http") != "http":
        raise HTTPException(status_code=400, detail="这个测试页只转发 HTTP /infer")
    _status, body = _request("POST", f"{str(record['endpoint']).rstrip('/')}/infer", payload, 300)
    return body


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
