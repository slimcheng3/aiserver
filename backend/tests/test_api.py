from __future__ import annotations

import hashlib
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


def client_with(tmp_path: Path, monkeypatch) -> TestClient:
    monkeypatch.setenv("AISERVER_DATA", str(tmp_path))
    return TestClient(app)


def test_catalog_has_four_models(tmp_path: Path, monkeypatch) -> None:
    client = client_with(tmp_path, monkeypatch)
    response = client.get("/api/models")
    assert response.status_code == 200
    models = {item["id"]: item for item in response.json()}
    assert set(models) == {"sam3", "genpose2", "pi05", "qwen"}
    assert models["sam3"]["artifacts"][0]["role"] == "checkpoint"
    assert models["pi05"]["engine"]["protocol"] == "websocket"
    assert models["qwen"]["artifacts"][0]["role"] == "model_dir"
    assert models["pi05"]["displayName"] == "π0.5"
    assert models["sam3"]["engine"]["codeRef"] == "engines/sam3"
    assert models["genpose2"]["engine"]["codeRef"] == "engines/genpose2"
    assert models["pi05"]["engine"]["image"] == "local/flashrt-engine:latest"
    assert models["pi05"]["engine"]["codeRef"] == "engines/flashrt"
    assert models["qwen"]["engine"]["image"] == "local/vllm-engine:latest"
    assert models["qwen"]["engine"]["codeRef"] == "engines/vllm"
    assert models["sam3"]["engine"]["models"] == ["sam3"]
    assert models["sam3"]["engine"]["weightQuants"] == ["fp16", "bf16"]
    assert "awq" in models["qwen"]["engine"]["weightQuants"]
    assert models["genpose2"]["artifacts"][1]["quantized"] is False


def test_local_weight_becomes_package(tmp_path: Path, monkeypatch) -> None:
    library = tmp_path / "library" / "local"
    library.mkdir(parents=True)
    payload = b"not-a-real-weight"
    (library / "demo.pt").write_bytes(payload)
    client = client_with(tmp_path, monkeypatch)

    assets = client.get("/api/weights").json()
    assert assets[0]["id"] == "local:demo.pt"
    assert assets[0]["sha256"] == hashlib.sha256(payload).hexdigest()

    created = client.post(
        "/api/packages",
        json={
            "modelId": "sam3",
            "name": "车间分割",
            "version": "1.0.0",
            "description": "本地权重",
            "weights": [{"role": "checkpoint", "assetId": "local:demo.pt", "quant": "fp16"}],
            "engine": _engine(),
        },
    )
    assert created.status_code == 200
    manifest = created.json()["manifest"]
    assert manifest["apiVersion"] == "aiserver.modelpackage/v1"
    assert manifest["model"]["id"] == "sam3"
    assert manifest["weights"][0]["uri"].startswith("file:")
    assert manifest["weights"][0]["sha256"] == hashlib.sha256(payload).hexdigest()
    assert manifest["engine"]["image"] == "local/sam3-engine:latest"
    pkg_id = manifest["metadata"]["id"]
    assert (tmp_path / "packages" / pkg_id / "manifest.json").exists()

    listed = client.get("/api/packages")
    assert listed.status_code == 200
    assert listed.json()[0]["manifest"]["metadata"]["id"] == pkg_id

    removed = client.delete(f"/api/packages/{pkg_id}")
    assert removed.status_code == 200
    assert client.get(f"/api/packages/{pkg_id}").status_code == 404


def test_s3_weight_is_referenced_not_downloaded(tmp_path: Path, monkeypatch) -> None:
    client = client_with(tmp_path, monkeypatch)
    registered = client.post(
        "/api/weights/s3",
        json={"uri": "s3://models/sam3/sam3.pt", "endpoint": "http://127.0.0.1:9000", "note": "预置"},
    )
    assert registered.status_code == 200
    asset = registered.json()
    assert asset["source"] == "s3"
    assert asset["bucket"] == "models"
    assert asset["key"] == "sam3/sam3.pt"

    created = client.post(
        "/api/packages",
        json={
            "modelId": "sam3",
            "name": "s3-sam3",
            "version": "0.1.0",
            "description": "",
            "weights": [{"role": "checkpoint", "assetId": asset["id"], "quant": "bf16"}],
            "engine": _engine(),
        },
    )
    assert created.status_code == 200
    weight = created.json()["manifest"]["weights"][0]
    assert weight["uri"] == "s3://models/sam3/sam3.pt"
    assert weight["quant"] == "bf16"
    assert weight["endpoint"] == "http://127.0.0.1:9000"
    assert weight["localPath"] is None
    assert created.json()["manifest"]["engine"]["models"] == ["sam3"]
    assert "bf16" in created.json()["manifest"]["engine"]["weightQuants"]


def test_upload_and_reject_missing_required_weight(tmp_path: Path, monkeypatch) -> None:
    client = client_with(tmp_path, monkeypatch)
    rejected = client.post(
        "/api/packages",
        json={
            "modelId": "qwen",
            "name": "缺权重",
            "version": "1",
            "description": "",
            "weights": [],
            "engine": {**_engine(), "image": "local/vllm-engine:latest", "port": 8082},
        },
    )
    assert rejected.status_code == 400

    uploaded = client.post(
        "/api/weights/upload",
        files={"file": ("qwen.safetensors", b"abc", "application/octet-stream")},
    )
    assert uploaded.status_code == 200
    asset_id = uploaded.json()["id"]
    created = client.post(
        "/api/packages",
        json={
            "modelId": "qwen",
            "name": "上传的千问",
            "version": "1",
            "description": "",
            "weights": [{"role": "model_dir", "assetId": asset_id, "quant": "fp16"}],
            "engine": {**_engine(), "image": "local/vllm-engine:latest", "port": 8082},
        },
    )
    assert created.status_code == 200
    deleted = client.delete("/api/weights", params={"id": asset_id})
    assert deleted.status_code == 200
    assert all(item["id"] != asset_id for item in client.get("/api/weights").json())


def test_quant_must_match_runtime(tmp_path: Path, monkeypatch) -> None:
    library = tmp_path / "library" / "local"
    library.mkdir(parents=True)
    (library / "demo.pt").write_bytes(b"weight")
    client = client_with(tmp_path, monkeypatch)

    rejected = client.post(
        "/api/packages",
        json={
            "modelId": "sam3",
            "name": "awq-sam3",
            "version": "1",
            "description": "",
            "weights": [{"role": "checkpoint", "assetId": "local:demo.pt", "quant": "awq"}],
            "engine": _engine(),
        },
    )
    assert rejected.status_code == 400

    missing = client.post(
        "/api/packages",
        json={
            "modelId": "qwen",
            "name": "没标精度",
            "version": "1",
            "description": "",
            "weights": [{"role": "model_dir", "assetId": "local:demo.pt"}],
            "engine": {**_engine(), "image": "local/vllm-engine:latest", "port": 8082},
        },
    )
    assert missing.status_code == 400

    created = client.post(
        "/api/packages",
        json={
            "modelId": "qwen",
            "name": "awq-qwen",
            "version": "1",
            "description": "",
            "weights": [{"role": "model_dir", "assetId": "local:demo.pt", "quant": "awq"}],
            "engine": {**_engine(), "image": "local/vllm-engine:latest", "port": 8082},
        },
    )
    assert created.status_code == 200
    manifest = created.json()["manifest"]
    assert manifest["weights"][0]["quant"] == "awq"
    assert manifest["engine"]["models"] == ["qwen"]
    assert "awq" in manifest["engine"]["weightQuants"]


def test_reject_disk_root_and_bad_s3(tmp_path: Path, monkeypatch) -> None:
    client = client_with(tmp_path, monkeypatch)
    root = client.post("/api/weights/local-path", json={"path": "C:\\", "note": ""})
    assert root.status_code == 400
    bad = client.post("/api/weights/s3", json={"uri": "http://bucket/key", "note": ""})
    assert bad.status_code == 400


def _engine() -> dict:
    return {
        "image": "local/sam3-engine:latest",
        "codeRef": "services/sam3",
        "command": ["python", "serve.py"],
        "port": 8080,
        "healthPath": "/health",
        "protocol": "http",
        "resources": {"gpu": 1, "gpuMemoryGi": 24, "cpu": 4, "memoryGi": 32},
        "env": {},
    }


_SERVE = """
def load(weights_path):
    return weights_path

def infer(model, request):
    return {"ok": True}

def main():
    return None
"""


def test_user_runtime_then_package(tmp_path: Path, monkeypatch) -> None:
    library = tmp_path / "library" / "local"
    library.mkdir(parents=True)
    (library / "demo.pt").write_bytes(b"weight")
    client = client_with(tmp_path, monkeypatch)
    rejected = client.post(
        "/api/runtimes",
        data={"name": "缺函数", "models": "sam3", "weightQuants": "fp16"},
        files={"serve": ("serve.py", "def load(path):\n    return path\n", "text/x-python")},
    )
    assert rejected.status_code == 400
    torch_req = client.post(
        "/api/runtimes",
        data={"name": "换 torch", "models": "sam3", "weightQuants": "fp16"},
        files={
            "serve": ("serve.py", _SERVE, "text/x-python"),
            "requirements": ("requirements.txt", "torch==2.4.0\n", "text/plain"),
        },
    )
    assert torch_req.status_code == 400
    created_runtime = client.post(
        "/api/runtimes",
        data={"name": "自定义分割", "models": "sam3", "weightQuants": "fp16"},
        files={
            "serve": ("serve.py", _SERVE, "text/x-python"),
            "requirements": ("requirements.txt", "pillow==10.4.0\n", "text/plain"),
        },
    )
    assert created_runtime.status_code == 200
    image = created_runtime.json()["image"]
    assert image.startswith("local/user-rt_")
    listed = client.get("/api/runtimes").json()
    assert any(item["image"] == image and item["kind"] == "user" for item in listed)
    assert any(item["kind"] == "base" for item in listed)

    mismatched = client.post(
        "/api/packages",
        json={
            "modelId": "sam3",
            "name": "不对的运行时",
            "version": "1",
            "description": "",
            "weights": [{"role": "checkpoint", "assetId": "local:demo.pt", "quant": "fp16"}],
            "engine": {**_engine(), "image": "local/not-a-runtime:latest"},
        },
    )
    assert mismatched.status_code == 400

    awq = client.post(
        "/api/packages",
        json={
            "modelId": "sam3",
            "name": "用户运行时不读 awq",
            "version": "1",
            "description": "",
            "weights": [{"role": "checkpoint", "assetId": "local:demo.pt", "quant": "awq"}],
            "engine": {**_engine(), "image": image},
        },
    )
    assert awq.status_code == 400

    created = client.post(
        "/api/packages",
        json={
            "modelId": "sam3",
            "name": "用户运行时",
            "version": "1",
            "description": "",
            "weights": [{"role": "checkpoint", "assetId": "local:demo.pt", "quant": "fp16"}],
            "engine": {**_engine(), "image": image, "port": 1, "command": ["nope"]},
        },
    )
    assert created.status_code == 200
    engine = created.json()["manifest"]["engine"]
    assert engine["image"] == image
    assert engine["port"] == 8080
    assert engine["command"] == ["python", "serve.py"]
    assert engine["weightQuants"] == ["fp16"]


def test_build_skips_when_image_exists_and_stops_without_source(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AISERVER_BUILD_SYNC", "1")
    monkeypatch.setenv("AISERVER_GENPOSE2_SRC", str(tmp_path / "no-genpose2"))
    client = client_with(tmp_path, monkeypatch)

    def present(args, cwd=None):
        if args[:2] == ["image", "inspect"]:
            return 0, "local"
        raise AssertionError(args)

    monkeypatch.setattr("app.docker_cli.run", present)
    ready = client.post("/api/builds", json={"image": "local/vllm-engine:latest"})
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    assert any("跳过构建" in line for line in ready.json()["logs"])

    def missing(args, cwd=None):
        if args[:2] == ["image", "inspect"]:
            return 1, "absent"
        raise AssertionError(args)

    monkeypatch.setattr("app.docker_cli.run", missing)
    failed = client.post("/api/builds", json={"image": "local/genpose2-engine:latest"})
    assert failed.status_code == 200
    assert failed.json()["status"] == "failed"
    assert any("找不到genpose2的模型代码" in line for line in failed.json()["logs"])


def test_sam3_build_uses_code_in_the_repo(tmp_path: Path, monkeypatch) -> None:
    from app.config import repo_root

    monkeypatch.setenv("AISERVER_BUILD_SYNC", "1")
    client = client_with(tmp_path, monkeypatch)
    builds: list[list[str]] = []

    def git_run(args, cwd=None):
        raise AssertionError(args)

    def docker_run(args, cwd=None):
        if args[:2] == ["image", "inspect"]:
            return 1, "absent"
        if args and args[0] == "build":
            builds.append(list(args))
            return 0, "built"
        raise AssertionError(args)

    monkeypatch.setattr("app.git_cli.run", git_run)
    monkeypatch.setattr("app.docker_cli.run", docker_run)
    job = client.post("/api/builds", json={"image": "local/sam3-engine:latest"})
    assert job.status_code == 200
    assert job.json()["status"] == "ready"
    assert builds
    assert not any(arg.startswith("sam3=") for arg in builds[0])
    assert (repo_root() / "engines" / "sam3" / "upstream" / "sam3" / "model_builder.py").is_file()


def test_user_runtime_build_builds_base_image_first(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AISERVER_BUILD_SYNC", "1")
    client = client_with(tmp_path, monkeypatch)
    created = client.post(
        "/api/runtimes",
        data={"name": "检测", "models": "yolo11", "weightQuants": "fp16"},
        files={"serve": ("serve.py", _SERVE, "text/x-python")},
    )
    image = created.json()["image"]
    calls: list[list[str]] = []

    def run(args, cwd=None):
        calls.append(list(args))
        if args[:2] == ["image", "inspect"]:
            return 1, "absent"
        if args and args[0] == "build":
            return 0, "built"
        return 0, ""

    monkeypatch.setattr("app.docker_cli.run", run)
    job = client.post("/api/builds", json={"image": image})
    assert job.status_code == 200
    assert job.json()["status"] == "ready"
    assert any("先构建它" in line for line in job.json()["logs"])
    builds = [args for args in calls if args and args[0] == "build"]
    assert len(builds) == 2
    assert "local/base-cuda128-torch:latest" in builds[0]
    assert image in builds[1]


def test_deploy_maps_host_port_and_edge_needs_s3(tmp_path: Path, monkeypatch) -> None:
    library = tmp_path / "library" / "local" / "ckpt"
    library.mkdir(parents=True)
    (library / "model.pt").write_bytes(b"weight")
    (library / "pipeline.py").write_text("def load(path):\n    return path\ndef infer(model, request):\n    return {}\n")
    client = client_with(tmp_path, monkeypatch)
    calls: list[list[str]] = []

    def run(args, cwd=None):
        calls.append(list(args))
        if args[:2] == ["image", "inspect"]:
            return 0, "ok"
        if args[:2] == ["rm", "-f"]:
            return 0, ""
        if args[:2] == ["inspect", "-f"]:
            return 0, "true"
        if args and args[0] == "run":
            return 0, "cid"
        return 0, ""

    monkeypatch.setattr("app.docker_cli.run", run)
    created = client.post(
        "/api/packages",
        json={
            "modelId": "sam3",
            "name": "本机包",
            "version": "1",
            "description": "",
            "weights": [{"role": "checkpoint", "assetId": "local:ckpt", "quant": "fp16"}],
            "engine": _engine(),
        },
    )
    assert created.status_code == 200
    package_id = created.json()["manifest"]["metadata"]["id"]
    edge = client.post("/api/deploys", json={"packageId": package_id, "target": "edge"})
    assert edge.status_code == 400

    local = client.post("/api/deploys", json={"packageId": package_id, "target": "local"})
    assert local.status_code == 200
    body = local.json()
    assert body["status"] == "running"
    assert body["hostPort"]
    assert body["endpoint"].startswith("http://127.0.0.1:")
    assert body["pipeline"]
    assert any(args and args[0] == "run" and "-p" in args for args in calls)

    registered = client.post("/api/weights/s3", json={"uri": "s3://models/sam3/sam3.pt", "note": ""})
    s3_package = client.post(
        "/api/packages",
        json={
            "modelId": "sam3",
            "name": "边侧包",
            "version": "1",
            "description": "",
            "weights": [{"role": "checkpoint", "assetId": registered.json()["id"], "quant": "fp16"}],
            "engine": _engine(),
        },
    )
    prepared = client.post(
        "/api/deploys",
        json={"packageId": s3_package.json()["manifest"]["metadata"]["id"], "target": "edge"},
    )
    assert prepared.status_code == 200
    assert prepared.json()["status"] == "prepared"
    assert prepared.json()["hostPort"] is None
    refused = client.post(
        "/api/deploys",
        json={"packageId": s3_package.json()["manifest"]["metadata"]["id"], "target": "local"},
    )
    assert refused.status_code == 400


def test_pipeline_beside_weights(tmp_path: Path) -> None:
    import importlib.util

    loader = Path(__file__).resolve().parents[2] / "engines" / "common" / "runtime_loader.py"
    spec = importlib.util.spec_from_file_location("runtime_loader_test", loader)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    folder = tmp_path / "weights"
    folder.mkdir()
    (folder / "model.pt").write_bytes(b"x")
    (folder / "pipeline.py").write_text("def load(path):\n    return 'custom'\ndef infer(model, request):\n    return {'model': model}\n")
    load, infer, found = module.resolve_fns(str(folder / "model.pt"), lambda path: "default", lambda model, request: {})
    assert load("ignored") == "custom"
    assert infer("custom", {}) == {"model": "custom"}
    assert found.endswith("pipeline.py")


def test_custom_runtime_accepts_a_new_model_id(tmp_path: Path, monkeypatch) -> None:
    library = tmp_path / "library" / "local"
    library.mkdir(parents=True)
    (library / "demo.pt").write_bytes(b"weight")
    client = client_with(tmp_path, monkeypatch)
    rejected = client.post(
        "/api/runtimes",
        data={"name": "坏 id", "models": "a/b", "weightQuants": "fp16"},
        files={"serve": ("serve.py", _SERVE, "text/x-python")},
    )
    assert rejected.status_code == 400

    created_runtime = client.post(
        "/api/runtimes",
        data={"name": "检测", "models": "SAM 3\nyolo11", "weightQuants": "fp16"},
        files={"serve": ("serve.py", _SERVE, "text/x-python")},
    )
    assert created_runtime.status_code == 200
    assert created_runtime.json()["models"] == ["sam3", "yolo11"]

    models = {item["id"]: item for item in client.get("/api/models").json()}
    assert models["yolo11"]["displayName"] == "yolo11"
    assert models["yolo11"]["artifacts"][0]["role"] == "checkpoint"
    assert models["yolo11"]["engine"]["image"] == created_runtime.json()["image"]

    created = client.post(
        "/api/packages",
        json={
            "modelId": "yolo11",
            "name": "自定义检测",
            "version": "1",
            "description": "",
            "weights": [{"role": "checkpoint", "assetId": "local:demo.pt", "quant": "fp16"}],
            "engine": {**_engine(), "image": created_runtime.json()["image"]},
        },
    )
    assert created.status_code == 200
    assert created.json()["manifest"]["model"]["id"] == "yolo11"


def test_reference_serve_sample_is_accepted(tmp_path: Path, monkeypatch) -> None:
    sample = Path(__file__).resolve().parents[2] / "frontend" / "public" / "samples" / "serve.py"
    client = client_with(tmp_path, monkeypatch)
    created = client.post(
        "/api/runtimes",
        data={"name": "样例", "models": "demo", "weightQuants": "fp16"},
        files={"serve": ("serve.py", sample.read_text(encoding="utf-8"), "text/x-python")},
    )
    assert created.status_code == 200
    assert created.json()["port"] == 8080
    assert created.json()["healthPath"] == "/health"


