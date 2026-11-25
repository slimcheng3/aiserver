from __future__ import annotations

import hashlib
import io
import json
import socket
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


def client_with(tmp_path: Path, monkeypatch) -> TestClient:
    monkeypatch.setenv("AISERVER_DATA", str(tmp_path))
    return TestClient(app)


def patch_docker(monkeypatch, run_fn) -> None:
    monkeypatch.setattr("app.docker_cli.run", run_fn)

    def stream(args, on_line, cwd=None):
        code, output = run_fn(args, cwd)
        if output:
            for line in output.splitlines():
                on_line(line)
        return code

    monkeypatch.setattr("app.docker_cli.stream", stream)


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
    assert models["qwen"]["engine"]["port"] == 8000
    assert models["qwen"]["engine"]["command"][:2] == ["--model", "/weights/checkpoint"]
    assert "--gpu-memory-utilization" in models["qwen"]["engine"]["command"]
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
    assert manifest["weights"][0]["path"].endswith("demo.pt")
    assert set(manifest) == {"apiVersion", "kind", "metadata", "model", "engine", "weights"}
    assert manifest["model"] == {"id": "sam3"}
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
    assert weight["path"] == "s3://models/sam3/sam3.pt"
    assert weight["quant"] == "bf16"
    assert "endpoint" not in weight


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
    assert manifest["engine"]["image"] == "local/vllm-engine:latest"
    assert manifest["engine"]["port"] == 8000
    assert manifest["engine"]["command"][:2] == ["--model", "/weights/checkpoint"]

    custom = client.post(
        "/api/packages",
        json={
            "modelId": "qwen",
            "name": "qwen-half-gpu",
            "version": "1",
            "description": "",
            "weights": [{"role": "model_dir", "assetId": "local:demo.pt", "quant": "fp16"}],
            "engine": {
                **_engine(),
                "image": "local/vllm-engine:latest",
                "command": [
                    "--model",
                    "/weights/checkpoint",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "8000",
                    "--gpu-memory-utilization",
                    "0.5",
                ],
            },
        },
    )
    assert custom.status_code == 200
    assert custom.json()["manifest"]["engine"]["command"][-2:] == ["--gpu-memory-utilization", "0.5"]

    extras_only = client.post(
        "/api/packages",
        json={
            "modelId": "qwen",
            "name": "qwen-extra-only",
            "version": "1",
            "description": "",
            "weights": [{"role": "model_dir", "assetId": "local:demo.pt", "quant": "fp16"}],
            "engine": {
                **_engine(),
                "image": "local/vllm-engine:latest",
                "command": ["--gpu-memory-utilization", "0.4", "--max-model-len", "2048"],
            },
        },
    )
    assert extras_only.status_code == 200
    extras_cmd = extras_only.json()["manifest"]["engine"]["command"]
    assert extras_cmd[:6] == ["--model", "/weights/checkpoint", "--host", "0.0.0.0", "--port", "8000"]
    assert extras_cmd[-4:] == ["--gpu-memory-utilization", "0.4", "--max-model-len", "2048"]

    hijack = client.post(
        "/api/packages",
        json={
            "modelId": "qwen",
            "name": "qwen-lock-model",
            "version": "1",
            "description": "",
            "weights": [{"role": "model_dir", "assetId": "local:demo.pt", "quant": "fp16"}],
            "engine": {
                **_engine(),
                "image": "local/vllm-engine:latest",
                "command": ["--model", "/tmp/other", "--gpu-memory-utilization", "0.3"],
            },
        },
    )
    assert hijack.status_code == 200
    hijack_cmd = hijack.json()["manifest"]["engine"]["command"]
    assert hijack_cmd[:2] == ["--model", "/weights/checkpoint"]
    assert "/tmp/other" not in hijack_cmd


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
    assert "weightQuants" not in engine


def test_build_skips_when_image_exists(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AISERVER_BUILD_SYNC", "1")
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


def test_build_force_rebuilds_when_image_exists(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AISERVER_BUILD_SYNC", "1")
    client = client_with(tmp_path, monkeypatch)
    builds: list[list[str]] = []

    def present(args, cwd=None):
        if args[:2] == ["image", "inspect"]:
            return 0, "local"
        if args and args[0] == "build":
            builds.append(list(args))
            return 0, "built"
        raise AssertionError(args)

    patch_docker(monkeypatch, present)
    job = client.post("/api/builds", json={"image": "local/vllm-engine:latest", "force": True})
    assert job.status_code == 200
    assert job.json()["status"] == "ready"
    assert builds
    assert any("重新构建" in line for line in job.json()["logs"])
    assert not any("跳过构建" in line for line in job.json()["logs"])


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
    patch_docker(monkeypatch, docker_run)
    job = client.post("/api/builds", json={"image": "local/sam3-engine:latest"})
    assert job.status_code == 200
    assert job.json()["status"] == "ready"
    assert builds
    assert not any(arg.startswith("sam3=") for arg in builds[0])
    assert (repo_root() / "engines" / "sam3" / "upstream" / "sam3" / "model_builder.py").is_file()


def test_genpose2_build_uses_code_in_the_repo(tmp_path: Path, monkeypatch) -> None:
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
    patch_docker(monkeypatch, docker_run)
    job = client.post("/api/builds", json={"image": "local/genpose2-engine:latest"})
    assert job.status_code == 200
    assert job.json()["status"] == "ready"
    assert builds
    assert "--progress" in builds[0]
    assert not any(arg.startswith("genpose2=") for arg in builds[0])
    assert (repo_root() / "engines" / "genpose2" / "upstream" / "runners" / "infer.py").is_file()
    assert (repo_root() / "engines" / "genpose2" / "torch-hub" / "facebookresearch_dinov2_main" / "hubconf.py").is_file()
    assert (repo_root() / "engines" / "genpose2" / "torch-hub" / "checkpoints" / "dinov2_vits14_pretrain.pth").is_file()


def test_flashrt_build_uses_code_in_the_repo(tmp_path: Path, monkeypatch) -> None:
    from app.config import repo_root

    monkeypatch.setenv("AISERVER_BUILD_SYNC", "1")
    monkeypatch.setenv("AISERVER_FLASHRT_GPU_ARCH", "89")
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
    patch_docker(monkeypatch, docker_run)
    job = client.post("/api/builds", json={"image": "local/flashrt-engine:latest"})
    assert job.status_code == 200
    assert job.json()["status"] == "ready"
    assert builds
    assert "GPU_ARCH=89" in builds[0]
    assert not any(arg.startswith("flashrt=") for arg in builds[0])
    assert (repo_root() / "engines" / "flashrt" / "upstream" / "CMakeLists.txt").is_file()
    assert any("构建参数 GPU_ARCH=89" in line for line in job.json()["logs"])


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

    patch_docker(monkeypatch, run)
    job = client.post("/api/builds", json={"image": image})
    assert job.status_code == 200
    assert job.json()["status"] == "ready"
    assert any("先构建它" in line for line in job.json()["logs"])
    builds = [args for args in calls if args and args[0] == "build"]
    assert len(builds) == 2
    assert "local/base-cuda128-torch:latest" in builds[0]
    assert image in builds[1]


def test_registered_base_can_parent_a_runtime(tmp_path: Path, monkeypatch) -> None:
    client = client_with(tmp_path, monkeypatch)
    present = {"local/other-base:latest"}

    def run(args, cwd=None):
        if args[:2] == ["image", "inspect"]:
            return (0, "ok") if args[-1] in present else (1, "absent")
        if args and args[0] == "build":
            return 0, "built"
        return 0, ""

    patch_docker(monkeypatch, run)
    monkeypatch.setenv("AISERVER_BUILD_SYNC", "1")
    missing = client.post("/api/bases", json={"name": "别的 CUDA", "image": "local/missing-base:latest"})
    assert missing.status_code == 400
    assert "拉取结束" in missing.json()["detail"]
    bare = client.post("/api/bases", json={"name": "没标签", "image": "local/other-base"})
    assert bare.status_code == 400
    builtin = client.post("/api/bases", json={"name": "自带", "image": "local/base-cuda128-torch:latest"})
    assert builtin.status_code == 400

    created = client.post("/api/bases", json={"name": "别的 CUDA", "image": "local/other-base:latest", "note": "本机已有"})
    assert created.status_code == 200
    assert created.json()["base"]["kind"] == "base"
    assert created.json()["job"] is None
    assert any(item["image"] == "local/other-base:latest" for item in client.get("/api/runtimes").json())

    runtime = client.post(
        "/api/runtimes",
        data={"name": "检测", "models": "yolo11", "weightQuants": "fp16", "baseImage": "local/other-base:latest"},
        files={"serve": ("serve.py", _SERVE, "text/x-python")},
    )
    assert runtime.status_code == 200
    assert runtime.json()["baseImage"] == "local/other-base:latest"
    runtime_id = runtime.json()["id"]
    dockerfile = (tmp_path / "runtimes" / runtime_id / "Dockerfile").read_text(encoding="utf-8")
    assert dockerfile.startswith("# syntax=docker/dockerfile:1\nFROM local/other-base:latest\n")

    kept = client.delete("/api/bases", params={"image": "local/other-base:latest"})
    assert kept.status_code == 400
    shipped = client.delete("/api/bases", params={"image": "local/base-cuda128-torch:latest"})
    assert shipped.status_code == 400

    monkeypatch.setenv("AISERVER_BUILD_SYNC", "1")
    present.clear()
    job = client.post("/api/builds", json={"image": runtime.json()["image"]})
    assert job.status_code == 200
    assert job.json()["status"] == "failed"
    assert any("不在交付台的构建列表里" in line for line in job.json()["logs"])


def test_register_base_pulls_a_missing_image(tmp_path: Path, monkeypatch) -> None:
    client = client_with(tmp_path, monkeypatch)
    present: set[str] = set()
    pulled: list[str] = []

    def run(args, cwd=None):
        if args[:1] == ["pull"]:
            pulled.append(args[-1])
            present.add(args[-1])
            return 0, "Downloaded newer image"
        if args[:2] == ["image", "inspect"]:
            return (0, "ok") if args[-1] in present else (1, "absent")
        return 1, "no"

    patch_docker(monkeypatch, run)
    monkeypatch.setenv("AISERVER_BUILD_SYNC", "1")
    created = client.post(
        "/api/bases",
        json={"name": "NVIDIA PyTorch", "image": "nvcr.io/nvidia/pytorch:24.08-py3", "note": "外网"},
    )
    assert created.status_code == 200
    body = created.json()
    assert pulled == ["nvcr.io/nvidia/pytorch:24.08-py3"]
    assert body["base"]["image"] == "nvcr.io/nvidia/pytorch:24.08-py3"
    assert body["base"]["kind"] == "base"
    assert any("拉取完成" in line for line in body["job"]["logs"])
    assert any(item["image"] == "nvcr.io/nvidia/pytorch:24.08-py3" for item in client.get("/api/runtimes").json())


def test_register_base_reports_pull_denial(tmp_path: Path, monkeypatch) -> None:
    client = client_with(tmp_path, monkeypatch)

    def run(args, cwd=None):
        if args[:1] == ["pull"]:
            return 1, "denied: unauthorized"
        if args[:2] == ["image", "inspect"]:
            return 1, "absent"
        return 1, "no"

    patch_docker(monkeypatch, run)
    monkeypatch.setenv("AISERVER_BUILD_SYNC", "1")
    denied = client.post("/api/bases", json={"name": "拒绝", "image": "example.com/private:latest"})
    assert denied.status_code == 400
    assert "unauthorized" in denied.json()["detail"]
    assert all(item["image"] != "example.com/private:latest" for item in client.get("/api/runtimes").json())


def test_deploy_maps_host_port_and_rejects_s3_weights(tmp_path: Path, monkeypatch) -> None:
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
    local = client.post("/api/deploys", json={"packageId": package_id})
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
            "name": "S3 包",
            "version": "1",
            "description": "",
            "weights": [{"role": "checkpoint", "assetId": registered.json()["id"], "quant": "fp16"}],
            "engine": _engine(),
        },
    )
    refused = client.post(
        "/api/deploys",
        json={"packageId": s3_package.json()["manifest"]["metadata"]["id"]},
    )
    assert refused.status_code == 400


def test_deploy_uses_requested_host_port_or_rejects_busy(tmp_path: Path, monkeypatch) -> None:
    library = tmp_path / "library" / "local" / "ckpt"
    library.mkdir(parents=True)
    (library / "model.pt").write_bytes(b"weight")
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
            "name": "指定端口",
            "version": "1",
            "description": "",
            "weights": [{"role": "checkpoint", "assetId": "local:ckpt", "quant": "fp16"}],
            "engine": _engine(),
        },
    )
    package_id = created.json()["manifest"]["metadata"]["id"]
    local = client.post("/api/deploys", json={"packageId": package_id, "hostPort": 18080})
    assert local.status_code == 200
    body = local.json()
    assert body["hostPort"] == 18080
    assert body["endpoint"] == "http://127.0.0.1:18080"
    assert any(args and args[0] == "run" and "18080:8080" in args for args in calls)
    invalid = client.post("/api/deploys", json={"packageId": package_id, "hostPort": 70000})
    assert invalid.status_code == 422

    busy = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    busy.bind(("0.0.0.0", 0))
    busy.listen(1)
    occupied = int(busy.getsockname()[1])
    other = client.post(
        "/api/packages",
        json={
            "modelId": "sam3",
            "name": "占用端口",
            "version": "1",
            "description": "",
            "weights": [{"role": "checkpoint", "assetId": "local:ckpt", "quant": "fp16"}],
            "engine": _engine(),
        },
    )
    try:
        refused = client.post(
            "/api/deploys",
            json={"packageId": other.json()["manifest"]["metadata"]["id"], "hostPort": occupied},
        )
        assert refused.status_code == 400
        assert str(occupied) in refused.json()["detail"]
    finally:
        busy.close()


def test_deploy_list_marks_exited_container_failed(tmp_path: Path, monkeypatch) -> None:
    library = tmp_path / "library" / "local"
    library.mkdir(parents=True)
    (library / "demo.pt").write_bytes(b"weight")
    client = client_with(tmp_path, monkeypatch)
    alive = {"value": True}

    def run(args, cwd=None):
        if args[:2] == ["image", "inspect"]:
            return 0, "ok"
        if args[:2] == ["rm", "-f"]:
            return 0, ""
        if args[:2] == ["inspect", "-f"]:
            return (0, "true") if alive["value"] else (0, "false")
        if args[:1] == ["logs"]:
            return 0, "Traceback: boom"
        if args and args[0] == "run":
            return 0, "cid"
        return 0, ""

    monkeypatch.setattr("app.docker_cli.run", run)
    created = client.post(
        "/api/packages",
        json={
            "modelId": "sam3",
            "name": "退出包",
            "version": "1",
            "description": "",
            "weights": [{"role": "checkpoint", "assetId": "local:demo.pt", "quant": "fp16"}],
            "engine": _engine(),
        },
    )
    package_id = created.json()["manifest"]["metadata"]["id"]
    started = client.post("/api/deploys", json={"packageId": package_id})
    assert started.json()["status"] == "running"

    alive["value"] = False
    listed = client.get("/api/deploys")
    assert listed.status_code == 200
    body = listed.json()[0]
    assert body["status"] == "failed"
    assert body["endpoint"] is None
    assert any("容器已退出" in line for line in body["logs"])
    assert any("Traceback: boom" in line for line in body["logs"])

    alive["value"] = True
    again = client.post("/api/deploys", json={"packageId": package_id})
    assert again.status_code == 200
    assert again.json()["status"] == "running"


def test_deploy_infer_forwards_to_container(tmp_path: Path, monkeypatch) -> None:
    library = tmp_path / "library" / "local" / "ckpt"
    library.mkdir(parents=True)
    (library / "model.pt").write_bytes(b"weight")
    client = client_with(tmp_path, monkeypatch)

    def run(args, cwd=None):
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
            "name": "测试包",
            "version": "1",
            "description": "",
            "weights": [{"role": "checkpoint", "assetId": "local:ckpt", "quant": "fp16"}],
            "engine": _engine(),
        },
    )
    package_id = created.json()["manifest"]["metadata"]["id"]
    started = client.post("/api/deploys", json={"packageId": package_id, "hostPort": 18080})
    assert started.json()["status"] == "running"

    def urlopen(request, timeout=None):
        class Resp:
            status = 200

            def read(self):
                if request.get_method() == "GET":
                    return b'{"status":"ok"}'
                return b'{"scores":[0.91],"boxes":[[1,2,3,4]],"masks":[]}'

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        assert "18080" in request.full_url
        return Resp()

    monkeypatch.setattr("app.deploys.urllib.request.urlopen", urlopen)
    health = client.get(f"/api/deploys/{package_id}/health")
    assert health.status_code == 200
    assert health.json()["ok"] is True
    infer = client.post(f"/api/deploys/{package_id}/infer", json={"image": "xx", "text": "person"})
    assert infer.status_code == 200
    assert infer.json()["scores"] == [0.91]
    stopped = client.delete(f"/api/deploys/{package_id}")
    assert stopped.json()["status"] == "stopped"
    refused = client.post(f"/api/deploys/{package_id}/infer", json={"text": "person"})
    assert refused.status_code == 400


def test_vllm_deploy_rejects_infer_proxy(tmp_path: Path, monkeypatch) -> None:
    library = tmp_path / "library" / "local" / "qwen"
    library.mkdir(parents=True)
    (library / "config.json").write_text("{}", encoding="utf-8")
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
            "modelId": "qwen",
            "name": "千问",
            "version": "1",
            "description": "",
            "weights": [{"role": "model_dir", "assetId": "local:qwen", "quant": "fp16"}],
            "engine": {**_engine(), "image": "local/vllm-engine:latest", "port": 8000},
        },
    )
    package_id = created.json()["manifest"]["metadata"]["id"]
    started = client.post("/api/deploys", json={"packageId": package_id, "hostPort": 18082})
    assert started.status_code == 200
    run_args = next(args for args in calls if args and args[0] == "run")
    assert "--ipc" in run_args
    assert "--gpu-memory-utilization" in run_args
    assert "--model" in run_args
    infer = client.post(f"/api/deploys/{package_id}/infer", json={"text": "hi"})
    assert infer.status_code == 400
    assert "OpenAI" in infer.json()["detail"]

    def urlopen(request, timeout=None):
        class Resp:
            status = 200

            def read(self):
                if "/v1/models" in request.full_url:
                    return b'{"data":[{"id":"Qwen2.5-1.5B-Instruct"}]}'
                return b'{"choices":[{"message":{"content":"hi"}}]}'

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        assert "18082" in request.full_url
        return Resp()

    monkeypatch.setattr("app.deploys.urllib.request.urlopen", urlopen)
    chat = client.post(f"/api/deploys/{package_id}/chat", json={"prompt": "你好"})
    assert chat.status_code == 200
    assert chat.json()["choices"][0]["message"]["content"] == "hi"


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


def test_runtime_tar_exports_a_local_image(tmp_path: Path, monkeypatch) -> None:
    def run(args, cwd=None):
        if args[:2] == ["image", "inspect"] and args[2] == "local/flashrt-engine:latest":
            return 0, "ok"
        return 1, "missing"

    patch_docker(monkeypatch, run)
    saved: list[str] = []

    def save(image: str):
        saved.append(image)
        yield b"tar-bytes"

    monkeypatch.setattr("app.docker_cli.save", save)
    client = client_with(tmp_path, monkeypatch)

    unknown = client.get("/api/runtimes/tar", params={"image": "local/not-a-runtime:latest"})
    assert unknown.status_code == 404

    absent = client.get("/api/runtimes/tar", params={"image": "local/sam3-engine:latest"})
    assert absent.status_code == 404
    assert "还没构建" in absent.json()["detail"]

    ready = client.get("/api/runtimes/tar", params={"image": "local/flashrt-engine:latest"})
    assert ready.status_code == 200
    assert ready.headers["content-type"].startswith("application/zip")
    assert 'filename="local_flashrt-engine_latest.zip"' in ready.headers["content-disposition"]
    assert saved == ["local/flashrt-engine:latest"]
    bundle = zipfile.ZipFile(io.BytesIO(ready.content))
    assert bundle.namelist() == ["runtime.json", "runtime.tar"]
    assert bundle.read("runtime.tar") == b"tar-bytes"
    document = json.loads(bundle.read("runtime.json"))
    assert document["image"] == "local/flashrt-engine:latest"
    assert document["protocol"] == "websocket"
    assert document["models"] == ["pi05"]


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


