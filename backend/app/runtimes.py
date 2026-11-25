"""内置运行时来自模型目录。用户运行时是基础镜像加上上传的 serve.py。"""

from __future__ import annotations

import ast
import json
import re
import shutil
import uuid
from pathlib import Path

from fastapi import HTTPException

from app import docker_cli
from app.catalog import load_models
from app.config import data_root, repo_root
from app.storage import now_iso

BASE_IMAGE = "local/base-cuda128-torch:latest"
_IMAGE_NAME = re.compile(
    r"(?:[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?(?::[0-9]+)?/)?"
    r"[a-z0-9]+(?:[._-][a-z0-9]+)*(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)*"
)
_IMAGE_TAG = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}")
_QUANTS: set[str] = {"fp16", "bf16", "int8", "awq", "gptq", "fp8"}
_FORBIDDEN = {"torch", "torchvision", "torchaudio", "torchtext"}

WRAPPER = '''"""交付台生成的入口。权重旁有 pipeline.py 时改用它的 load 和 infer。"""

import os

import serve_user as user
from runtime_loader import resolve_fns


def main() -> None:
    weights = os.environ.get("WEIGHTS_PATH", "/weights/checkpoint")
    load_fn, infer_fn, pipeline = resolve_fns(weights, user.load, user.infer)
    user.load = load_fn
    user.infer = infer_fn
    if pipeline:
        print(f"使用权重旁的 pipeline.py：{pipeline}", flush=True)
    user.main()


if __name__ == "__main__":
    main()
'''

def user_dockerfile(base_image: str) -> str:
    return f"""# syntax=docker/dockerfile:1
FROM {base_image}
COPY requirements.txt serve_user.py runtime_loader.py serve.py /opt/engine/
RUN python -m pip install --no-cache-dir --break-system-packages -r /opt/engine/requirements.txt
WORKDIR /opt/engine
ENV WEIGHTS_PATH=/weights/checkpoint PORT=8080
EXPOSE 8080
CMD ["python", "serve.py"]
"""


def _folder(runtime_id: str) -> Path:
    return data_root() / "runtimes" / runtime_id


def _read_user(folder: Path) -> dict:
    return json.loads((folder / "meta.json").read_text(encoding="utf-8"))


def check_serve(source: str) -> None:
    if len(source.encode("utf-8")) > 1_000_000:
        raise HTTPException(status_code=400, detail="serve.py 过大")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise HTTPException(status_code=400, detail=f"serve.py 无法解析：{exc.msg}") from exc
    names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    missing = [name for name in ("load", "infer", "main") if name not in names]
    if missing:
        raise HTTPException(status_code=400, detail="serve.py 需要实现函数：" + "、".join(missing))


def check_requirements(text: str) -> str:
    if len(text.encode("utf-8")) > 64_000:
        raise HTTPException(status_code=400, detail="requirements.txt 过大")
    kept: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-") or "://" in line:
            raise HTTPException(status_code=400, detail="requirements.txt 只写钉死版本的包，不写选项和地址")
        if "==" not in line:
            raise HTTPException(status_code=400, detail=f"请钉死版本：{line}")
        name = line.split("==", 1)[0].strip().lower().replace("_", "-")
        name = name.split("[", 1)[0]
        if name in _FORBIDDEN or name.startswith("nvidia-"):
            raise HTTPException(status_code=400, detail="PyTorch 和 CUDA 来自基础镜像，不能在 requirements.txt 里更换")
        kept.append(line)
    return ("\n".join(kept) + "\n") if kept else ""


def _builtin(model) -> dict:
    engine = model.engine
    return {
        "id": engine.image,
        "name": engine.image,
        "image": engine.image,
        "kind": "builtin",
        "codeRef": engine.codeRef,
        "command": list(engine.command),
        "port": engine.port,
        "healthPath": engine.healthPath,
        "protocol": engine.protocol,
        "models": list(engine.models),
        "weightQuants": list(engine.weightQuants),
        "baseImage": "",
        "note": "",
        "createdAt": "",
    }


def _builtin_base() -> dict:
    return {
        "id": BASE_IMAGE,
        "name": BASE_IMAGE,
        "image": BASE_IMAGE,
        "kind": "base",
        "codeRef": "engines/base",
        "command": ["python", "serve.py"],
        "port": 8080,
        "healthPath": "/health",
        "protocol": "http",
        "models": [],
        "weightQuants": [],
        "baseImage": "",
        "note": "只有 CUDA 12.8、Python 和 PyTorch。交付台可以构建这张镜像。",
        "createdAt": "",
    }


def _bases_root() -> Path:
    return data_root() / "bases"


def _base_records() -> list[dict]:
    records = [_builtin_base()]
    root = _bases_root()
    if root.exists():
        for folder in sorted(root.iterdir()):
            meta = folder / "meta.json"
            if not folder.is_dir() or not meta.exists():
                continue
            try:
                item = json.loads(meta.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if item.get("kind") == "base" and item.get("image"):
                records.append(item)
    return records


def check_image_ref(image: str) -> str:
    image = image.strip()
    if not image or len(image) > 255 or any(ch.isspace() for ch in image):
        raise HTTPException(status_code=400, detail="镜像名需要 1 到 255 个字符，不能有空格")
    if "@" in image:
        name, _, digest = image.partition("@")
        if not _IMAGE_NAME.fullmatch(name) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            raise HTTPException(status_code=400, detail="摘要需要是 sha256: 加 64 位十六进制，仓库名要小写")
        return image
    name, sep, tag = image.rpartition(":")
    if not sep or not tag or not _IMAGE_TAG.fullmatch(tag) or not _IMAGE_NAME.fullmatch(name):
        raise HTTPException(status_code=400, detail="镜像名要带标签，仓库部分小写，例如 local/my-base:latest")
    return image


def _pull_failure(logs: list[str]) -> str:
    for line in reversed(logs):
        if line and line != "拉取失败":
            return line
    return "拉取失败"


def register_base(name: str, image: str, note: str) -> dict:
    name = name.strip()
    note = note.strip()
    if not name or len(name) > 80:
        raise HTTPException(status_code=400, detail="基础镜像名称需要 1 到 80 个字符")
    if len(note) > 200:
        raise HTTPException(status_code=400, detail="说明不要超过 200 个字符")
    image = check_image_ref(image)
    known = {item["image"] for item in list_runtimes()}
    if image in known:
        raise HTTPException(status_code=400, detail="这张镜像已经在运行时列表里")
    try:
        code, _output = docker_cli.run(["image", "inspect", image])
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if code == 0:
        return {"base": _save_base(name, image, note), "job": None}
    from app.builds import start_image_pull

    def finish() -> None:
        if any(item["image"] == image for item in _base_records()):
            return
        _save_base(name, image, note)

    job = start_image_pull(image, finish)
    if job.get("status") == "failed":
        raise HTTPException(status_code=400, detail=_pull_failure(job.get("logs") or []))
    if job.get("status") == "ready":
        base = next(item for item in _base_records() if item["image"] == image)
        return {"base": base, "job": job}
    return {"base": None, "job": job}


def _save_base(name: str, image: str, note: str) -> dict:
    base_id = "base_" + uuid.uuid4().hex[:8]
    folder = _bases_root() / base_id
    folder.mkdir(parents=True, exist_ok=False)
    meta = {
        "id": base_id,
        "name": name,
        "image": image,
        "kind": "base",
        "codeRef": "",
        "command": ["python", "serve.py"],
        "port": 8080,
        "healthPath": "/health",
        "protocol": "http",
        "models": [],
        "weightQuants": [],
        "baseImage": "",
        "note": note,
        "createdAt": now_iso(),
    }
    (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def delete_base(image: str) -> None:
    image = image.strip()
    if image == BASE_IMAGE:
        raise HTTPException(status_code=400, detail="交付台自带的基础镜像不能删除")
    folder = None
    root = _bases_root()
    if root.exists():
        for item in root.iterdir():
            meta_path = item / "meta.json"
            if not meta_path.is_file():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if meta.get("image") == image:
                folder = item
                break
    if folder is None:
        raise HTTPException(status_code=404, detail="没有这张基础镜像")
    used = [runtime["name"] for runtime in list_runtimes() if runtime["kind"] == "user" and runtime.get("baseImage") == image]
    if used:
        raise HTTPException(status_code=400, detail="还有运行时用着它：" + "、".join(used))
    shutil.rmtree(folder)


def list_runtimes() -> list[dict]:
    grouped: dict[str, dict] = {}
    order: list[str] = []
    for model in load_models():
        image = model.engine.image
        current = grouped.get(image)
        if current is None:
            grouped[image] = _builtin(model)
            order.append(image)
            continue
        for model_id in model.engine.models:
            if model_id not in current["models"]:
                current["models"].append(model_id)
        for quant in model.engine.weightQuants:
            if quant not in current["weightQuants"]:
                current["weightQuants"].append(quant)
    users: list[dict] = []
    root = data_root() / "runtimes"
    if root.exists():
        for folder in sorted(root.iterdir()):
            meta = folder / "meta.json"
            if folder.is_dir() and meta.exists():
                try:
                    users.append(_read_user(folder))
                except (OSError, json.JSONDecodeError):
                    continue
    return [*_base_records(), *[grouped[image] for image in order], *users]


def find_runtime(model_id: str, image: str) -> dict:
    for runtime in list_runtimes():
        if runtime["kind"] == "base":
            continue
        if runtime["image"] == image and model_id in runtime["models"]:
            return runtime
    raise HTTPException(status_code=400, detail="这个模型没有这份运行时")


def normalize_model_ids(raw: list[str]) -> list[str]:
    catalog = load_models()
    by_id = {model.id.casefold(): model.id for model in catalog}
    by_name = {model.displayName.casefold(): model.id for model in catalog}
    found: list[str] = []
    for item in raw:
        token = " ".join(item.split())
        if not token:
            continue
        key = token.casefold()
        if key in by_id:
            token = by_id[key]
        elif key in by_name:
            token = by_name[key]
        elif len(token) > 64 or any(char in token for char in '\\/:*?"<>|,'):
            raise HTTPException(status_code=400, detail=f"模型 id 不合法：{token}")
        if token not in found:
            found.append(token)
    if not found:
        raise HTTPException(status_code=400, detail="请填写这份运行时可以加载的模型")
    return found


def create_user_runtime(
    name: str,
    model_ids: list[str],
    quants: list[str],
    serve_source: str,
    requirements: str,
    base_image: str = "",
) -> dict:
    name = name.strip()
    if not name or len(name) > 80:
        raise HTTPException(status_code=400, detail="运行时名称需要 1 到 80 个字符")
    model_ids = normalize_model_ids(model_ids)
    if not quants or any(quant not in _QUANTS for quant in quants):
        raise HTTPException(status_code=400, detail="请选择这份运行时能读的权重精度")
    if len(quants) != len(set(quants)):
        raise HTTPException(status_code=400, detail="同一个权重精度不用选两次")
    check_serve(serve_source)
    pinned = check_requirements(requirements)
    parent = (base_image or BASE_IMAGE).strip()
    if parent not in {item["image"] for item in _base_records()}:
        raise HTTPException(status_code=400, detail="请选择一张基础镜像")
    runtime_id = "rt_" + uuid.uuid4().hex[:8]
    image = f"local/user-{runtime_id}:latest"
    folder = _folder(runtime_id)
    folder.mkdir(parents=True, exist_ok=False)
    meta = {
        "id": runtime_id,
        "name": name,
        "image": image,
        "kind": "user",
        "codeRef": f"runtimes/{runtime_id}",
        "command": ["python", "serve.py"],
        "port": 8080,
        "healthPath": "/health",
        "protocol": "http",
        "models": model_ids,
        "weightQuants": quants,
        "baseImage": parent,
        "note": "",
        "createdAt": now_iso(),
    }
    (folder / "serve_user.py").write_text(serve_source, encoding="utf-8")
    (folder / "requirements.txt").write_text(pinned, encoding="utf-8")
    (folder / "serve.py").write_text(WRAPPER, encoding="utf-8")
    (folder / "Dockerfile").write_text(user_dockerfile(parent), encoding="utf-8")
    shutil.copyfile(repo_root() / "engines" / "common" / "runtime_loader.py", folder / "runtime_loader.py")
    (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def user_context(image: str) -> Path | None:
    for runtime in list_runtimes():
        if runtime["kind"] == "user" and runtime["image"] == image:
            folder = _folder(runtime["id"])
            if (folder / "Dockerfile").exists():
                return folder
    return None


def user_base_image(folder: Path) -> str:
    meta_path = folder / "meta.json"
    if meta_path.is_file():
        try:
            parent = str(json.loads(meta_path.read_text(encoding="utf-8")).get("baseImage") or "").strip()
        except (OSError, json.JSONDecodeError):
            parent = ""
        if parent:
            return parent
    return BASE_IMAGE


def runtime_document(image: str) -> dict:
    for runtime in list_runtimes():
        if runtime["image"] != image:
            continue
        return {
            "apiVersion": "aiserver.runtime/v1",
            "id": runtime["id"],
            "name": runtime["name"],
            "image": runtime["image"],
            "kind": runtime["kind"],
            "command": list(runtime["command"]),
            "port": runtime["port"],
            "healthPath": runtime["healthPath"],
            "protocol": runtime["protocol"],
            "models": list(runtime["models"]),
            "weightQuants": list(runtime["weightQuants"]),
            "baseImage": runtime.get("baseImage") or "",
            "codeRef": runtime.get("codeRef") or "",
            "note": runtime.get("note") or "",
        }
    raise HTTPException(status_code=404, detail="这台机体没有这份运行时")


def known_images() -> list[str]:
    return [runtime["image"] for runtime in list_runtimes()]
