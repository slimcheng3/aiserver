from __future__ import annotations

import json
import re
import uuid

from fastapi import HTTPException

from app.catalog import get_model
from app.config import data_root
from app.runtimes import find_runtime
from app.schemas import PackageCreate, PackageView
from app.storage import get_asset, now_iso

API_VERSION = "aiserver.modelpackage/v1"
PACKAGER_VERSION = "0.1.0"
_PKG_ID = re.compile(r"pkg_[a-f0-9]{12}")


def _check_id(pkg_id: str) -> None:
    if not _PKG_ID.fullmatch(pkg_id):
        raise HTTPException(status_code=400, detail="包编号不合法")


def _view(folder) -> PackageView:
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    return PackageView(absoluteRoot=str(folder.resolve()), manifest=manifest)


def list_packages() -> list[PackageView]:
    root = data_root() / "packages"
    views: list[PackageView] = []
    if not root.exists():
        return views
    for folder in root.iterdir():
        manifest_path = folder / "manifest.json"
        if not folder.is_dir() or not manifest_path.exists():
            continue
        try:
            views.append(_view(folder))
        except (OSError, json.JSONDecodeError):
            continue
    views.sort(key=lambda item: item.manifest.get("metadata", {}).get("createdAt", ""), reverse=True)
    return views


def get_package(pkg_id: str) -> PackageView:
    _check_id(pkg_id)
    folder = data_root() / "packages" / pkg_id
    manifest_path = folder / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="找不到这个模型包")
    return _view(folder)


_LOCKED_FLAGS = ("--model", "--host", "--port")


def _split_cli(parts: list[str]) -> tuple[list[str], list[str]]:
    locked: list[str] = []
    extra: list[str] = []
    index = 0
    while index < len(parts):
        token = parts[index]
        if token in _LOCKED_FLAGS:
            locked.append(token)
            index += 1
            if index < len(parts) and not parts[index].startswith("-"):
                locked.append(parts[index])
                index += 1
            continue
        extra.append(token)
        index += 1
    return locked, extra


def _command_for_package(runtime: dict, submitted: list[str]) -> list[str]:
    catalog = [str(part) for part in runtime["command"]]
    if not catalog or not str(catalog[0]).startswith("--"):
        return catalog
    locked, catalog_extra = _split_cli(catalog)
    command = [str(part).strip() for part in submitted if str(part).strip()]
    if command and command[0].startswith("-"):
        _, extra = _split_cli(command)
    else:
        extra = catalog_extra
    return locked + extra


def delete_package(pkg_id: str) -> None:
    _check_id(pkg_id)
    folder = data_root() / "packages" / pkg_id
    manifest_path = folder / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="找不到这个模型包")
    manifest_path.unlink()
    try:
        folder.rmdir()
    except OSError:
        pass


def create_package(body: PackageCreate) -> PackageView:
    model = get_model(body.modelId)
    if model is None:
        raise HTTPException(status_code=404, detail="未知模型")

    roles = {artifact.role: artifact for artifact in model.artifacts}
    bound = [item.role for item in body.weights]
    if len(bound) != len(set(bound)):
        raise HTTPException(status_code=400, detail="同一个权重角色不能绑定两次")
    for role in bound:
        if role not in roles:
            raise HTTPException(status_code=400, detail=f"这个模型没有权重角色 {role}")
    for role, spec in roles.items():
        if spec.required and role not in bound:
            raise HTTPException(status_code=400, detail=f"还缺必填权重：{spec.label}")
    if not body.weights:
        raise HTTPException(status_code=400, detail="至少绑定一个权重")
    if model.id not in model.engine.models:
        raise HTTPException(status_code=400, detail="这个运行时没有声明支持当前模型")
    runtime = find_runtime(model.id, body.engine.image)
    supported = set(runtime["weightQuants"])

    weight_entries = []
    for binding in body.weights:
        asset = get_asset(binding.assetId)
        if asset is None or asset.missing:
            raise HTTPException(status_code=400, detail="有权重不存在，或磁盘路径已经失效")
        spec = roles[binding.role]
        if spec.quantized:
            if binding.quant is None:
                raise HTTPException(status_code=400, detail=f"请标明量化类型：{spec.label}")
            if binding.quant not in supported:
                raise HTTPException(
                    status_code=400,
                    detail=f"这个运行时不能加载 {binding.quant} 权重",
                )
        elif binding.quant is not None:
            raise HTTPException(status_code=400, detail=f"{spec.label} 不标量化类型")
        weight_entries.append(
            {
                "role": binding.role,
                "label": spec.label,
                "quant": binding.quant,
                "source": asset.source,
                "kind": asset.kind,
                "name": asset.name,
                "uri": asset.uri,
                "sizeBytes": asset.sizeBytes,
                "sha256": asset.sha256,
                "relativePath": asset.relativePath,
                "localPath": asset.localPath,
                "endpoint": asset.endpoint,
                "region": asset.region,
                "bucket": asset.bucket,
                "key": asset.key,
                "note": asset.note,
            }
        )

    pkg_id = "pkg_" + uuid.uuid4().hex[:12]
    folder = data_root() / "packages" / pkg_id
    folder.mkdir(parents=True, exist_ok=False)
    manifest = {
        "apiVersion": API_VERSION,
        "kind": "ModelPackage",
        "packager": {"name": "aiserver-console", "version": PACKAGER_VERSION},
        "metadata": {
            "id": pkg_id,
            "name": body.name,
            "version": body.version,
            "description": body.description,
            "createdAt": now_iso(),
        },
        "model": {
            "id": model.id,
            "displayName": model.displayName,
            "vendor": model.vendor,
            "task": model.task,
            "interfaces": model.interfaces.model_dump(),
        },
        "engine": {
            "image": runtime["image"],
            "codeRef": runtime["codeRef"],
            "command": _command_for_package(runtime, body.engine.command),
            "port": runtime["port"],
            "healthPath": runtime["healthPath"],
            "protocol": runtime["protocol"],
            "env": body.engine.env,
            "resources": body.engine.resources.model_dump(),
            "models": list(runtime["models"]),
            "weightQuants": list(runtime["weightQuants"]),
        },
        "weights": weight_entries,
        "package": {"id": pkg_id, "root": f"packages/{pkg_id}"},
    }
    (folder / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return _view(folder)
