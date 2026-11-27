from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import HTTPException, UploadFile

from app.config import data_root
from app.schemas import LocalPathRegister, S3Register, S3VerifyResult, WeightAsset

_LOCK = threading.Lock()
_HASH_LIMIT = 64 * 1024 * 1024
_WALK_CAP = 20000
_WEIGHT_EXTS = {
    ".pt",
    ".pth",
    ".bin",
    ".safetensors",
    ".onnx",
    ".ckpt",
    ".gguf",
    ".npz",
    ".msgpack",
    ".ot",
    ".engine",
}
_CONFIG_NAMES = {"config.json", "tokenizer.json", "preprocessor_config.json", "model.safetensors.index.json"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str | None:
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size > _HASH_LIMIT:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str | None:
    try:
        return path.resolve().relative_to(data_root().resolve()).as_posix()
    except ValueError:
        return None


def _mtime(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
    except OSError:
        return None


def is_weight_file(path: Path) -> bool:
    name = path.name.lower()
    if name.endswith(".index.json"):
        return True
    return path.suffix.lower() in _WEIGHT_EXTS


def _directory_is_model(path: Path) -> bool:
    checked = 0
    for root, dirs, files in os.walk(path):
        dirs[:] = [item for item in dirs if not item.startswith(".")]
        for name in files:
            if name.startswith("."):
                continue
            checked += 1
            if checked > 5000:
                return True
            candidate = Path(root) / name
            if name in _CONFIG_NAMES or is_weight_file(candidate):
                return True
    return False


def _directory_stat(path: Path) -> tuple[int, int]:
    total = 0
    count = 0
    for root, dirs, files in os.walk(path):
        dirs[:] = [item for item in dirs if not item.startswith(".")]
        for name in files:
            if name.startswith("."):
                continue
            count += 1
            if count > _WALK_CAP:
                return total, _WALK_CAP
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                continue
    return total, count


def _file_asset(
    path: Path,
    *,
    asset_id: str,
    source: str,
    origin: str,
    removable: bool,
    note: str | None = None,
) -> WeightAsset:
    exists = path.exists()
    size = None
    digest = None
    if exists and path.is_file():
        try:
            size = path.stat().st_size
        except OSError:
            size = None
        digest = _sha256_file(path) if exists else None
    return WeightAsset(
        id=asset_id,
        source=source,  # type: ignore[arg-type]
        origin=origin,  # type: ignore[arg-type]
        kind="file",
        name=path.name,
        uri=path.resolve().as_uri() if exists else path.as_uri(),
        removable=removable,
        missing=not exists,
        sizeBytes=size,
        fileCount=1 if exists else 0,
        sha256=digest,
        relativePath=_relative(path) if exists else None,
        localPath=str(path.resolve()) if exists else str(path),
        note=note,
        createdAt=_mtime(path),
    )


def _model_type(path: Path) -> str | None:
    config = path / "config.json"
    try:
        if not config.is_file() or config.stat().st_size > 2_000_000:
            return None
        data = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    value = data.get("model_type") if isinstance(data, dict) else None
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _dir_asset(
    path: Path,
    *,
    asset_id: str,
    source: str,
    origin: str,
    removable: bool,
    note: str | None = None,
) -> WeightAsset:
    exists = path.exists()
    size = None
    count = None
    if exists and path.is_dir():
        size, count = _directory_stat(path)
    return WeightAsset(
        id=asset_id,
        source=source,  # type: ignore[arg-type]
        origin=origin,  # type: ignore[arg-type]
        kind="directory",
        name=path.name,
        uri=path.resolve().as_uri() if exists else path.as_uri(),
        removable=removable,
        missing=not exists,
        sizeBytes=size,
        fileCount=count,
        relativePath=_relative(path) if exists else None,
        localPath=str(path.resolve()) if exists else str(path),
        note=note,
        createdAt=_mtime(path),
        modelType=_model_type(path) if exists and path.is_dir() else None,
    )


def _read_json(path: Path, default: list) -> list:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _s3_file() -> Path:
    return data_root() / "registry" / "s3.json"


def _path_file() -> Path:
    return data_root() / "registry" / "local_paths.json"


def _library() -> Path:
    return data_root() / "library" / "local"


def _uploads() -> Path:
    return data_root() / "library" / "uploads"


def normalize_s3(uri: str) -> tuple[str, str, str]:
    raw = uri.strip()
    if not raw.startswith("s3://"):
        raise HTTPException(status_code=400, detail="地址需要以 s3:// 开头")
    rest = raw[5:]
    bucket, sep, key = rest.partition("/")
    if not sep or not bucket or not key or bucket.startswith("/") or "\\" in rest:
        raise HTTPException(status_code=400, detail="地址形如 s3://bucket/path/to/weight")
    if any(part == ".." for part in key.split("/")):
        raise HTTPException(status_code=400, detail="key 不合法")
    return f"s3://{bucket}/{key}", bucket, key


def _check_endpoint(endpoint: str) -> str:
    endpoint = endpoint.strip().rstrip("/")
    if endpoint and not endpoint.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Endpoint 需要是 http 或 https 地址")
    return endpoint


def list_assets() -> list[WeightAsset]:
    assets: list[WeightAsset] = []
    library = _library()
    if library.exists():
        for child in sorted(library.iterdir(), key=lambda item: item.name.lower()):
            if child.name.startswith("."):
                continue
            if child.is_file() and is_weight_file(child):
                rel = child.relative_to(library).as_posix()
                assets.append(
                    _file_asset(child, asset_id=f"local:{rel}", source="local", origin="library", removable=False)
                )
            elif child.is_dir() and _directory_is_model(child):
                rel = child.relative_to(library).as_posix()
                assets.append(
                    _dir_asset(child, asset_id=f"local:{rel}", source="local", origin="library", removable=False)
                )

    uploads = _uploads()
    if uploads.exists():
        for folder in sorted(uploads.iterdir(), key=lambda item: item.name.lower()):
            if not folder.is_dir():
                continue
            for child in sorted(folder.iterdir(), key=lambda item: item.name.lower()):
                if child.is_file():
                    rel = child.relative_to(uploads).as_posix()
                    assets.append(
                        _file_asset(
                            child,
                            asset_id=f"upload:{rel}",
                            source="upload",
                            origin="upload",
                            removable=True,
                        )
                    )

    for record in _read_json(_path_file(), []):
        path = Path(record["path"])
        note = record.get("note") or None
        created = record.get("createdAt")
        asset_id = record["id"]
        if path.is_dir() or (not path.exists() and record.get("kind") == "directory"):
            asset = _dir_asset(path, asset_id=asset_id, source="local", origin="registered", removable=True, note=note)
        else:
            asset = _file_asset(path, asset_id=asset_id, source="local", origin="registered", removable=True, note=note)
        asset.createdAt = created
        assets.append(asset)

    for record in _read_json(_s3_file(), []):
        assets.append(
            WeightAsset(
                id=record["id"],
                source="s3",
                origin="s3",
                kind="remote",
                name=record["uri"],
                uri=record["uri"],
                removable=True,
                sizeBytes=record.get("sizeBytes"),
                endpoint=record.get("endpoint") or None,
                region=record.get("region") or None,
                bucket=record.get("bucket"),
                key=record.get("key"),
                note=record.get("note") or None,
                createdAt=record.get("createdAt"),
            )
        )

    assets.sort(key=lambda item: (item.missing, item.source, item.name.lower()))
    return assets


def get_asset(asset_id: str) -> WeightAsset | None:
    for asset in list_assets():
        if asset.id == asset_id:
            return asset
    return None


def sanitize_filename(name: str | None) -> str:
    raw = (name or "weight.bin").replace("\\", "/").split("/")[-1].replace("\x00", "")
    if raw in {"", ".", ".."}:
        return "weight.bin"
    return raw[:180]


async def save_upload(upload: UploadFile) -> WeightAsset:
    asset_key = uuid.uuid4().hex[:12]
    filename = sanitize_filename(upload.filename)
    dest_dir = _uploads() / asset_key
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    try:
        with dest.open("wb") as handle:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    except Exception:
        if dest.exists():
            dest.unlink()
        raise
    finally:
        await upload.close()
    if dest.stat().st_size == 0:
        dest.unlink()
        dest_dir.rmdir()
        raise HTTPException(status_code=400, detail="文件是空的")
    rel = dest.relative_to(_uploads()).as_posix()
    return _file_asset(dest, asset_id=f"upload:{rel}", source="upload", origin="upload", removable=True)


def register_s3(body: S3Register) -> WeightAsset:
    uri, bucket, key = normalize_s3(body.uri)
    endpoint = _check_endpoint(body.endpoint)
    region = body.region.strip()
    note = body.note.strip()
    with _LOCK:
        records = _read_json(_s3_file(), [])
        for record in records:
            if record["uri"] == uri and (record.get("endpoint") or "") == endpoint:
                found = get_asset(record["id"])
                if found:
                    return found
        record = {
            "id": "s3:" + uuid.uuid4().hex[:12],
            "uri": uri,
            "bucket": bucket,
            "key": key,
            "endpoint": endpoint,
            "region": region,
            "note": note,
            "createdAt": now_iso(),
        }
        records.append(record)
        _write_json(_s3_file(), records)
    created = get_asset(record["id"])
    if created is None:
        raise HTTPException(status_code=500, detail="S3 地址登记后没有出现在列表里")
    return created


def verify_s3(uri: str, endpoint: str, region: str) -> S3VerifyResult:
    normalized, bucket, key = normalize_s3(uri)
    endpoint = _check_endpoint(endpoint)
    if key.endswith("/"):
        return S3VerifyResult(ok=False, detail="这是一个前缀，对象存储没法对前缀做 HEAD。可以直接登记，部署时再拉取。")
    try:
        import boto3
        from botocore.config import Config
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError:
        return S3VerifyResult(ok=False, detail="运行环境里没有 boto3")

    try:
        client = boto3.client(
            "s3",
            region_name=region.strip() or None,
            endpoint_url=endpoint or None,
            config=Config(connect_timeout=3, read_timeout=8, retries={"max_attempts": 1}),
        )
        head = client.head_object(Bucket=bucket, Key=key)
    except (ClientError, BotoCoreError, OSError) as exc:
        return S3VerifyResult(ok=False, detail=f"没能访问 {normalized}：{exc}")
    length = head.get("ContentLength")
    return S3VerifyResult(ok=True, sizeBytes=length, detail=None)


def register_local_path(body: LocalPathRegister) -> WeightAsset:
    raw = body.path.strip().strip('"')
    if not raw:
        raise HTTPException(status_code=400, detail="请填写路径")
    path = Path(raw).expanduser().resolve()
    if path == Path(path.anchor):
        raise HTTPException(status_code=400, detail="请指到具体的权重文件或目录，不要登记整块磁盘")
    if not path.exists():
        raise HTTPException(status_code=400, detail="路径不存在")
    library = _library().resolve()
    try:
        path.relative_to(library)
    except ValueError:
        inside_library = False
    else:
        inside_library = True
    if inside_library:
        raise HTTPException(status_code=400, detail="这个路径已经在本地库目录里，刷新列表后直接选用")

    kind = "directory" if path.is_dir() else "file"
    asset_id = "path:" + hashlib.sha1(str(path).lower().encode()).hexdigest()[:16]
    note = body.note.strip()
    with _LOCK:
        records = _read_json(_path_file(), [])
        if any(record["id"] == asset_id for record in records):
            found = get_asset(asset_id)
            if found:
                return found
        records.append(
            {
                "id": asset_id,
                "path": str(path),
                "kind": kind,
                "note": note,
                "createdAt": now_iso(),
            }
        )
        _write_json(_path_file(), records)
    created = get_asset(asset_id)
    if created is None:
        raise HTTPException(status_code=500, detail="路径登记后没有出现在列表里")
    return created


def remove_asset(asset_id: str) -> None:
    if asset_id.startswith("local:"):
        raise HTTPException(status_code=400, detail="本地库里的文件请到磁盘上删除，这里只取消额外登记的路径")
    if asset_id.startswith("upload:"):
        rel = asset_id[len("upload:") :]
        target = (_uploads() / rel).resolve()
        root = _uploads().resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise HTTPException(status_code=404, detail="找不到这个上传文件")
        target.unlink()
        parent = target.parent
        if parent != root and not any(parent.iterdir()):
            parent.rmdir()
        return
    if asset_id.startswith("s3:"):
        with _LOCK:
            records = [record for record in _read_json(_s3_file(), []) if record["id"] != asset_id]
            _write_json(_s3_file(), records)
        return
    if asset_id.startswith("path:"):
        with _LOCK:
            records = [record for record in _read_json(_path_file(), []) if record["id"] != asset_id]
            _write_json(_path_file(), records)
        return
    raise HTTPException(status_code=404, detail="找不到这个权重")
