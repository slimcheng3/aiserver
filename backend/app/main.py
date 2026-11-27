from __future__ import annotations

import json
import queue
import threading
import zipfile
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app import docker_cli
from app.builds import image_present, list_jobs, resume_orphans, start_build
from app.catalog import all_models, get_model
from app.config import catalog_path, data_root, repo_root
from app.deploys import call_chat, call_infer, deploy, list_deploys, probe, stop
from app.environment import environment
from app.packaging import create_package, delete_package, get_package, list_packages
from app.quantize import get_job as get_quant_job
from app.quantize import list_jobs as list_quant_jobs
from app.quantize import start_quant
from app.runtimes import (
    create_user_runtime,
    delete_base,
    known_images,
    list_runtimes,
    register_base,
    runtime_document,
)
from app.schemas import (
    BaseCreate,
    BuildRequest,
    ChatRequest,
    DeployCreate,
    LocalPathRegister,
    MetaInfo,
    ModelDef,
    PackageCreate,
    PackageView,
    QuantRequest,
    S3Register,
    S3Verify,
    S3VerifyResult,
    WeightAsset,
)
from app.storage import list_assets, register_local_path, register_s3, remove_asset, save_upload, verify_s3

app = FastAPI(
    title="具身模型交付台",
    summary="选出运行时，绑好权重，得到一份能在本机跑起来的模型包。",
    version="0.2.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://192.168.11.49:5173",
        "http://192.168.11.155:5173",
    ],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1|192\.168\.\d{1,3}\.\d{1,3})(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _resume_builds() -> None:
    resume_orphans()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/meta", response_model=MetaInfo)
def meta() -> MetaInfo:
    root = data_root()
    return MetaInfo(
        dataDir=str(root),
        localLibrary=str(root / "library" / "local"),
        catalogPath=str(catalog_path()),
        packageDir=str(root / "packages"),
    )


@app.get("/api/models", response_model=list[ModelDef])
def models() -> list[ModelDef]:
    try:
        return all_models()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"模型目录读不出来: {exc}") from exc


@app.get("/api/models/{model_id}", response_model=ModelDef)
def model_detail(model_id: str) -> ModelDef:
    try:
        found = get_model(model_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"模型目录读不出来: {exc}") from exc
    if found is None:
        raise HTTPException(status_code=404, detail="未知模型")
    return found


@app.get("/api/weights", response_model=list[WeightAsset])
def weights() -> list[WeightAsset]:
    return list_assets()


@app.post("/api/weights/upload", response_model=WeightAsset)
async def upload_weight(file: UploadFile = File(...)) -> WeightAsset:
    return await save_upload(file)


@app.post("/api/weights/s3", response_model=WeightAsset)
def add_s3(body: S3Register) -> WeightAsset:
    return register_s3(body)


@app.post("/api/weights/s3/verify", response_model=S3VerifyResult)
def check_s3(body: S3Verify) -> S3VerifyResult:
    return verify_s3(body.uri, body.endpoint, body.region)


@app.post("/api/weights/local-path", response_model=WeightAsset)
def add_local_path(body: LocalPathRegister) -> WeightAsset:
    return register_local_path(body)


@app.delete("/api/weights")
def delete_weight(id: str) -> dict[str, bool]:
    remove_asset(id)
    return {"ok": True}


@app.get("/api/packages", response_model=list[PackageView])
def packages() -> list[PackageView]:
    return list_packages()


@app.post("/api/packages", response_model=PackageView)
def add_package(body: PackageCreate) -> PackageView:
    return create_package(body)


@app.get("/api/packages/{pkg_id}", response_model=PackageView)
def package_detail(pkg_id: str) -> PackageView:
    return get_package(pkg_id)


def _role_folder(role: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in role).strip("._") or "weight"


def _zip_weight_path(role: str, path: Path) -> str:
    folder = _role_folder(role)
    if path.is_file():
        return f"weights/{folder}/{path.name}"
    return f"weights/{folder}"


def _weight_files(manifest: dict) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for weight in manifest.get("weights") or []:
        raw = str(weight.get("path") or weight.get("localPath") or weight.get("uri") or "")
        role = str(weight.get("role") or "weight")
        if not raw or raw.startswith("s3://"):
            raise HTTPException(status_code=400, detail="S3 地址不能打进下载包，请用本机路径")
        path = Path(raw)
        if not path.exists():
            raise HTTPException(status_code=400, detail=f"本机路径不存在：{path}")
        if path.is_file():
            files.append((path, _zip_weight_path(role, path)))
            continue
        folder = _role_folder(role)
        found = False
        for child in sorted(path.rglob("*")):
            if not child.is_file():
                continue
            found = True
            files.append((child, f"weights/{folder}/{child.relative_to(path).as_posix()}"))
        if not found:
            raise HTTPException(status_code=400, detail=f"权重目录里没有文件：{path}")
    if not files:
        raise HTTPException(status_code=400, detail="这份模型包没有可下载的本机权重")
    return files


def _archive_manifest(manifest: dict) -> bytes:
    """压缩包里的 path 改成包内相对路径。磁盘上的清单仍保留本机原路径。"""
    packed = json.loads(json.dumps(manifest))
    for weight in packed.get("weights") or []:
        raw = str(weight.get("path") or weight.get("localPath") or weight.get("uri") or "")
        role = str(weight.get("role") or "weight")
        path = Path(raw)
        weight.pop("localPath", None)
        weight.pop("uri", None)
        weight["path"] = _zip_weight_path(role, path)
    return (json.dumps(packed, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _export_package(document: bytes, files: list[tuple[Path, str]]) -> Iterator[bytes]:
    """压缩包里放 manifest.json 和本机权重。权重大文件按块写入，不先收齐整包。"""
    stop = threading.Event()
    chunks: queue.Queue[bytes | None] = queue.Queue(maxsize=4)
    failed: list[BaseException] = []

    class _Writer:
        def write(self, data: bytes) -> int:
            if stop.is_set():
                raise RuntimeError("下载已取消")
            chunks.put(data)
            return len(data)

        def seekable(self) -> bool:
            return False

        def flush(self) -> None:
            return None

    def produce() -> None:
        try:
            with zipfile.ZipFile(_Writer(), "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
                archive.writestr("manifest.json", document)
                for source, member in files:
                    with archive.open(member, "w", force_zip64=True) as dest, source.open("rb") as handle:
                        while True:
                            if stop.is_set():
                                raise RuntimeError("下载已取消")
                            block = handle.read(1024 * 1024)
                            if not block:
                                break
                            dest.write(block)
        except Exception as exc:
            failed.append(exc)
        finally:
            chunks.put(None)

    threading.Thread(target=produce, daemon=True).start()
    try:
        while True:
            piece = chunks.get()
            if piece is None:
                break
            yield piece
    finally:
        stop.set()
    if failed:
        raise failed[0]


@app.get("/api/packages/{pkg_id}/manifest")
def download_manifest(pkg_id: str):
    """只返回清单 JSON。权重在另一个地址，避免同一条链接有时是压缩包。"""
    view = get_package(pkg_id)
    name = view.manifest.get("metadata", {}).get("name", pkg_id)
    path = data_root() / "packages" / pkg_id / "manifest.json"
    return FileResponse(
        path,
        media_type="application/json",
        filename=f"{name}.manifest.json",
        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
    )


@app.get("/api/packages/{pkg_id}/archive")
def download_archive(pkg_id: str):
    view = get_package(pkg_id)
    name = view.manifest.get("metadata", {}).get("name", pkg_id)
    files = _weight_files(view.manifest)
    document = _archive_manifest(view.manifest)
    filename = f"{name}.zip"
    ascii_name = "".join(ch if ch.isascii() and (ch.isalnum() or ch in "-._") else "_" for ch in filename).strip("._") or "package.zip"
    disposition = f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(filename)}"
    return StreamingResponse(
        _export_package(document, files),
        media_type="application/zip",
        headers={"Content-Disposition": disposition, "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
    )


@app.delete("/api/packages/{pkg_id}")
def remove_package(pkg_id: str) -> dict[str, bool]:
    delete_package(pkg_id)
    return {"ok": True}


@app.get("/api/runtimes")
def runtimes() -> list[dict]:
    return list_runtimes()


@app.post("/api/runtimes")
async def add_runtime(
    name: str = Form(...),
    models: str = Form(...),
    weightQuants: str = Form(...),
    serve: UploadFile = File(...),
    requirements: UploadFile = File(None),
    baseImage: str = Form(""),
) -> dict:
    try:
        serve_source = (await serve.read()).decode("utf-8")
        requirement_text = ""
        if requirements is not None:
            requirement_text = (await requirements.read()).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="上传的文件需要是 UTF-8 文本") from exc
    chunks = models.replace("，", ",").replace("\r", "\n").replace("\n", ",")
    model_ids = [item.strip() for item in chunks.split(",") if item.strip()]
    quants = [item.strip() for item in weightQuants.split(",") if item.strip()]
    return create_user_runtime(name, model_ids, quants, serve_source, requirement_text, baseImage)


@app.post("/api/bases")
def add_base(body: BaseCreate) -> dict:
    return register_base(body.name, body.image, body.note)


@app.delete("/api/bases")
def remove_base(image: str) -> dict[str, bool]:
    delete_base(image)
    return {"ok": True}


def _download_stem(image: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in image)


def _export_runtime(image: str) -> Iterator[bytes]:
    """压缩包里放镜像 tar 和 runtime.json。镜像按块写入，不先收齐整包。"""
    document = json.dumps(runtime_document(image), ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    tar_name = "runtime.tar"
    stop = threading.Event()
    chunks: queue.Queue[bytes | None] = queue.Queue(maxsize=4)
    failed: list[BaseException] = []

    class _Writer:
        def write(self, data: bytes) -> int:
            if stop.is_set():
                raise RuntimeError("下载已取消")
            chunks.put(data)
            return len(data)

        def seekable(self) -> bool:
            return False

        def flush(self) -> None:
            return None

    def produce() -> None:
        saved = docker_cli.save(image)
        try:
            with zipfile.ZipFile(_Writer(), "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
                archive.writestr("runtime.json", document)
                with archive.open(tar_name, "w", force_zip64=True) as dest:
                    for piece in saved:
                        if stop.is_set():
                            break
                        dest.write(piece)
        except Exception as exc:
            failed.append(exc)
        finally:
            close = getattr(saved, "close", None)
            if callable(close):
                close()
            chunks.put(None)

    threading.Thread(target=produce, daemon=True).start()
    try:
        while True:
            piece = chunks.get()
            if piece is None:
                break
            yield piece
    finally:
        stop.set()
    if failed:
        raise failed[0]


@app.get("/api/runtimes/tar")
def download_runtime_tar(image: str) -> StreamingResponse:
    image = image.strip()
    if image not in set(known_images()):
        raise HTTPException(status_code=404, detail="这台机体没有这份运行时")
    try:
        present = image_present(image)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not present:
        raise HTTPException(status_code=404, detail="镜像还没构建，不能下载")
    filename = _download_stem(image) + ".zip"
    return StreamingResponse(
        _export_runtime(image),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/quants")
def quants() -> list[dict]:
    return list_quant_jobs()


@app.post("/api/quants")
def add_quant(body: QuantRequest) -> dict:
    result = start_quant(body.assetId, body.quant, body.calibration, body.image)
    asset = result["asset"]
    return {"job": result["job"], "asset": None if asset is None else asset.model_dump()}


@app.get("/api/quants/{job_id}")
def quant_job(job_id: str) -> dict:
    return get_quant_job(job_id)


@app.get("/api/builds")
def builds(image: str = "") -> list[dict]:
    return list_jobs(image)


@app.post("/api/builds")
def add_build(body: BuildRequest) -> dict:
    return start_build(body.image, body.force)


@app.get("/api/environment")
def env_status() -> dict:
    return environment()


@app.get("/api/deploys")
def deploys() -> list[dict]:
    return list_deploys()


@app.post("/api/deploys")
def add_deploy(body: DeployCreate) -> dict:
    return deploy(body.packageId, body.hostPort)


@app.get("/api/deploys/{pkg_id}/health")
def deploy_health(pkg_id: str) -> dict:
    return probe(pkg_id)


@app.post("/api/deploys/{pkg_id}/infer")
def deploy_infer(pkg_id: str, body: dict) -> dict:
    return call_infer(pkg_id, body)


@app.post("/api/deploys/{pkg_id}/chat")
def deploy_chat(pkg_id: str, body: ChatRequest) -> dict:
    return call_chat(pkg_id, body.prompt, body.max_tokens, body.temperature)


@app.delete("/api/deploys/{pkg_id}")
def remove_deploy(pkg_id: str) -> dict:
    return stop(pkg_id)


DIST = repo_root() / "frontend" / "dist"
if (DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")


@app.get("/")
def index():
    page = DIST / "index.html"
    if page.exists():
        return FileResponse(page)
    return {
        "service": "aiserver-packager",
        "ui": "http://127.0.0.1:5173",
        "docs": "/docs",
    }
