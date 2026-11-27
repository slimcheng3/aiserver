"""把本机上的 Qwen 模型目录量化成 vLLM 能加载的 fp8、awq 或 gptq 目录。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import uuid
from pathlib import Path

from fastapi import HTTPException

from app import docker_cli
from app.config import data_root, repo_root
from app.schemas import LocalPathRegister, WeightAsset
from app.storage import get_asset, now_iso, register_local_path

VLLM_IMAGE = "local/vllm-engine:latest"
SAM3_IMAGE = "local/sam3-engine:latest"
_OFFLINE = {"fp8", "awq", "gptq"}
_SAM3 = {"fp16", "bf16", "int8"}
_CALIBRATED = {"awq", "gptq"}
_MIN_SAMPLES = 8
_LOCK = threading.Lock()


def _dir() -> Path:
    path = data_root() / "quants"
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


def get_job(job_id: str) -> dict:
    job = _read(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="没有这次量化")
    return job


def list_jobs() -> list[dict]:
    jobs: list[dict] = []
    for path in _dir().glob("qnt_*.json"):
        try:
            jobs.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    jobs.sort(key=lambda job: job.get("createdAt") or "", reverse=True)
    return jobs


def _append(job_id: str, line: str, status: str | None = None) -> None:
    with _LOCK:
        job = _read(job_id)
        if job is None:
            return
        job["logs"].append(line)
        if len(job["logs"]) > 4000:
            job["logs"] = job["logs"][-3000:]
        if status:
            job["status"] = status
        _write(job)


def _builtin_calibration() -> str:
    path = repo_root() / "engines" / "vllm" / "calibration.txt"
    if not path.is_file():
        raise HTTPException(status_code=500, detail="找不到内置校准文本")
    return path.read_text(encoding="utf-8")


def calibration_samples(text: str) -> list[str]:
    body = text.replace("\r\n", "\n").strip()
    if not body:
        return []
    if re.search(r"\n\s*\n", body):
        parts = re.split(r"\n\s*\n", body)
    else:
        parts = body.splitlines()
    return [part.strip() for part in parts if part.strip()]


def _calibration_text(quant: str, calibration: str) -> str:
    if quant not in _CALIBRATED:
        return ""
    text = calibration.strip() or _builtin_calibration()
    samples = calibration_samples(text)
    if len(samples) < _MIN_SAMPLES:
        raise HTTPException(
            status_code=400,
            detail=f"校准文本至少要 {_MIN_SAMPLES} 条，现在只有 {len(samples)} 条。空行分段，每段一条；留空则用内置文本。",
        )
    return text


def _key(source: Path, quant: str, calibration: str = "") -> str:
    material = f"{source.resolve()}\n{quant}"
    if quant in _CALIBRATED:
        material += "\n" + calibration
    return hashlib.sha256(material.encode()).hexdigest()[:16]


def _output_name(source: Path, quant: str) -> str:
    raw = source.name.strip() or "weights"
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in raw).strip("._")
    return f"{safe or 'weights'}-{quant}"[:120]


def _output(key: str, source: Path, quant: str) -> Path:
    return data_root() / "quantized" / key / _output_name(source, quant)


def _ready_asset(source: Path, quant: str, calibration: str = "") -> WeightAsset | None:
    folder = _output(_key(source, quant, calibration), source, quant)
    config = folder / "config.json"
    if not config.is_file():
        return None
    try:
        payload = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if "quantization_config" not in payload:
        return None
    return register_local_path(
        LocalPathRegister(path=str(folder), note=f"{quant} 量化自 {source}"),
    )


def _running_for(key: str) -> dict | None:
    for path in _dir().glob("qnt_*.json"):
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if job.get("key") == key and job.get("status") in {"queued", "running"}:
            return job
    return None


def _file_output(key: str, source: Path, quant: str) -> Path:
    raw = source.stem.strip() or "checkpoint"
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in raw).strip("._")
    return data_root() / "quantized" / key / f"{safe or 'checkpoint'}-{quant}.pt"


def _ready_file(path: Path, quant: str, source: Path) -> WeightAsset | None:
    if not path.is_file() or path.stat().st_size <= 0:
        return None
    return register_local_path(LocalPathRegister(path=str(path), note=f"{quant} 量化自 {source}"))


def _start_sam3(asset_id: str, quant: str) -> dict:
    if quant not in _SAM3:
        raise HTTPException(status_code=400, detail="SAM 3 只做 fp16 和 bf16")
    asset = get_asset(asset_id)
    if asset is None or asset.missing or not asset.localPath:
        raise HTTPException(status_code=400, detail="请选择本机上的 checkpoint 文件")
    source = Path(asset.localPath)
    if not source.is_file():
        raise HTTPException(status_code=400, detail="SAM 3 量化需要 checkpoint 文件，不是目录")
    script = repo_root() / "engines" / "sam3" / "quantize.py"
    if not script.is_file():
        raise HTTPException(status_code=500, detail="找不到量化脚本")
    try:
        present = docker_cli.run(["image", "inspect", SAM3_IMAGE])[0] == 0
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not present:
        raise HTTPException(status_code=400, detail="先构建 local/sam3-engine:latest")

    key = _key(source, quant)
    final = _file_output(key, source, quant)
    existing = _ready_file(final, quant, source)
    if existing is not None:
        return {"job": None, "asset": existing}
    running = _running_for(key)
    if running is not None:
        return {"job": running, "asset": None}

    work = final.parent / "work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    job = {
        "id": "qnt_" + uuid.uuid4().hex[:12],
        "key": key,
        "assetId": asset_id,
        "quant": quant,
        "image": SAM3_IMAGE,
        "source": str(source),
        "output": str(final),
        "status": "running",
        "logs": [f"准备把 {source.name} 量化成 {quant}"],
        "resultAssetId": None,
        "createdAt": now_iso(),
        "updatedAt": now_iso(),
    }
    _write(job)
    if os.environ.get("AISERVER_BUILD_SYNC") == "1":
        _execute_sam3(job["id"], script, source, work, quant, final)
        done = _read(job["id"]) or job
        asset_out = get_asset(done["resultAssetId"]) if done.get("resultAssetId") else None
        return {"job": done, "asset": asset_out}
    threading.Thread(
        target=_execute_sam3,
        args=(job["id"], script, source, work, quant, final),
        daemon=True,
    ).start()
    return {"job": job, "asset": None}


def _execute_sam3(job_id: str, script: Path, source: Path, work: Path, quant: str, final: Path) -> None:
    args = [
        "run",
        "--rm",
        "--entrypoint",
        "python3",
        "-e",
        "PYTHONUNBUFFERED=1",
        "-e",
        "WEIGHTS_PATH=/weights/checkpoint",
        "-e",
        "QUANT_OUTPUT=/quantized/checkpoint.pt",
        "-e",
        f"QUANT={quant}",
        "-v",
        f"{source.resolve()}:/weights/checkpoint:ro",
        "-v",
        f"{work.resolve()}:/quantized",
        "-v",
        f"{script.resolve()}:/opt/aiserver/quantize.py:ro",
        SAM3_IMAGE,
        "/opt/aiserver/quantize.py",
    ]
    if not _stream(job_id, args):
        return
    produced = work / "checkpoint.pt"
    if not produced.is_file() or produced.stat().st_size <= 0:
        _append(job_id, "量化结束，但没有写出 checkpoint", "failed")
        return
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        final.unlink()
    produced.replace(final)
    try:
        registered = register_local_path(LocalPathRegister(path=str(final), note=f"{quant} 量化自 {source}"))
    except Exception as exc:
        detail = getattr(exc, "detail", None)
        _append(job_id, str(detail or exc), "failed")
        return
    with _LOCK:
        job = _read(job_id)
        if job is None:
            return
        job["resultAssetId"] = registered.id
        job["status"] = "ready"
        job["logs"].append(f"已登记 {registered.id}")
        _write(job)


def _stream(job_id: str, args: list[str]) -> bool:
    try:
        code = docker_cli.stream(args, lambda line: _append(job_id, line))
    except Exception as exc:
        _append(job_id, str(exc), "failed")
        return False
    if code != 0:
        _append(job_id, "量化失败", "failed")
        return False
    return True


def start_quant(asset_id: str, quant: str, calibration: str = "", image: str = "") -> dict:
    quant = quant.strip().lower()
    image = image.strip()
    if image == SAM3_IMAGE or (not image and quant in _SAM3):
        return _start_sam3(asset_id, quant)
    if image and image != VLLM_IMAGE:
        raise HTTPException(status_code=400, detail="这个运行时还不能离线量化")
    if quant not in _OFFLINE:
        raise HTTPException(status_code=400, detail="当前离线量化只做 fp8、awq、gptq，SAM 3 还可以做 fp16、bf16")
    text = _calibration_text(quant, calibration)
    asset = get_asset(asset_id)
    if asset is None or asset.missing or not asset.localPath:
        raise HTTPException(status_code=400, detail="请选择本机上的模型目录")
    source = Path(asset.localPath)
    if not source.is_dir() or not (source / "config.json").is_file():
        raise HTTPException(status_code=400, detail="量化需要 Hugging Face 模型目录，里面要有 config.json")
    script = repo_root() / "engines" / "vllm" / "quantize.py"
    if not script.is_file():
        raise HTTPException(status_code=500, detail="找不到量化脚本")
    try:
        present = docker_cli.run(["image", "inspect", VLLM_IMAGE])[0] == 0
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not present:
        raise HTTPException(status_code=400, detail="先构建 local/vllm-engine:latest")

    key = _key(source, quant, text)
    calibration_path = _write_calibration(key, text) if text else None
    existing = _ready_asset(source, quant, text)
    if existing is not None:
        return {"job": None, "asset": existing}
    running = _running_for(key)
    if running is not None:
        return {"job": running, "asset": None}

    work = data_root() / "quantized" / key / "work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    job = {
        "id": "qnt_" + uuid.uuid4().hex[:12],
        "key": key,
        "assetId": asset_id,
        "quant": quant,
        "image": VLLM_IMAGE,
        "source": str(source),
        "output": str(_output(key, source, quant)),
        "status": "running",
        "logs": [_start_note(source.name, quant, text)],
        "resultAssetId": None,
        "createdAt": now_iso(),
        "updatedAt": now_iso(),
    }
    _write(job)
    if os.environ.get("AISERVER_BUILD_SYNC") == "1":
        _execute(job["id"], script, source, work, quant, calibration_path)
        done = _read(job["id"]) or job
        asset_out = get_asset(done["resultAssetId"]) if done.get("resultAssetId") else None
        return {"job": done, "asset": asset_out}
    threading.Thread(
        target=_execute,
        args=(job["id"], script, source, work, quant, calibration_path),
        daemon=True,
    ).start()
    return {"job": job, "asset": None}


def _start_note(name: str, quant: str, text: str) -> str:
    note = f"准备把 {name} 量化成 {quant}"
    if quant not in _CALIBRATED:
        return note
    count = len(calibration_samples(text))
    origin = "内置" if text == _builtin_calibration() else "提交的"
    return f"{note}，使用{origin}校准文本 {count} 条"


def _write_calibration(key: str, text: str) -> Path:
    folder = data_root() / "quantized" / key / "calibration"
    folder.mkdir(parents=True, exist_ok=True)
    rows = [{"text": sample} for sample in calibration_samples(text)]
    (folder / "train.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return folder


def _execute(job_id: str, script: Path, source: Path, work: Path, quant: str, calibration: Path | None) -> None:
    args = [
        "run",
        "--rm",
        "--gpus",
        "all",
        "--entrypoint",
        "python3",
        "-e",
        "HF_HUB_OFFLINE=1",
        "-e",
        "HF_DATASETS_OFFLINE=1",
        "-e",
        "PYTHONUNBUFFERED=1",
        "-e",
        "WEIGHTS_PATH=/weights/checkpoint",
        "-e",
        "QUANT_OUTPUT=/quantized",
        "-e",
        f"QUANT={quant}",
    ]
    if calibration is not None:
        args.extend(
            [
                "-e",
                "CALIBRATION_PATH=/opt/aiserver/calibration",
                "-v",
                f"{calibration.resolve()}:/opt/aiserver/calibration:ro",
            ]
        )
    args.extend(
        [
        "-v",
        f"{source.resolve()}:/weights/checkpoint:ro",
        "-v",
        f"{work.resolve()}:/quantized",
        "-v",
        f"{script.resolve()}:/opt/aiserver/quantize.py:ro",
        VLLM_IMAGE,
        "/opt/aiserver/quantize.py",
        ]
    )
    try:
        code = docker_cli.stream(args, lambda line: _append(job_id, line))
    except Exception as exc:
        _append(job_id, str(exc), "failed")
        return
    if code != 0:
        _append(job_id, "量化失败", "failed")
        return
    config = work / "config.json"
    try:
        payload = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _append(job_id, "量化结束，但结果目录里没有可用的 config.json", "failed")
        return
    if "quantization_config" not in payload:
        _append(job_id, "量化结束，但 config.json 里没有 quantization_config", "failed")
        return
    with _LOCK:
        current = _read(job_id)
    if current is None or not current.get("output"):
        _append(job_id, "量化结束，但任务记录里没有输出目录", "failed")
        return
    final = Path(current["output"])
    if final.exists():
        shutil.rmtree(final)
    work.replace(final)
    try:
        registered = register_local_path(LocalPathRegister(path=str(final), note=f"{quant} 量化自 {source}"))
    except Exception as exc:
        detail = getattr(exc, "detail", None)
        _append(job_id, str(detail or exc), "failed")
        return
    with _LOCK:
        job = _read(job_id)
        if job is None:
            return
        job["resultAssetId"] = registered.id
        job["status"] = "ready"
        job["logs"].append(f"已登记 {registered.id}")
        _write(job)
