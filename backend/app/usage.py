"""正在运行的容器占用：CPU、内存、显存。采样放在后台，列表接口不等待。"""

from __future__ import annotations

import os
import re
import subprocess
import threading
import time

from app import docker_cli

_SIZE = re.compile(r"([0-9]*\.?[0-9]+)\s*([A-Za-z]+)")
_UNITS = {
    "b": 1,
    "kb": 1000,
    "mb": 1000**2,
    "gb": 1000**3,
    "tb": 1000**4,
    "kib": 1024,
    "mib": 1024**2,
    "gib": 1024**3,
    "tib": 1024**4,
}

_lock = threading.Lock()
_cache: dict[str, dict] = {}
_names: list[str] = []
_seen = 0.0
_thread: threading.Thread | None = None


def parse_size(text: str) -> int | None:
    match = _SIZE.fullmatch(text.strip())
    if not match:
        return None
    factor = _UNITS.get(match.group(2).lower())
    if factor is None:
        return None
    return round(float(match.group(1)) * factor)


def parse_stats(output: str) -> dict[str, tuple[float | None, int | None]]:
    found: dict[str, tuple[float | None, int | None]] = {}
    for line in output.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        name, cpu_text, mem_text = parts[0].strip(), parts[1].strip(), parts[2].strip()
        cpu = None
        if cpu_text.endswith("%"):
            try:
                cpu = float(cpu_text[:-1])
            except ValueError:
                cpu = None
        found[name] = (cpu, parse_size(mem_text.split("/", 1)[0]))
    return found


def parse_gpu(output: str) -> dict[int, int]:
    by_pid: dict[int, int] = {}
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            by_pid[int(parts[0])] = round(float(parts[1]) * 1024 * 1024)
        except ValueError:
            continue
    return by_pid


def parse_pids(output: str) -> set[int]:
    pids: set[int] = set()
    for line in output.splitlines():
        text = line.strip()
        if text.isdigit():
            pids.add(int(text))
    return pids


def _docker(args: list[str]) -> tuple[int, str]:
    try:
        return docker_cli.run(args)
    except RuntimeError:
        return 1, ""


def _nvidia() -> tuple[int, str]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 1, ""
    output = (completed.stdout or "") + (completed.stderr or "")
    return completed.returncode, output.strip()


def collect(names: list[str]) -> dict[str, dict]:
    if not names:
        return {}
    code, output = _docker(
        ["stats", "--no-stream", "--format", "{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}", *names]
    )
    stats = parse_stats(output) if code == 0 else {}
    gpu_code, gpu_output = _nvidia()
    gpu = parse_gpu(gpu_output) if gpu_code == 0 else None
    found: dict[str, dict] = {}
    for name in names:
        cpu, memory = stats.get(name, (None, None))
        found[name] = {
            "cpuPercent": cpu,
            "memoryBytes": memory,
            "gpuMemoryBytes": _gpu_bytes(name, gpu),
        }
    return found


def _gpu_bytes(name: str, gpu: dict[int, int] | None) -> int | None:
    if gpu is None:
        return None
    code, output = _docker(["top", name, "-eo", "pid"])
    if code != 0:
        return None
    return sum(gpu.get(pid, 0) for pid in parse_pids(output))


def attach(records: list[dict]) -> None:
    names = [str(record["container"]) for record in records if record.get("status") == "running" and record.get("container")]
    found = _read(names)
    for record in records:
        name = str(record.get("container") or "")
        record["usage"] = found.get(name) if record.get("status") == "running" else None


def _read(names: list[str]) -> dict[str, dict]:
    if os.environ.get("AISERVER_USAGE_SYNC") == "1":
        return collect(names)
    return _cached(names)


def _cached(names: list[str]) -> dict[str, dict]:
    global _thread, _seen
    with _lock:
        _names[:] = names
        _seen = time.monotonic()
        snapshot = {name: dict(_cache[name]) for name in names if name in _cache}
        running = _thread is not None and _thread.is_alive()
        if names and not running:
            _thread = threading.Thread(target=_loop, name="deploy-usage", daemon=True)
            _thread.start()
    return snapshot


def _loop() -> None:
    while True:
        with _lock:
            if not _names or time.monotonic() - _seen > 10:
                _names.clear()
                return
            names = list(_names)
        try:
            found = collect(names)
        except Exception:
            found = {}
        with _lock:
            _cache.clear()
            _cache.update(found)
            if not _names:
                return
        if not found:
            time.sleep(2)
