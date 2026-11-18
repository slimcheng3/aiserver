"""容器启动时，若权重旁边有 pipeline.py，就用它的 load 和 infer。"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def pipeline_path(weights_path: str) -> Path | None:
    candidate = Path(weights_path)
    folder = candidate if candidate.is_dir() else candidate.parent
    found = folder / "pipeline.py"
    if found.is_file():
        return found
    return None


def resolve_fns(weights_path: str, load, infer):
    found = pipeline_path(weights_path)
    if found is None:
        return load, infer, None
    spec = importlib.util.spec_from_file_location("weight_pipeline", found)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"读不了 {found}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "load", None)) or not callable(getattr(module, "infer", None)):
        raise RuntimeError("pipeline.py 需要实现 load 和 infer")
    return module.load, module.infer, str(found)
