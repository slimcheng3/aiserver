"""内置镜像怎么构建。SAM 3 / GenPose2 / FlashRT 的推理代码在仓库 engines/*/upstream。"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from app.config import repo_root


@dataclass(frozen=True)
class ImageRecipe:
    image: str
    dockerfile: Path
    context: Path
    contexts: dict[str, Path]
    required: dict[str, Path]
    clones: dict[str, str]
    arguments: dict[str, str] = field(default_factory=dict)


def detect_flashrt_target() -> tuple[str, str]:
    """本机构建目标。默认构建架构 = 这台机器要跑的架构。可用环境变量覆盖。"""
    forced = os.environ.get("AISERVER_FLASHRT_GPU_ARCH", "").strip()
    nvidia = _nvidia_arch()
    npu = _has_npu()
    if npu and not nvidia:
        return "npu", forced or os.environ.get("ASCEND_SOC_VERSION", "Ascend910B4")
    if forced:
        return "cuda", forced
    if nvidia:
        return "cuda", nvidia
    return "cuda", "89"


def _nvidia_arch() -> str:
    if shutil.which("nvidia-smi") is None:
        return ""
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if completed.returncode != 0:
        return ""
    line = (completed.stdout or "").splitlines()
    if not line:
        return ""
    cap = line[0].strip()
    if "." in cap:
        major, _, minor = cap.partition(".")
        if major.isdigit() and minor[:1].isdigit():
            return f"{major}{minor[0]}"
    digits = "".join(ch for ch in cap if ch.isdigit())
    return digits


def _has_npu() -> bool:
    if shutil.which("npu-smi") is None:
        return False
    try:
        completed = subprocess.run(
            ["npu-smi", "info"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def recipes() -> dict[str, ImageRecipe]:
    root = repo_root()
    common = root / "engines" / "common"
    backend, arch = detect_flashrt_target()
    flashrt_file = "Dockerfile.npu" if backend == "npu" else "Dockerfile"
    flashrt_args = {"GPU_ARCH": arch}
    if backend == "npu":
        flashrt_args["SOC_VERSION"] = arch
        base = os.environ.get("AISERVER_FLASHRT_NPU_BASE", "").strip()
        if base:
            flashrt_args["BASE_IMAGE"] = base

    def recipe(
        image: str,
        folder: str,
        contexts: dict[str, Path],
        required: dict[str, Path],
        clones: dict[str, str] | None = None,
        dockerfile: str = "Dockerfile",
        arguments: dict[str, str] | None = None,
    ) -> ImageRecipe:
        return ImageRecipe(
            image=image,
            dockerfile=root / "engines" / folder / dockerfile,
            context=root / "engines" / folder,
            contexts=contexts,
            required=required,
            clones=clones or {},
            arguments=arguments or {},
        )

    items = [
        recipe("local/base-cuda128-torch:latest", "base", {"common": common}, {}),
        recipe("local/sam3-engine:latest", "sam3", {"common": common}, {}),
        recipe(
            "local/genpose2-engine:latest",
            "genpose2",
            {"common": common},
            {"dinov2": root / "engines" / "genpose2" / "torch-hub" / "facebookresearch_dinov2_main"},
        ),
        recipe("local/vllm-engine:latest", "vllm", {}, {}),
        recipe(
            "local/flashrt-engine:latest",
            "flashrt",
            {"common": common},
            {},
            dockerfile=flashrt_file,
            arguments=flashrt_args,
        ),
    ]
    return {item.image: item for item in items}


def build_args(recipe: ImageRecipe) -> list[str]:
    args = ["build", "--progress", "plain", "-f", str(recipe.dockerfile), "-t", recipe.image]
    for name, path in recipe.contexts.items():
        args.extend(["--build-context", f"{name}={path}"])
    for key, value in recipe.arguments.items():
        args.extend(["--build-arg", f"{key}={value}"])
    args.append(str(recipe.context))
    return args
