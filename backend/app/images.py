"""内置镜像怎么构建。模型代码目录不在交付台仓库里。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from app.config import repo_root


@dataclass(frozen=True)
class ImageRecipe:
    image: str
    dockerfile: Path
    context: Path
    contexts: dict[str, Path]
    required: dict[str, Path]


def _src(env_name: str, default: str) -> Path:
    return Path(os.environ.get(env_name, default))


def recipes() -> dict[str, ImageRecipe]:
    root = repo_root()
    common = root / "engines" / "common"
    sam3 = _src("AISERVER_SAM3_SRC", r"D:\playgound\sam3")
    genpose2 = _src("AISERVER_GENPOSE2_SRC", r"D:\playgound\genpose2")
    flashrt = _src("AISERVER_FLASHRT_SRC", r"D:\playgound\flashrt")

    def recipe(image: str, folder: str, contexts: dict[str, Path], required: dict[str, Path]) -> ImageRecipe:
        return ImageRecipe(
            image=image,
            dockerfile=root / "engines" / folder / "Dockerfile",
            context=root / "engines" / folder,
            contexts=contexts,
            required=required,
        )

    items = [
        recipe("local/base-cuda128-torch:latest", "base", {"common": common}, {}),
        recipe(
            "local/sam3-engine:latest",
            "sam3",
            {"common": common, "sam3": sam3},
            {"sam3": sam3},
        ),
        recipe(
            "local/genpose2-engine:latest",
            "genpose2",
            {"common": common, "genpose2": genpose2},
            {"genpose2": genpose2},
        ),
        recipe("local/vllm-engine:latest", "vllm", {"common": common}, {}),
        recipe(
            "local/flashrt-engine:latest",
            "flashrt",
            {"common": common, "flashrt": flashrt},
            {"flashrt": flashrt},
        ),
    ]
    return {item.image: item for item in items}


def build_args(recipe: ImageRecipe) -> list[str]:
    args = ["build", "-f", str(recipe.dockerfile), "-t", recipe.image]
    for name, path in recipe.contexts.items():
        args.extend(["--build-context", f"{name}={path}"])
    args.append(str(recipe.context))
    return args


def missing_sources(recipe: ImageRecipe) -> list[str]:
    missing = []
    for name, path in recipe.required.items():
        if not path.exists():
            missing.append(f"找不到{name}的模型代码：{path}")
    return missing
