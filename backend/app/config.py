from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO_ROOT / "data"
DEFAULT_CATALOG = REPO_ROOT / "backend" / "catalog" / "models.yaml"

_DATA_DIRS = ("library/local", "library/uploads", "registry", "packages", "runtimes", "builds", "deploys")


def repo_root() -> Path:
    return REPO_ROOT


def catalog_path() -> Path:
    raw = os.environ.get("AISERVER_CATALOG")
    return Path(raw) if raw else DEFAULT_CATALOG


def data_root() -> Path:
    raw = os.environ.get("AISERVER_DATA")
    path = Path(raw) if raw else DEFAULT_DATA
    for rel in _DATA_DIRS:
        (path / rel).mkdir(parents=True, exist_ok=True)
    return path
