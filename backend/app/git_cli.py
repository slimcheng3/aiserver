"""对本机 git 的薄封装。测试可以换掉 run。"""

from __future__ import annotations

import subprocess
from pathlib import Path


def run(args: list[str], cwd: Path | None = None) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("这台机体没有 git") from exc
    output = (completed.stdout or "") + (completed.stderr or "")
    return completed.returncode, output.strip()
