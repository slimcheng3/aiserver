#!/usr/bin/env python3
"""直连本机 GenPose2 运行时。默认端口与当前部署一致。"""

from __future__ import annotations

import argparse
import base64
import json
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:39483")
    parser.add_argument("--rgb", default=str(HERE / "rgb.png"))
    parser.add_argument("--depth", default=str(HERE / "depth.png"))
    parser.add_argument("--mask", default=str(HERE / "mask.png"))
    parser.add_argument("--camera", default=str(HERE / "camera.json"))
    args = parser.parse_args()
    url = args.url.rstrip("/")
    camera = json.loads(Path(args.camera).read_text(encoding="utf-8"))
    health = urllib.request.urlopen(f"{url}/health", timeout=10)
    print("health:", json.loads(health.read().decode()))
    body = {
        "rgb": b64(Path(args.rgb)),
        "depth": b64(Path(args.depth)),
        "mask": b64(Path(args.mask)),
        "camera": camera,
    }
    req = urllib.request.Request(
        f"{url}/infer",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            result = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"{exc.code} {exc.read().decode()}") from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
