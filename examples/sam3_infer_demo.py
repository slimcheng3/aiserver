#!/usr/bin/env python3
"""调用本机已部署的 SAM 3 服务：改下面几个常量后直接运行。"""

from __future__ import annotations

import base64
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

# 改这里
IMAGE = "/path/to/photo.jpg"
TEXT = "person"
URL = "http://127.0.0.1:51121"
OUT_DIR = "sam3_out"
THRESHOLD = None  # 例如 0.5；不设阈值就保持 None


def get_json(url: str, timeout: float = 10) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_json(url: str, payload: dict, timeout: float = 180) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    image_path = Path(IMAGE).expanduser().resolve()
    if not image_path.is_file():
        print(f"找不到图片：{image_path}", file=sys.stderr)
        return 1

    base = URL.rstrip("/")
    try:
        health = get_json(f"{base}/health")
    except urllib.error.URLError as exc:
        print(f"连不上服务 {base}/health：{exc}", file=sys.stderr)
        return 1
    print("health:", health)

    body = {
        "image": base64.b64encode(image_path.read_bytes()).decode("ascii"),
        "text": TEXT,
    }
    if THRESHOLD is not None:
        body["threshold"] = THRESHOLD

    try:
        result = post_json(f"{base}/infer", body)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(f"infer 失败 HTTP {exc.code}: {detail}", file=sys.stderr)
        return 1

    scores = result.get("scores") or []
    boxes = result.get("boxes") or []
    masks = result.get("masks") or []
    print(f"检出 {len(scores)} 个目标")
    for i, (score, box) in enumerate(zip(scores, boxes)):
        print(f"  [{i}] score={score:.4f} box={box}")

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, mask_b64 in enumerate(masks):
        dest = out_dir / f"mask_{i:02d}.png"
        dest.write_bytes(base64.b64decode(mask_b64))
        print(f"  写入 {dest}")

    (out_dir / "result.json").write_text(
        json.dumps({"scores": scores, "boxes": boxes}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    main()
