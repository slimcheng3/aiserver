"""GenPose2 运行时。镜像里的默认 load 和 infer。"""

from __future__ import annotations

import base64
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path("/opt/genpose2")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime_loader import resolve_fns

WEIGHTS_PATH = os.environ.get("WEIGHTS_PATH", "/weights/checkpoint")
PORT = int(os.environ.get("PORT", "8081"))


def _find_ckpt(root: Path, *needles: str) -> Path:
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".pth", ".pt", ".ckpt"}]
    lowered = [(p, p.name.lower(), str(p).lower()) for p in files]
    for needle in needles:
        hits = [p for p, name, full in lowered if needle in name or needle in full]
        if hits:
            hits.sort(key=lambda item: len(str(item)))
            return hits[0]
    raise FileNotFoundError(f"在 {root} 里找不到含 {needles} 的权重")


def load(weights_path: str):
    root = Path(weights_path)
    if not root.exists():
        raise FileNotFoundError(f"找不到 GenPose2 权重：{root}")
    folder = root if root.is_dir() else root.parent
    score = os.environ.get("GENPOSE2_SCORE_CKPT")
    energy = os.environ.get("GENPOSE2_ENERGY_CKPT")
    scale = os.environ.get("GENPOSE2_SCALE_CKPT")
    score_path = Path(score) if score else _find_ckpt(folder, "score")
    energy_path = Path(energy) if energy else _find_ckpt(folder, "energy")
    scale_path = Path(scale) if scale else _find_ckpt(folder, "scale")
    for path in (score_path, energy_path, scale_path):
        if not path.is_file():
            raise FileNotFoundError(f"找不到权重文件：{path}")
    from runners.infer import create_genpose2

    return create_genpose2(str(score_path), str(energy_path), str(scale_path))


def _decode_image(payload: str, flags: int):
    import cv2
    import numpy as np

    raw = payload.split(",", 1)[1] if "," in payload else payload
    data = np.frombuffer(base64.b64decode(raw), dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise ValueError("图片不是合法的 png/jpeg")
    return image


def _camera_meta(camera: dict, width: int, height: int) -> dict:
    if "camera" in camera and "intrinsics" in camera.get("camera", {}):
        meta = dict(camera)
        meta.setdefault("depth_scale", float(camera.get("depth_scale", 0.001)))
        return meta
    if "cam_K" not in camera:
        raise ValueError("camera 需要 cam_K 或 camera.intrinsics")
    k = camera["cam_K"]
    if len(k) != 9:
        raise ValueError("cam_K 必须是 9 个数")
    return {
        "camera": {
            "intrinsics": {
                "fx": float(k[0]),
                "fy": float(k[4]),
                "cx": float(k[2]),
                "cy": float(k[5]),
                "width": int(width),
                "height": int(height),
            }
        },
        "depth_scale": float(camera.get("depth_scale", 0.001)),
    }


def infer(model, request: dict) -> dict:
    import cv2
    import numpy as np
    from datasets.datasets_infer import InferDataset

    if not isinstance(request, dict):
        raise ValueError("请求需要是 JSON 对象")
    if not request.get("rgb") or not request.get("depth") or not request.get("camera") or not request.get("mask"):
        raise ValueError("需要 rgb、depth、mask（base64）和 camera（JSON）")

    color = cv2.cvtColor(_decode_image(str(request["rgb"]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    depth_img = _decode_image(str(request["depth"]), cv2.IMREAD_UNCHANGED)
    if depth_img.ndim == 3:
        depth_img = depth_img[:, :, 0]
    camera = request["camera"]
    if isinstance(camera, str):
        camera = json.loads(camera)
    meta = _camera_meta(camera, width=color.shape[1], height=color.shape[0])
    scale = float(meta.get("depth_scale", 0.001))
    if depth_img.dtype == np.uint16 or (depth_img.dtype == np.uint8 and scale != 1.0):
        depth = depth_img.astype(np.float32) * scale
    else:
        depth = depth_img.astype(np.float32)

    mask_img = _decode_image(str(request["mask"]), cv2.IMREAD_UNCHANGED)
    if mask_img.ndim == 3:
        mask_img = mask_img[:, :, 0]
    mask = mask_img.astype(np.int32)
    if mask.max() == 255 and np.unique(mask).size <= 3:
        mask = (mask > 0).astype(np.int32)

    data = InferDataset(
        {"depth": depth, "color": color, "mask": mask, "meta": meta},
        img_size=model.cfg.img_size,
        device=model.cfg.device,
        n_pts=model.cfg.num_points,
    )
    pose, length = model.inference(data, prev_pose=None, tracking=False)
    pose_np = pose[0].detach().cpu().numpy()
    length_np = length[0].detach().cpu().numpy()
    return {
        "poses": pose_np.tolist(),
        "lengths": length_np.tolist(),
        "count": int(pose_np.shape[0]),
    }


class Handler(BaseHTTPRequestHandler):
    processor = None
    infer_fn = infer

    def do_GET(self):
        if self.path.rstrip("/") == "/health":
            self._json(200, {"status": "ok"})
            return
        self._json(404, {"detail": "未知路径"})

    def do_POST(self):
        if self.path.rstrip("/") != "/infer":
            self._json(404, {"detail": "未知路径"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length).decode("utf-8"))
            self._json(200, Handler.infer_fn(self.processor, request))
        except ValueError as exc:
            self._json(400, {"detail": str(exc)})
        except Exception as exc:
            self._json(500, {"detail": str(exc)})

    def log_message(self, fmt, *args):
        print(f"[genpose2] {self.address_string()} {fmt % args}", flush=True)

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    load_fn, infer_fn, pipeline = resolve_fns(WEIGHTS_PATH, load, infer)
    if pipeline:
        print(f"使用权重旁的 pipeline.py：{pipeline}", flush=True)
    Handler.infer_fn = infer_fn
    Handler.processor = load_fn(WEIGHTS_PATH)
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"GenPose2 运行时监听 {PORT}，权重 {WEIGHTS_PATH}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
