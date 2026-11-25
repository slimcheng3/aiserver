"""OpenPI WebSocket client against the FlashRT server, with a fake model."""

import asyncio
import sys
import urllib.request
from pathlib import Path

import numpy as np
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent))
import serve


class FakeModel:
    def __init__(self) -> None:
        self.prompts = []
        self.states = []
        self.seen = []

    def set_prompt(self, prompt, state=None):
        self.prompts.append(prompt)
        self.states.append(None if state is None else np.asarray(state).copy())

    def infer(self, observation):
        self.seen.append(observation)
        return {"actions": np.arange(8, dtype=np.float32).reshape(2, 4)}


def _get(url: str) -> tuple[int, str, str]:
    with urllib.request.urlopen(url, timeout=5) as response:
        body = response.read().decode("utf-8")
        return response.status, response.headers.get("Content-Type", ""), body


async def _session(port: int, model: FakeModel) -> None:
    uri = f"ws://127.0.0.1:{port}"
    async with websockets.connect(uri, compression=None, max_size=None) as ws:
        metadata = serve.unpackb(await ws.recv())
        assert metadata == {}, metadata
        image = np.zeros((3, 4, 5), dtype=np.float32)
        image[0] = 1.0
        wrist = np.zeros((4, 5, 3), dtype=np.uint8)
        state = np.arange(7, dtype=np.float32)
        obs = {
            "observation/image": image,
            "observation/wrist_image": wrist,
            "observation/state": state,
            "prompt": "pick up the cup",
        }
        await ws.send(serve.Packer().pack(obs))
        first = serve.unpackb(await ws.recv())
        assert isinstance(first["actions"], np.ndarray), type(first["actions"])
        assert first["actions"].shape == (2, 4)
        assert first["actions"].dtype == np.float32
        assert "infer_ms" in first["server_timing"]
        assert "prev_total_ms" not in first["server_timing"]

        await ws.send(serve.Packer().pack(obs))
        second = serve.unpackb(await ws.recv())
        assert "prev_total_ms" in second["server_timing"]

    seen = model.seen[0]
    assert seen["image"].dtype == np.uint8
    assert seen["image"].shape == (4, 5, 3)
    assert seen["image"][..., 0].min() == 255
    assert seen["wrist_image"].shape == (4, 5, 3)
    assert model.prompts == ["pick up the cup", "pick up the cup"]
    assert model.states[0].shape == (7,)

    async with websockets.connect(uri, compression=None, max_size=None) as ws:
        await ws.recv()
        await ws.send("not-msgpack")
        error = await ws.recv()
        assert isinstance(error, str), type(error)
        assert "Traceback" in error


def test_checkpoint_framework(tmp: Path | None = None) -> None:
    import tempfile

    root = Path(tempfile.mkdtemp()) if tmp is None else tmp
    torch_dir = root / "torch"
    torch_dir.mkdir()
    (torch_dir / "model.safetensors").write_bytes(b"")
    assert serve.checkpoint_framework(str(torch_dir)) == "torch"

    jax_dir = root / "jax"
    (jax_dir / "params").mkdir(parents=True)
    assert serve.checkpoint_framework(str(jax_dir)) == "jax"

    both = root / "both"
    (both / "params").mkdir(parents=True)
    (both / "model.safetensors").write_bytes(b"")
    assert serve.checkpoint_framework(str(both)) == "torch"


def test_predict_calibrates_before_infer() -> None:
    class Pipe:
        def __init__(self) -> None:
            self.calls = []

        def predict(self, images, prompt=None, state=None):
            self.calls.append((prompt, None if state is None else np.asarray(state).copy()))
            return np.ones((2, 7), dtype=np.float32)

    model = Pipe()
    model._pipe = object()
    result = serve.infer(model, {
        "observation/state": np.arange(8, dtype=np.float32),
        "prompt": "pick up the cup",
        "observation/image": np.zeros((2, 2, 3), dtype=np.uint8),
    })
    assert result["actions"].shape == (2, 7)
    assert model.calls[0][0] == "pick up the cup"
    assert model.calls[0][1].shape == (8,)


def test_libero_and_droid_observations() -> None:
    libero = serve._state_of({"observation/state": np.arange(8, dtype=np.float32)})
    assert libero.shape == (8,)
    assert libero[7] == 7

    droid = serve._state_of({
        "observation/joint_position": np.arange(7, dtype=np.float32),
        "observation/gripper_position": np.float32(0.25),
    })
    assert droid.shape == (8,)
    assert droid[:7].tolist() == list(range(7))
    assert droid[7] == np.float32(0.25)

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    wrist = np.ones((8, 8, 3), dtype=np.uint8)
    mapped = serve.to_model_observation({
        "observation/exterior_image_1_left": image,
        "observation/wrist_image_left": wrist,
    })
    assert mapped["image"].shape == (8, 8, 3)
    assert mapped["wrist_image"].shape == (8, 8, 3)
    assert int(mapped["wrist_image"][0, 0, 0]) == 1


async def main() -> None:
    test_checkpoint_framework()
    test_predict_calibrates_before_infer()
    test_libero_and_droid_observations()
    model = FakeModel()
    server = await serve.serve(model, serve.infer, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        status, content_type, body = await asyncio.to_thread(_get, f"http://127.0.0.1:{port}/healthz")
        assert status == 200 and body == "OK\n", (status, body)
        status, content_type, body = await asyncio.to_thread(_get, f"http://127.0.0.1:{port}/health")
        assert status == 200 and body == '{"status":"ok"}', (status, content_type, body)
        assert "application/json" in content_type
        await _session(port, model)
    finally:
        server.close()
        await server.wait_closed()
    print("openpi websocket protocol ok")


if __name__ == "__main__":
    asyncio.run(main())
