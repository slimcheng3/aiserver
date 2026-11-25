import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { protocolLabel } from "../format";
import type { Deployment } from "../types";

type Result = {
  scores: number[];
  boxes: number[][];
  masks: string[];
};

function asResult(payload: Record<string, unknown>): Result {
  const scores = Array.isArray(payload.scores) ? payload.scores.map((item) => Number(item)) : [];
  const boxes = Array.isArray(payload.boxes)
    ? payload.boxes.map((item) => (Array.isArray(item) ? item.map((value) => Number(value)) : []))
    : [];
  const masks = Array.isArray(payload.masks) ? payload.masks.map((item) => String(item)) : [];
  return { scores, boxes, masks };
}

function pythonLiteral(value: string): string {
  return JSON.stringify(value);
}

const GENPOSE_CAMERA = `{
  "cam_K": [913.36, 0, 629.48, 0, 914.34, 362.86, 0, 0, 1],
  "depth_scale": 0.001
}`;

function pythonSample(endpoint: string, text: string, threshold: string, imageName: string): string {
  const url = endpoint.replace(/\/$/, "") || "http://127.0.0.1:8080";
  const image = imageName || "/path/to/photo.jpg";
  const thresh = threshold.trim();
  const threshLine = thresh === "" || Number.isNaN(Number(thresh)) ? "None" : String(Number(thresh));
  return `import base64
import json
import urllib.request
from pathlib import Path

URL = ${pythonLiteral(url)}
IMAGE = ${pythonLiteral(image)}
TEXT = ${pythonLiteral(text.trim() || "person")}
THRESHOLD = ${threshLine}

image = base64.b64encode(Path(IMAGE).read_bytes()).decode("ascii")
body = {"image": image, "text": TEXT}
if THRESHOLD is not None:
    body["threshold"] = THRESHOLD

health = urllib.request.urlopen(f"{URL}/health", timeout=10)
print("health:", json.loads(health.read().decode()))

req = urllib.request.Request(
    f"{URL}/infer",
    data=json.dumps(body).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=180) as resp:
    result = json.loads(resp.read().decode())

print("scores:", result.get("scores"))
print("boxes:", result.get("boxes"))
for i, mask in enumerate(result.get("masks") or []):
    Path(f"mask_{i:02d}.png").write_bytes(base64.b64decode(mask))
`;
}

function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const text = String(reader.result || "");
      const comma = text.indexOf(",");
      resolve(comma >= 0 ? text.slice(comma + 1) : text);
    };
    reader.onerror = () => reject(new Error("读图片失败"));
    reader.readAsDataURL(file);
  });
}

export function TestPage({ packageId }: { packageId: string | null }) {
  const [deploys, setDeploys] = useState<Deployment[]>([]);
  const [selected, setSelected] = useState(packageId ?? "");
  const [text, setText] = useState("person");
  const [threshold, setThreshold] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [depthFile, setDepthFile] = useState<File | null>(null);
  const [maskFile, setMaskFile] = useState<File | null>(null);
  const [cameraJson, setCameraJson] = useState(GENPOSE_CAMERA);
  const [preview, setPreview] = useState("");
  const [health, setHealth] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<Result | null>(null);
  const [poseText, setPoseText] = useState("");
  const [prompt, setPrompt] = useState("你好");
  const [chatText, setChatText] = useState("");
  const [copied, setCopied] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const running = useMemo(() => deploys.filter((item) => item.status === "running" && item.endpoint), [deploys]);
  const current = running.find((item) => item.packageId === selected) ?? running[0] ?? null;
  const isGenpose = Boolean(current?.image.includes("genpose2"));
  const isVllm = Boolean(current?.image.includes("vllm"));
  const isWebsocket = current?.protocol === "websocket";

  useEffect(() => {
    api
      .deploys()
      .then((list) => {
        setDeploys(list);
        const wanted = packageId || selected;
        const live = list.filter((item) => item.status === "running" && item.endpoint);
        if (wanted && live.some((item) => item.packageId === wanted)) {
          setSelected(wanted);
          return;
        }
        if (live[0]) setSelected(live[0].packageId);
      })
      .catch((reason: Error) => setError(reason.message));
  }, [packageId]);

  useEffect(() => {
    if (!file) {
      setPreview("");
      return;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  useEffect(() => {
    if (!preview || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const image = new Image();
    image.onload = () => {
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.drawImage(image, 0, 0);
      if (!result) return;
      result.masks.forEach((mask, index) => {
        const overlay = new Image();
        overlay.onload = () => {
          ctx.save();
          ctx.globalAlpha = 0.45;
          ctx.drawImage(overlay, 0, 0, canvas.width, canvas.height);
          ctx.restore();
          const box = result.boxes[index];
          if (!box || box.length < 4) return;
          ctx.strokeStyle = "#3dcaa4";
          ctx.lineWidth = Math.max(2, canvas.width / 400);
          ctx.strokeRect(box[0], box[1], box[2] - box[0], box[3] - box[1]);
          const score = result.scores[index];
          if (Number.isFinite(score)) {
            ctx.fillStyle = "#3dcaa4";
            ctx.font = `${Math.max(14, canvas.width / 50)}px sans-serif`;
            ctx.fillText(score.toFixed(2), box[0] + 4, Math.max(16, box[1] - 6));
          }
        };
        overlay.src = `data:image/png;base64,${mask}`;
      });
      if (result.masks.length === 0) {
        result.boxes.forEach((box, index) => {
          if (box.length < 4) return;
          ctx.strokeStyle = "#3dcaa4";
          ctx.lineWidth = Math.max(2, canvas.width / 400);
          ctx.strokeRect(box[0], box[1], box[2] - box[0], box[3] - box[1]);
          const score = result.scores[index];
          if (Number.isFinite(score)) {
            ctx.fillStyle = "#3dcaa4";
            ctx.font = `${Math.max(14, canvas.width / 50)}px sans-serif`;
            ctx.fillText(score.toFixed(2), box[0] + 4, Math.max(16, box[1] - 6));
          }
        });
      }
    };
    image.src = preview;
  }, [preview, result]);

  async function checkHealth() {
    if (!current) return;
    setError("");
    setHealth("");
    try {
      const data = await api.deployHealth(current.packageId);
      setHealth(JSON.stringify(data.body));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "健康检查失败");
    }
  }

  async function runInfer() {
    if (!current) return;
    setError("");
    setBusy(true);
    setResult(null);
    setPoseText("");
    setChatText("");
    try {
      let body: Record<string, unknown>;
      if (isGenpose) {
        if (!file || !depthFile || !maskFile) {
          throw new Error("GenPose2 需要 RGB、深度图和 mask");
        }
        body = {
          rgb: await fileToBase64(file),
          depth: await fileToBase64(depthFile),
          mask: await fileToBase64(maskFile),
          camera: JSON.parse(cameraJson),
        };
      } else {
        if (!file) throw new Error("请先选一张图片");
        const prompt = text.trim();
        if (!prompt) throw new Error("请填写文本概念");
        body = { image: await fileToBase64(file), text: prompt };
        const raw = threshold.trim();
        if (raw) body.threshold = Number(raw);
      }
      const payload = await api.infer(current.packageId, body);
      if (isGenpose) setPoseText(JSON.stringify(payload, null, 2));
      setResult(asResult(payload));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "推理失败");
    } finally {
      setBusy(false);
    }
  }

  async function runChat() {
    if (!current) return;
    const value = prompt.trim();
    if (!value) {
      setError("请填写提示");
      return;
    }
    setError("");
    setBusy(true);
    setChatText("");
    try {
      const payload = await api.chat(current.packageId, { prompt: value, max_tokens: 256, temperature: 0 });
      const choices = Array.isArray(payload.choices) ? payload.choices : [];
      const first = choices[0] && typeof choices[0] === "object" ? (choices[0] as Record<string, unknown>) : null;
      const message = first && typeof first.message === "object" && first.message ? (first.message as Record<string, unknown>) : null;
      const content = message && typeof message.content === "string" ? message.content : JSON.stringify(payload, null, 2);
      setChatText(content);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "对话失败");
    } finally {
      setBusy(false);
    }
  }

  async function copySample(sample: string) {
    try {
      await navigator.clipboard.writeText(sample);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      setError("复制失败，请手动选择下面的样例");
    }
  }

  return (
    <div className="page">
      <p className="page-kicker">调用</p>
      <h1 className="page-title">测试</h1>
      <p className="page-lead">
        页面测试：SAM 3 / GenPose2 走交付台转发 /infer。Qwen 在页面里走交付台转发 /v1/chat/completions，下面的 Python 样例是拷到本机解释器跑的，浏览器不能执行 Python。π0.5 用 OpenPI WebSocket 客户端直连端口。
      </p>
      {error && <p className="error">{error}</p>}
      {running.length === 0 ? (
        <>
          <p className="empty">
            还没有正在运行的容器。先去<a href="#/runtime">部署</a>。下面是一份直连 /infer 的 Python 样例，部署后把 URL 换成实际端口即可。
          </p>
          <PythonSample
            sample={pythonSample("http://127.0.0.1:8080", text, threshold, file?.name ?? "")}
            copied={copied}
            onCopy={(sample) => void copySample(sample)}
          />
        </>
      ) : (
        <>
          <div className="form-grid">
            <label className="field">
              <span>运行中的模型包</span>
              <select
                className="select"
                value={current?.packageId ?? ""}
                onChange={(event) => {
                  setSelected(event.target.value);
                  setResult(null);
                  setHealth("");
                  location.hash = `#/test?package=${event.target.value}`;
                }}
              >
                {running.map((item) => (
                  <option key={item.packageId} value={item.packageId}>
                    {item.name} · {item.hostPort}
                  </option>
                ))}
              </select>
            </label>
            <div className="field">
              <span>服务</span>
              <strong>
                {current ? `${protocolLabel(current.protocol)} · ${current.endpoint}` : "—"}
              </strong>
            </div>
          </div>
          {isWebsocket ? (
            <p className="warn">页面上的表单只支持 HTTP。π0.5 用下面的 OpenPI WebSocket 客户端直连 {current?.endpoint}。</p>
          ) : current && current.protocol !== "http" ? (
            <p className="warn">这个测试页只支持 HTTP。当前协议是 {protocolLabel(current.protocol)}。</p>
          ) : isVllm ? (
            <section className="section">
              <h2 className="section-title">Qwen · 官方 OpenAI 接口</h2>
              <p className="hint">页面按钮由交付台转到容器的官方接口。健康检查 GET /health，对话 POST /v1/chat/completions。下面那段 Python 只是样例，要拷出去跑。</p>
              <label className="field">
                <span>提示</span>
                <textarea className="textarea" rows={4} value={prompt} onChange={(event) => setPrompt(event.target.value)} placeholder="你好" />
              </label>
              <div className="toolbar">
                <button type="button" className="btn" onClick={() => void checkHealth()}>
                  健康检查
                </button>
                <button type="button" className="btn primary" disabled={busy} onClick={() => void runChat()}>
                  {busy ? "生成中…" : "在页面试一下"}
                </button>
                {current && (
                  <a className="btn" href={`#/runtime?package=${current.packageId}`}>
                    返回部署
                  </a>
                )}
              </div>
              {health && <p className="ok">健康检查：{health}</p>}
              {chatText && (
                <>
                  <p className="ok">已返回</p>
                  <pre className="manifest">{chatText}</pre>
                </>
              )}
            </section>
          ) : (
            <section className="section">
              <h2 className="section-title">{isGenpose ? "GenPose2 RGB + 深度 + mask" : "SAM 3 图像 + 文本"}</h2>
              <div className="test-layout">
                <div>
                  <label className="drop">
                    <strong>{isGenpose ? "RGB" : "选择图片"}</strong>
                    <span className="hint">{file ? file.name : "jpg / png / webp"}</span>
                    <input
                      className="drop-input"
                      type="file"
                      accept="image/*"
                      onChange={(event) => {
                        setFile(event.target.files?.[0] ?? null);
                        setResult(null);
                      }}
                    />
                  </label>
                  {isGenpose ? (
                    <>
                      <label className="drop" style={{ marginTop: 12 }}>
                        <strong>深度图</strong>
                        <span className="hint">{depthFile ? depthFile.name : "png / tiff，16 位深度"}</span>
                        <input
                          className="drop-input"
                          type="file"
                          accept="image/*,.tif,.tiff"
                          onChange={(event) => {
                            setDepthFile(event.target.files?.[0] ?? null);
                            setResult(null);
                          }}
                        />
                      </label>
                      <label className="drop" style={{ marginTop: 12 }}>
                        <strong>mask</strong>
                        <span className="hint">{maskFile ? maskFile.name : "分割掩码，白/实例 id"}</span>
                        <input
                          className="drop-input"
                          type="file"
                          accept="image/*"
                          onChange={(event) => {
                            setMaskFile(event.target.files?.[0] ?? null);
                            setResult(null);
                          }}
                        />
                      </label>
                      <label className="field" style={{ marginTop: 12 }}>
                        <span>camera JSON</span>
                        <textarea className="input" rows={8} value={cameraJson} onChange={(event) => setCameraJson(event.target.value)} spellCheck={false} />
                      </label>
                    </>
                  ) : (
                    <>
                      <label className="field" style={{ marginTop: 12 }}>
                        <span>文本概念</span>
                        <input className="input" value={text} onChange={(event) => setText(event.target.value)} placeholder="person" />
                      </label>
                      <label className="field">
                        <span>阈值（可空）</span>
                        <input className="input" value={threshold} onChange={(event) => setThreshold(event.target.value)} placeholder="例如 0.5" />
                      </label>
                    </>
                  )}
                  <div className="toolbar">
                    <button type="button" className="btn" onClick={() => void checkHealth()}>
                      健康检查
                    </button>
                    <button type="button" className="btn primary" disabled={busy} onClick={() => void runInfer()}>
                      {busy ? "推理中…" : "开始测试"}
                    </button>
                    {current && (
                      <a className="btn" href={`#/runtime?package=${current.packageId}`}>
                        返回部署
                      </a>
                    )}
                  </div>
                  {health && <p className="ok">健康检查：{health}</p>}
                  {result && !isGenpose && (
                    <p className="ok">
                      检出 {result.scores.length} 个目标
                      {result.scores.length ? `，最高分 ${Math.max(...result.scores).toFixed(3)}` : ""}
                    </p>
                  )}
                  {poseText && (
                    <>
                      <p className="ok">已返回位姿</p>
                      <pre className="manifest">{poseText}</pre>
                    </>
                  )}
                </div>
                <div className="test-stage">
                  {preview ? <canvas ref={canvasRef} className="test-canvas" /> : <p className="muted">选图后会显示原图和分割结果</p>}
                  {result && result.masks.length > 0 && (
                    <div className="mask-thumbs">
                      {result.masks.map((mask, index) => (
                        <img key={index} src={`data:image/png;base64,${mask}`} alt={`mask ${index}`} />
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </section>
          )}
          <PythonSample
            sample={
              isWebsocket
                ? pythonSamplePi05(current?.endpoint ?? "")
                : isVllm
                  ? pythonSampleVllm(current?.endpoint ?? "", prompt)
                  : isGenpose
                    ? pythonSampleGenpose(current?.endpoint ?? "", file?.name ?? "", depthFile?.name ?? "", maskFile?.name ?? "", cameraJson)
                    : pythonSample(current?.endpoint ?? "", text, threshold, file?.name ?? "")
            }
            hint={
              isWebsocket
                ? "直连容器映射出来的宿主机端口，不经过交付台。换机器调用时把 host 改成这台机体的局域网地址。图像换成 uint8 的 HWC，state 用对应机器人的 8 维向量。"
                : isVllm
                  ? "浏览器不能执行这段 Python。要在页面试，用上面的按钮。把脚本拷到本机解释器才能直连容器端口；换机器时把 127.0.0.1 改成这台机体的局域网地址。"
                  : undefined
            }
            copied={copied}
            onCopy={(sample) => void copySample(sample)}
          />
        </>
      )}
    </div>
  );
}

function pythonSamplePi05(endpoint: string): string {
  let host = "127.0.0.1";
  let port = "8000";
  try {
    const url = new URL(endpoint || "ws://127.0.0.1:8000");
    host = url.hostname || host;
    port = url.port || port;
  } catch {
    /* 端点不是 URL 时用默认值 */
  }
  return `import numpy as np
from openpi_client import websocket_client_policy

# LIBERO: observation/image、observation/wrist_image、observation/state（8 维）
# DROID:  observation/exterior_image_1_left、observation/wrist_image_left、
#         observation/joint_position（7 维）、observation/gripper_position
policy = websocket_client_policy.WebsocketClientPolicy(host=${pythonLiteral(host)}, port=${port})
result = policy.infer({
    "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
    "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
    "observation/state": np.zeros(8, dtype=np.float32),
    "prompt": "pick up the cup",
})
print(result["actions"].shape)
`;
}

function pythonSampleVllm(endpoint: string, text: string): string {
  const url = endpoint.replace(/\/$/, "") || "http://127.0.0.1:8000";
  const prompt = text.trim() || "你好";
  return `import json
import urllib.request

URL = ${pythonLiteral(url)}
PROMPT = ${pythonLiteral(prompt)}

health = urllib.request.urlopen(f"{URL}/health", timeout=10)
print("health:", health.status)
models = json.loads(urllib.request.urlopen(f"{URL}/v1/models", timeout=10).read().decode())
model = models["data"][0]["id"]

body = {
    "model": model,
    "messages": [{"role": "user", "content": PROMPT}],
    "max_tokens": 256,
    "temperature": 0,
}
req = urllib.request.Request(
    f"{URL}/v1/chat/completions",
    data=json.dumps(body).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=180) as resp:
    print(json.loads(resp.read().decode()))
`;
}

function pythonSampleGenpose(endpoint: string, rgb: string, depth: string, mask: string, camera: string): string {
  const url = endpoint.replace(/\/$/, "") || "http://127.0.0.1:8081";
  return `import base64
import json
import urllib.request
from pathlib import Path

URL = ${pythonLiteral(url)}
RGB = ${pythonLiteral(rgb || "/path/to/rgb.png")}
DEPTH = ${pythonLiteral(depth || "/path/to/depth.png")}
MASK = ${pythonLiteral(mask || "/path/to/mask.png")}
CAMERA = ${camera.trim() || GENPOSE_CAMERA}

def b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")

health = urllib.request.urlopen(f"{URL}/health", timeout=10)
print("health:", json.loads(health.read().decode()))

body = {"rgb": b64(RGB), "depth": b64(DEPTH), "mask": b64(MASK), "camera": CAMERA}
req = urllib.request.Request(
    f"{URL}/infer",
    data=json.dumps(body).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=180) as resp:
    print(json.loads(resp.read().decode()))
`;
}

function PythonSample({
  sample,
  copied,
  onCopy,
  hint,
}: {
  sample: string;
  copied: boolean;
  onCopy: (sample: string) => void;
  hint?: string;
}) {
  return (
    <section className="section">
      <div className="slot-head">
        <h2 className="section-title" style={{ margin: 0 }}>
          客户端样例（Python）
        </h2>
        <button type="button" className="btn small" onClick={() => onCopy(sample)}>
          {copied ? "已复制" : "复制"}
        </button>
      </div>
      <p className="hint">
        {hint ?? "直连容器映射出来的宿主机端口，不经过交付台。换机器调用时把 URL 里的 127.0.0.1 改成这台机体的局域网地址。IMAGE 改成脚本所在机器上的图片路径。"}
      </p>
      <pre className="manifest">{sample}</pre>
    </section>
  );
}
