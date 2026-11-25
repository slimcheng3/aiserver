import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { BuildJob } from "../types";
import { FlowSteps } from "./FlowSteps";

export function BuildPage({ modelId, image }: { modelId: string | null; image: string | null }) {
  const [target, setTarget] = useState(image ?? "");
  const [job, setJob] = useState<BuildJob | null>(null);
  const [imageReady, setImageReady] = useState(false);
  const [error, setError] = useState("");
  const [starting, setStarting] = useState(false);
  const logRef = useRef<HTMLOListElement>(null);

  useEffect(() => {
    if (image) {
      setTarget(image);
      return;
    }
    if (!modelId) return;
    api
      .models()
      .then((models) => {
        const found = models.find((item) => item.id === modelId);
        if (!found) {
          setError("未知模型");
          return;
        }
        setTarget(found.engine.image);
      })
      .catch((reason: Error) => setError(reason.message));
  }, [modelId, image]);

  useEffect(() => {
    if (!target) return;
    let stop = false;
    const pull = () => {
      api
        .builds(target)
        .then((jobs) => {
          if (!stop) setJob(jobs[0] ?? null);
        })
        .catch((reason: Error) => {
          if (!stop) setError(reason.message);
        });
      api
        .environment()
        .then((env) => {
          if (!stop) setImageReady(Boolean(env.images[target]));
        })
        .catch((reason: Error) => {
          if (!stop) setError(reason.message);
        });
    };
    pull();
    const timer = window.setInterval(pull, 800);
    return () => {
      stop = true;
      window.clearInterval(timer);
    };
  }, [target]);

  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [job?.logs]);

  async function start(force = false) {
    if (!target) return;
    setStarting(true);
    setError("");
    try {
      setJob(await api.startBuild(target, force));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "构建没有开始");
    } finally {
      setStarting(false);
    }
  }

  if (!modelId && !image) {
    return (
      <div className="page">
        <p className="muted">先从运行时里选一张镜像。</p>
      </div>
    );
  }

  const status = job?.status ?? "missing";
  const running = status === "running" || status === "queued" || starting;
  const built = status === "ready" && imageReady;
  const stale = status === "ready" && !imageReady;
  const progress = progressFromLogs(job?.logs ?? []);
  const lines = stale
    ? ["镜像已经不在本机。上一次的构建记录不能再用，点开始构建会重新打镜像。"]
    : (job?.logs ?? ["还没有开始。"]);

  return (
    <div className="page">
      <p className="page-kicker">第一次构建</p>
      <h1 className="page-title">构建运行时</h1>
      <p className="page-lead">
        {running && imageReady
          ? `正在重新构建 ${target || "这张镜像"}。成功后会换掉本机上的旧镜像。没改过的层会用缓存。已有容器要重新部署才换得上。`
          : `构建 ${target || "这张镜像"}。本机已经有它时，普通构建会直接跳过；要换进新的服务代码，用重新构建。拉基础镜像和安装依赖会比较久。Docker 输出会按行刷到下面，大约每秒更新。`}
      </p>
      {target === "local/sam3-engine:latest" && (
        <p className="hint">推理代码已经在 engines/sam3/upstream。构建只打镜像，不克隆仓库，权重也不进镜像。</p>
      )}
      {target === "local/genpose2-engine:latest" && (
        <p className="hint">推理代码已经在 engines/genpose2/upstream。构建会编译 PointNet2，并把 DINOv2 打进镜像。三个位姿 pth 仍不进镜像。</p>
      )}
      {target === "local/vllm-engine:latest" && (
        <p className="hint">基于官方 vllm-openai（默认从 DaoCloud 拉，避开 Docker Hub 429）。入口是 OpenAI API（/v1/chat/completions），不是项目自己的 /infer。权重不进镜像。</p>
      )}
      {target === "local/flashrt-engine:latest" && (
        <p className="hint">推理代码已经在 engines/flashrt/upstream。构建按本机架构编译内核（可用 AISERVER_FLASHRT_GPU_ARCH 覆盖），不克隆仓库，权重也不进镜像。</p>
      )}
      {error && <p className="error">{error}</p>}
      <FlowSteps active={built ? 2 : 1} />
      <section className="section">
        <div className="slot-head">
          <h2 className="section-title" style={{ margin: 0 }}>
            {target || "未选择镜像"}
          </h2>
          <span className={`badge ${built ? "ready" : status === "failed" ? "stopped" : running ? "starting" : "stopped"}`}>
            {built ? "本机已有" : status === "failed" && !stale ? "构建失败" : running ? "构建中" : "未构建"}
          </span>
        </div>
        {running && progress.label && (
          <div className="build-progress">
            <p className="hint" style={{ margin: "0 0 8px" }}>
              {progress.label}
              {progress.pct != null ? ` · ${progress.pct}%` : ""}
            </p>
            {progress.pct != null && (
              <div className="build-progress-bar">
                <span style={{ width: `${progress.pct}%` }} />
              </div>
            )}
          </div>
        )}
        <ol className="log-list" ref={logRef}>
          {lines.map((line, index) => (
            <li key={`${index}-${line}`}>{line}</li>
          ))}
        </ol>
        <div className="toolbar">
          {built ? (
            <a className="btn primary" href="#/new">
              进入装配
            </a>
          ) : null}
          {imageReady && target ? (
            <a className="btn" href={`/api/runtimes/tar?image=${encodeURIComponent(target)}`}>
              下载 tar
            </a>
          ) : null}
          {built || imageReady ? (
            <button type="button" className="btn" disabled={!target || running} onClick={() => void start(true)}>
              {running ? "构建中" : "重新构建"}
            </button>
          ) : (
            <button type="button" className="btn primary" disabled={!target || running} onClick={() => void start(false)}>
              {running ? "构建中" : "开始构建"}
            </button>
          )}
        </div>
      </section>
    </div>
  );
}

function toBytes(value: number, unit: string): number {
  const key = unit.toLowerCase();
  if (key === "gb") return value * 1024 * 1024 * 1024;
  if (key === "mb") return value * 1024 * 1024;
  if (key === "kb") return value * 1024;
  return value;
}

function progressFromLogs(logs: string[]): { label: string; pct: number | null } {
  for (let i = logs.length - 1; i >= 0; i -= 1) {
    const line = logs[i];
    const layer = line.match(/([\d.]+)\s*(kB|MB|GB|B)\s*\/\s*([\d.]+)\s*(kB|MB|GB|B)/i);
    if (layer) {
      const done = toBytes(Number(layer[1]), layer[2]);
      const total = toBytes(Number(layer[3]), layer[4]);
      const pct = total ? Math.min(100, Math.round((100 * done) / total)) : 0;
      return { label: `拉取 ${layer[1]} ${layer[2]} / ${layer[3]} ${layer[4]}`, pct };
    }
    const steps = line.match(/\[(\d+)\s*\/\s*(\d+)\]/);
    if (steps && Number(steps[2]) > 0) {
      const pct = Math.min(100, Math.round((100 * Number(steps[1])) / Number(steps[2])));
      return { label: `编译 ${steps[1]}/${steps[2]}`, pct };
    }
  }
  const last = logs[logs.length - 1] ?? "";
  return { label: last.slice(0, 120), pct: null };
}
