import { useEffect, useState } from "react";
import { api } from "../api";
import type { BuildJob } from "../types";
import { FlowSteps } from "./FlowSteps";

export function BuildPage({ modelId, image }: { modelId: string | null; image: string | null }) {
  const [target, setTarget] = useState(image ?? "");
  const [job, setJob] = useState<BuildJob | null>(null);
  const [error, setError] = useState("");
  const [starting, setStarting] = useState(false);

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
    };
    pull();
    const timer = window.setInterval(pull, 1000);
    return () => {
      stop = true;
      window.clearInterval(timer);
    };
  }, [target]);

  async function start() {
    if (!target) return;
    setStarting(true);
    setError("");
    try {
      setJob(await api.startBuild(target));
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

  return (
    <div className="page">
      <p className="page-kicker">第一次构建</p>
      <h1 className="page-title">构建运行时</h1>
      <p className="page-lead">
        构建 {target || "这张镜像"}。本机已经有它时会直接跳过。拉基础镜像和安装依赖会比较久，这一步在后台跑，装配页不会触发它。
      </p>
      {target === "local/sam3-engine:latest" && (
        <p className="hint">
          本机还没有 SAM 3 模型代码时，会先从 https://github.com/facebookresearch/sam3 clone 到项目根目录的 tmp/sam3，已有则跳过。权重不在这次 clone 里。
        </p>
      )}
      {error && <p className="error">{error}</p>}
      <FlowSteps active={status === "ready" ? 2 : 1} />
      <section className="section">
        <div className="slot-head">
          <h2 className="section-title" style={{ margin: 0 }}>
            {target || "未选择镜像"}
          </h2>
          <span className={`badge ${status === "ready" ? "ready" : status === "failed" ? "stopped" : running ? "starting" : "stopped"}`}>
            {status === "ready" ? "本机已有" : status === "failed" ? "构建失败" : running ? "构建中" : "未构建"}
          </span>
        </div>
        <ol className="log-list">
          {(job?.logs ?? ["还没有开始。"]).map((line, index) => (
            <li key={`${index}-${line}`}>{line}</li>
          ))}
        </ol>
        <div className="toolbar">
          {status === "ready" ? (
            <a className="btn primary" href="#/new">
              进入装配
            </a>
          ) : (
            <button type="button" className="btn primary" disabled={!target || running} onClick={() => void start()}>
              {running ? "构建中" : "开始构建"}
            </button>
          )}
        </div>
      </section>
    </div>
  );
}
