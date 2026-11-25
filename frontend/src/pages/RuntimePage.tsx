import { useEffect, useState } from "react";
import { api } from "../api";
import { protocolLabel } from "../format";
import type { Deployment, PackageView } from "../types";

function parseHostPort(raw: string): number | undefined {
  const text = raw.trim();
  if (!text) return undefined;
  const port = Number(text);
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    throw new Error("宿主机端口要填 1 到 65535 的整数，留空则自动分配");
  }
  return port;
}

export function RuntimePage({ focus }: { focus: string | null }) {
  const [packages, setPackages] = useState<PackageView[]>([]);
  const [deploys, setDeploys] = useState<Deployment[]>([]);
  const [hostPorts, setHostPorts] = useState<Record<string, string>>({});
  const [error, setError] = useState("");

  function reload() {
    return Promise.all([api.packages(), api.deploys()]).then(([packageList, deployList]) => {
      setPackages(packageList);
      setDeploys(deployList);
    });
  }

  useEffect(() => {
    reload().catch((reason: Error) => setError(reason.message));
    const timer = window.setInterval(() => {
      reload().catch(() => undefined);
    }, 3000);
    return () => window.clearInterval(timer);
  }, []);

  async function start(view: PackageView) {
    setError("");
    try {
      const hostPort = parseHostPort(hostPorts[view.manifest.metadata.id] ?? "");
      await api.deploy(view.manifest.metadata.id, hostPort);
      await reload();
      location.hash = `#/test?package=${view.manifest.metadata.id}`;
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "部署失败");
    }
  }

  async function stop(packageId: string) {
    setError("");
    try {
      await api.stopDeploy(packageId);
      await reload();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "停止失败");
    }
  }

  const focused = focus ? deploys.find((item) => item.packageId === focus) ?? null : null;

  return (
    <div className="page">
      <p className="page-kicker">本机</p>
      <h1 className="page-title">部署</h1>
      <p className="page-lead">
        每个模型包对应一个容器。容器内端口固定，宿主机端口可以指定，留空则自动分配。调用直接连这个宿主机端口。部署使用已经构建好的镜像，不再构建。
      </p>
      {error && <p className="error">{error}</p>}

      <div className="facts">
        <div className="fact">
          <span>调用</span>
          <strong>宿主机端口</strong>
          <span>HTTP 和 WebSocket 都直接连这里</span>
        </div>
        <div className="fact">
          <span>权重</span>
          <strong>本机路径</strong>
          <span>挂进容器，不复制进镜像</span>
        </div>
        <div className="fact">
          <span>运行时</span>
          <strong>已有镜像</strong>
          <span>部署时不再构建</span>
        </div>
      </div>

      <section className="section">
        <h2 className="section-title">可部署的模型包</h2>
        {packages.length === 0 ? (
          <p className="empty">
            还没有模型包。先去<a href="#/new">装配一份</a>。
          </p>
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>名称</th>
                <th>镜像</th>
                <th>协议</th>
                <th>端口号</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {packages.map((item) => {
                const id = item.manifest.metadata.id;
                const current = deploys.find((deploy) => deploy.packageId === id && deploy.status !== "stopped");
                const blocking = current && (current.status === "running" || current.status === "prepared");
                return (
                  <tr key={id}>
                    <td>
                      <a href={`#/packages/${id}`}>
                        <strong>{item.manifest.metadata.name}</strong>
                      </a>
                      <div className="hint">{item.manifest.model.displayName}</div>
                    </td>
                    <td className="mono">{item.manifest.engine.image}</td>
                    <td>{protocolLabel(item.manifest.engine.protocol)}</td>
                    <td>
                      {blocking && current?.hostPort ? (
                        <span className="mono">{current.hostPort}</span>
                      ) : (
                        <input
                          className="input port-input"
                          inputMode="numeric"
                          placeholder={current?.hostPort ? String(current.hostPort) : "自动"}
                          value={hostPorts[id] ?? ""}
                          onChange={(event) => setHostPorts((ports) => ({ ...ports, [id]: event.target.value }))}
                          aria-label="端口号，留空自动分配"
                        />
                      )}
                    </td>
                    <td>
                      {blocking && current ? (
                        <span className="toolbar">
                          <a className="btn small" href={`#/runtime?package=${id}`}>
                            {statusLabel(current.status)}
                          </a>
                          {current.status === "running" && (
                            <a className="btn primary small" href={`#/test?package=${id}`}>
                              测试
                            </a>
                          )}
                        </span>
                      ) : (
                        <span className="toolbar">
                          {current?.status === "failed" && (
                            <a className="btn small" href={`#/runtime?package=${id}`}>
                              失败
                            </a>
                          )}
                          <button type="button" className="btn primary small" onClick={() => void start(item)}>
                            {current?.status === "failed" ? "重新部署" : "部署"}
                          </button>
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </section>

      {focused && <ServicePanel service={focused} onStop={() => void stop(focused.packageId)} />}
    </div>
  );
}

function ServicePanel({ service, onStop }: { service: Deployment; onStop: () => void }) {
  const health = service.endpoint ? `${service.endpoint}${service.protocol === "websocket" ? "" : "/health"}` : "还没有映射端口";
  return (
    <section className="section">
      <div className="slot-head">
        <h2 className="section-title" style={{ margin: 0 }}>
          {service.name}
        </h2>
        <span className={`badge ${service.status === "prepared" ? "ready" : service.status}`}>{statusLabel(service.status)}</span>
      </div>
      <p className="hint">
        本机 · {service.image} · {protocolLabel(service.protocol)}
        {service.pipeline ? " · 使用权重旁的 pipeline.py" : ""}
      </p>
      <dl className="call-list">
        <div>
          <dt>调用入口</dt>
          <dd className="mono">{service.endpoint ?? "还没有映射端口"}</dd>
        </div>
        <div>
          <dt>健康检查</dt>
          <dd className="mono">{health}</dd>
        </div>
        <div>
          <dt>端口</dt>
          <dd className="mono">
            容器 {service.containerPort}
            {service.hostPort ? ` · 宿主机 ${service.hostPort}` : ""}
          </dd>
        </div>
      </dl>
      <ol className="log-list">
        {service.logs.map((line, index) => (
          <li key={`${index}-${line}`}>{line}</li>
        ))}
      </ol>
      <div className="toolbar">
        <button type="button" className="btn" disabled={service.status === "stopped" || service.status === "prepared"} onClick={onStop}>
          停止
        </button>
        {service.status === "running" && (
          <a className="btn primary" href={`#/test?package=${service.packageId}`}>
            测试
          </a>
        )}
        <a className="btn" href={`#/packages/${service.packageId}`}>
          查看模型包
        </a>
      </div>
    </section>
  );
}

function statusLabel(status: Deployment["status"]): string {
  if (status === "running") return "运行中";
  if (status === "prepared") return "待启动";
  if (status === "failed") return "失败";
  return "已停止";
}
