import { useEffect, useState } from "react";
import { api } from "../api";
import { flow, formatTime, protocolLabel, weightPath } from "../format";
import type { ModelDef, PackageView } from "../types";

function parseHostPort(raw: string): number | undefined {
  const text = raw.trim();
  if (!text) return undefined;
  const port = Number(text);
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    throw new Error("宿主机端口要填 1 到 65535 的整数，留空则自动分配");
  }
  return port;
}

export function PackageDetailPage({ id }: { id: string }) {
  const [item, setItem] = useState<PackageView | null>(null);
  const [models, setModels] = useState<ModelDef[]>([]);
  const [hostPort, setHostPort] = useState("");
  const [error, setError] = useState("");
  const [copied, setCopied] = useState("");

  useEffect(() => {
    setItem(null);
    api.package(id).then(setItem).catch((reason: Error) => setError(reason.message));
    api.models().then(setModels).catch(() => undefined);
  }, [id]);

  async function copy(text: string, label: string) {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(label);
    } catch {
      setError("复制失败，请手动选择下面的路径或清单");
    }
  }

  async function remove() {
    if (!item) return;
    if (!window.confirm(`删除模型包「${item.manifest.metadata.name}」？参数文件会留在原处。`)) return;
    await api.deletePackage(id);
    location.hash = "#/packages";
  }

  async function deploy() {
    setError("");
    try {
      await api.deploy(id, parseHostPort(hostPort));
      location.hash = `#/test?package=${id}`;
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "部署失败");
    }
  }

  if (!item) {
    return (
      <div className="page">
        {error ? <p className="error">{error}</p> : <p className="muted">正在读取模型包…</p>}
      </div>
    );
  }

  const manifest = item.manifest;
  const named = models.find((model) => model.id === manifest.model.id);
  const modelName = named?.displayName ?? manifest.model.displayName ?? manifest.model.id;
  const interfaces = named?.interfaces ?? manifest.model.interfaces;
  const separator = item.absoluteRoot.includes("\\") ? "\\" : "/";
  const manifestPath = `${item.absoluteRoot}${separator}manifest.json`;
  const remoteWeight = manifest.weights.some((weight) => weightPath(weight).startsWith("s3://"));
  const manifestName = `${manifest.metadata.name}.manifest.json`;
  const archiveName = `${manifest.metadata.name}.zip`;

  return (
    <div className="page">
      <p className="page-kicker">{manifest.metadata.id}</p>
      <h1 className="page-title">{manifest.metadata.name}</h1>
      <p className="page-lead">
        版本 {manifest.metadata.version}
        {manifest.metadata.createdAt ? ` · ${formatTime(manifest.metadata.createdAt)}` : ""}
      </p>
      {error && <p className="error">{error}</p>}
      <div className="facts">
        <div className="fact">
          <span>模型</span>
          <strong>{modelName}</strong>
          {interfaces ? <span>{flow(interfaces.inputs, interfaces.outputs)}</span> : null}
        </div>
        <div className="fact">
          <span>运行时</span>
          <strong>{manifest.engine.image}</strong>
          <span>
            {protocolLabel(manifest.engine.protocol)} · {manifest.engine.port}
          </span>
        </div>
      </div>
      <section className="section">
        <h2 className="section-title">部署到本机</h2>
        <p className="hint">在这台机体上启动容器，挂载清单里的本机路径。使用已经构建好的镜像，部署时不再构建。宿主机端口可指定，留空则自动分配。</p>
        <ul className="check-list">
          <li>运行时镜像 {manifest.engine.image} 需要已经构建，部署时不再构建</li>
          <li>参数 {manifest.weights.map((weight) => weightPath(weight).split(/[/\\]/).pop() || weight.role).join("、") || "未绑定"} 留在原处</li>
          <li>容器内端口 {manifest.engine.port} 映射到宿主机端口；填写端口则使用该端口，留空则自动分配</li>
        </ul>
        <div className="toolbar">
          <label className="field" style={{ minWidth: 140 }}>
            <span>端口号</span>
            <input
              className="input port-input"
              inputMode="numeric"
              placeholder="自动"
              value={hostPort}
              onChange={(event) => setHostPort(event.target.value)}
            />
          </label>
          <button type="button" className="btn primary" onClick={() => void deploy()}>
            部署到本机
          </button>
        </div>
      </section>
      <section className="section">
        <h2 className="section-title">清单</h2>
        <p className="mono">{manifestPath}</p>
        <div className="toolbar" style={{ marginTop: 12 }}>
          <button type="button" className="btn" onClick={() => copy(manifestPath, "路径")}>
            {copied === "路径" ? "已复制路径" : "复制路径"}
          </button>
          <button type="button" className="btn" onClick={() => copy(JSON.stringify(manifest, null, 2), "清单")}>
            {copied === "清单" ? "已复制清单" : "复制清单"}
          </button>
          <a className="btn primary" href={`/api/packages/${id}/manifest`} download={manifestName}>
            下载清单
          </a>
          {remoteWeight ? (
            <button type="button" className="btn" disabled>
              下载压缩包
            </button>
          ) : (
            <a className="btn" href={`/api/packages/${id}/archive`} download={archiveName}>
              下载压缩包
            </a>
          )}
          <button type="button" className="btn danger" onClick={() => void remove()}>
            删除包
          </button>
        </div>
        {remoteWeight ? (
          <p className="hint">S3 地址不能打进下载包。下载清单得到 {manifestName}，只有清单。</p>
        ) : (
          <p className="hint">
            下载清单得到 {manifestName}，路径是这台机器上的原路径。下载压缩包得到 {archiveName}，包里的 path 改成压缩包内的相对路径。
          </p>
        )}
        <p className="hint">启动命令：{manifest.engine.command.join(" ")}</p>
        <p className="hint">容器内权重：{manifest.engine.weightsPath || "/weights/checkpoint"}</p>
        {manifest.engine.resources ? (
          <p className="hint">
            资源：加速卡 {manifest.engine.resources.accelerators ?? manifest.engine.resources.gpu ?? 0} ·{" "}
            {manifest.engine.resources.acceleratorMemoryGi ?? manifest.engine.resources.gpuMemoryGi ?? 0} GiB · CPU{" "}
            {manifest.engine.resources.cpu} · {manifest.engine.resources.memoryGi} GiB
          </p>
        ) : null}
      </section>
      <section className="section">
        <h2 className="section-title">参数</h2>
        <table className="table">
          <thead>
            <tr>
              <th>角色</th>
              <th>量化</th>
              <th>路径</th>
            </tr>
          </thead>
          <tbody>
            {manifest.weights.map((weight) => (
              <tr key={weight.role}>
                <td>{weight.role}</td>
                <td>{weight.quant ?? "—"}</td>
                <td className="mono">{weightPath(weight)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {manifest.weights.some((weight) => weightPath(weight).startsWith("s3://")) && (
          <p className="warn">S3 地址不能挂进本机容器。部署请改用本机路径或上传的权重。</p>
        )}
      </section>
      <section className="section">
        <h2 className="section-title">manifest.json</h2>
        <pre className="manifest">{JSON.stringify(manifest, null, 2)}</pre>
      </section>
    </div>
  );
}
