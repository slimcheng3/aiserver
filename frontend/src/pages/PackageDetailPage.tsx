import { useEffect, useState } from "react";
import { api } from "../api";
import { flow, formatBytes, formatTime, protocolLabel, sourceLabel } from "../format";
import type { PackageView } from "../types";

export function PackageDetailPage({ id }: { id: string }) {
  const [item, setItem] = useState<PackageView | null>(null);
  const [error, setError] = useState("");
  const [copied, setCopied] = useState("");

  useEffect(() => {
    setItem(null);
    api.package(id).then(setItem).catch((reason: Error) => setError(reason.message));
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

  async function deploy(target: "local" | "edge") {
    setError("");
    try {
      await api.deploy(id, target);
      location.hash = `#/runtime?package=${id}`;
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
  const separator = item.absoluteRoot.includes("\\") ? "\\" : "/";
  const manifestPath = `${item.absoluteRoot}${separator}manifest.json`;

  return (
    <div className="page">
      <p className="page-kicker">{manifest.metadata.id}</p>
      <h1 className="page-title">{manifest.metadata.name}</h1>
      <p className="page-lead">
        {manifest.metadata.description || "这份模型包只描述怎么部署，不包含参数大文件。"} 版本 {manifest.metadata.version} ·{" "}
        {formatTime(manifest.metadata.createdAt)}
      </p>
      {error && <p className="error">{error}</p>}
      <div className="facts">
        <div className="fact">
          <span>模型</span>
          <strong>{manifest.model.displayName}</strong>
          <span>{flow(manifest.model.interfaces.inputs, manifest.model.interfaces.outputs)}</span>
        </div>
        <div className="fact">
          <span>运行时</span>
          <strong>{manifest.engine.image}</strong>
          <span>
            {protocolLabel(manifest.engine.protocol)} · {manifest.engine.port}
          </span>
        </div>
        <div className="fact">
          <span>资源</span>
          <strong>
            GPU {manifest.engine.resources.gpu} · {manifest.engine.resources.gpuMemoryGi} GiB
          </strong>
          <span>
            CPU {manifest.engine.resources.cpu} · 内存 {manifest.engine.resources.memoryGi} GiB
          </span>
        </div>
      </div>
      <section className="section">
        <h2 className="section-title">部署到本机或边侧</h2>
        <p className="hint">目标可以是这台机体，也可以是边侧。两边都使用已经构建好的镜像。本机挂载清单里的路径；边侧要求权重已经是 S3 地址，并且不在本机启动容器。</p>
        <ul className="check-list">
          <li>运行时镜像 {manifest.engine.image} 需要已经构建，部署时不再构建</li>
          <li>参数 {manifest.weights.map((weight) => weight.name).join("、") || "未绑定"} 留在原处</li>
          <li>容器内端口 {manifest.engine.port} 会映射到宿主机的一个空闲端口，调用直接连那里</li>
        </ul>
        <div className="toolbar">
          <button type="button" className="btn primary" onClick={() => void deploy("local")}>
            部署到本机
          </button>
          <button type="button" className="btn" onClick={() => void deploy("edge")}>
            部署到边侧
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
          <a className="btn primary" href={`/api/packages/${id}/manifest`}>
            下载 manifest
          </a>
          <button type="button" className="btn danger" onClick={() => void remove()}>
            删除包
          </button>
        </div>
        {manifest.engine.codeRef && (
          <p className="hint">推理代码：{manifest.engine.codeRef}。装配台只记录这个位置，不复制代码。</p>
        )}
        <p className="hint">启动命令：{manifest.engine.command.join(" ")}</p>
        {manifest.engine.models && manifest.engine.models.length > 0 && (
          <p className="hint">
            这份运行时支持的模型：
            {manifest.engine.models
              .map((id) => (id === manifest.model.id ? manifest.model.displayName : id))
              .join("、")}
          </p>
        )}
        {manifest.engine.weightQuants && manifest.engine.weightQuants.length > 0 && (
          <p className="hint">这份运行时支持的权重：{manifest.engine.weightQuants.join("、")}</p>
        )}
      </section>
      <section className="section">
        <h2 className="section-title">参数</h2>
        <table className="table">
          <thead>
            <tr>
              <th>角色</th>
              <th>量化</th>
              <th>来源</th>
              <th>地址</th>
              <th>大小</th>
            </tr>
          </thead>
          <tbody>
            {manifest.weights.map((weight) => (
              <tr key={weight.role}>
                <td>{weight.label}</td>
                <td>{weight.quant ?? "—"}</td>
                <td>
                  <span className={`badge ${weight.source}`}>{sourceLabel(weight.source)}</span>
                </td>
                <td className="mono">{weight.uri}</td>
                <td>{formatBytes(weight.sizeBytes)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {manifest.weights.some((weight) => weight.source === "local" || weight.source === "upload") && (
          <p className="warn">本机路径只在这台机体上有效。部署到边侧时改用 S3 地址。</p>
        )}
      </section>
      {Object.keys(manifest.engine.env).length > 0 && (
        <section className="section">
          <h2 className="section-title">环境变量</h2>
          <pre className="manifest">{Object.entries(manifest.engine.env).map(([key, value]) => `${key}=${value}`).join("\n")}</pre>
        </section>
      )}
      <section className="section">
        <h2 className="section-title">manifest.json</h2>
        <pre className="manifest">{JSON.stringify(manifest, null, 2)}</pre>
      </section>
    </div>
  );
}
