import { useEffect, useState } from "react";
import { api } from "../api";
import { flow, formatTime } from "../format";
import type { ModelDef, PackageView } from "../types";

export function PackagesPage() {
  const [packages, setPackages] = useState<PackageView[]>([]);
  const [models, setModels] = useState<ModelDef[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    Promise.all([api.packages(), api.models()])
      .then(([packageList, modelList]) => {
        setPackages(packageList);
        setModels(modelList);
      })
      .catch((reason: Error) => setError(reason.message));
  }, []);

  return (
    <div className="page">
      <p className="page-kicker">可部署的装配结果</p>
      <h1 className="page-title">模型包</h1>
      <p className="page-lead">每份模型包是一个目录，里面只有清单。参数仍留在原来的位置，部署时再挂进容器。</p>
      {error && <p className="error">{error}</p>}
      {packages.length === 0 ? (
        <div className="section">
          <p className="empty">还没有模型包。</p>
          <a className="btn primary" href="#/new">
            装配第一份
          </a>
        </div>
      ) : (
        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr>
                <th>名称</th>
                <th>模型</th>
                <th>参数</th>
                <th>生成时间</th>
              </tr>
            </thead>
            <tbody>
              {packages.map((item) => {
                const manifest = item.manifest;
                const named = models.find((model) => model.id === manifest.model.id);
                const quants = [...new Set(manifest.weights.map((weight) => weight.quant).filter((quant): quant is string => Boolean(quant)))].join("、");
                return (
                  <tr key={manifest.metadata.id}>
                    <td>
                      <a href={`#/packages/${manifest.metadata.id}`}>
                        <strong>{manifest.metadata.name}</strong>
                      </a>
                      <div className="hint">{manifest.metadata.version}</div>
                    </td>
                    <td>
                      {named?.displayName ?? manifest.model.displayName ?? manifest.model.id}
                      {named ? <div className="hint">{flow(named.interfaces.inputs, named.interfaces.outputs)}</div> : null}
                    </td>
                    <td>
                      {manifest.weights.length} 个
                      {quants ? ` · ${quants}` : ""}
                    </td>
                    <td>{formatTime(manifest.metadata.createdAt ?? null)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
