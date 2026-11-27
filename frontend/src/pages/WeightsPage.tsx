import { useEffect, useState } from "react";
import { api, uploadWeight } from "../api";
import { formatBytes, sourceLabel } from "../format";
import type { MetaInfo, WeightAsset } from "../types";

type WeightTab = "library" | "upload" | "s3";

export function WeightsPage() {
  const [assets, setAssets] = useState<WeightAsset[]>([]);
  const [meta, setMeta] = useState<MetaInfo | null>(null);
  const [tab, setTab] = useState<WeightTab>("library");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [progress, setProgress] = useState<number | null>(null);
  const [s3Uri, setS3Uri] = useState("s3://");
  const [s3Endpoint, setS3Endpoint] = useState("");
  const [s3Region, setS3Region] = useState("");
  const [s3Note, setS3Note] = useState("");
  const [localPath, setLocalPath] = useState("");
  const [localNote, setLocalNote] = useState("");
  const [loaded, setLoaded] = useState(false);

  async function reload() {
    const [weightList, info] = await Promise.all([api.weights(), api.meta()]);
    setAssets(weightList);
    setMeta(info);
    setLoaded(true);
  }

  useEffect(() => {
    reload().catch((reason: Error) => {
      setError(reason.message);
      setLoaded(true);
    });
  }, []);

  const grouped = {
    library: assets.filter((asset) => asset.origin === "library" || asset.origin === "registered"),
    upload: assets.filter((asset) => asset.origin === "upload"),
    s3: assets.filter((asset) => asset.origin === "s3"),
  };

  async function onUpload(file: File | undefined) {
    if (!file) return;
    setError("");
    setNotice("");
    setProgress(0);
    try {
      const asset = await uploadWeight(file, setProgress);
      await reload();
      setNotice(`已上传 ${asset.name}`);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "上传失败");
    } finally {
      setProgress(null);
    }
  }

  async function onRegisterS3() {
    setError("");
    try {
      await api.registerS3({ uri: s3Uri, endpoint: s3Endpoint, region: s3Region, note: s3Note });
      await reload();
      setNotice("已登记 S3 地址，文件留在对象存储里。");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "登记失败");
    }
  }

  async function onVerifyS3() {
    setError("");
    setNotice("");
    try {
      const result = await api.verifyS3({ uri: s3Uri, endpoint: s3Endpoint, region: s3Region });
      setNotice(result.ok ? `可以访问，大小 ${formatBytes(result.sizeBytes)}。` : result.detail || "没能访问");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "检查失败");
    }
  }

  async function onRegisterPath() {
    setError("");
    try {
      await api.registerPath({ path: localPath, note: localNote });
      await reload();
      setNotice("已登记本机路径。");
      setLocalPath("");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "登记失败");
    }
  }

  async function onDeleteAsset(asset: WeightAsset) {
    const message = asset.source === "upload" ? `删除已上传的 ${asset.name}？` : `取消登记 ${asset.name}？磁盘上的文件不会删。`;
    if (!window.confirm(message)) return;
    setError("");
    try {
      await api.deleteWeight(asset.id);
      await reload();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "删除失败");
    }
  }

  return (
    <div className="page">
      <p className="page-kicker">先把权重量出来</p>
      <h1 className="page-title">权重</h1>
      <p className="page-lead">
        登记本机路径、上传文件，或记下 S3 地址。文件留在原地。之后可以去<a href="#/quant">量化</a>，或直接到<a href="#/new">装配</a>页选用。
      </p>
      {error && <p className="error">{error}</p>}
      {notice && <p className="ok">{notice}</p>}
      {!loaded ? (
        <p className="muted">正在读取…</p>
      ) : (
        <section className="section">
          <p className="hint">
            本地库目录：<span className="mono">{meta?.localLibrary ?? "…"}</span>
          </p>
          <div className="tabs">
            <button type="button" className={tab === "library" ? "tab on" : "tab"} onClick={() => setTab("library")}>
              本地
            </button>
            <button type="button" className={tab === "upload" ? "tab on" : "tab"} onClick={() => setTab("upload")}>
              上传
            </button>
            <button type="button" className={tab === "s3" ? "tab on" : "tab"} onClick={() => setTab("s3")}>
              S3
            </button>
          </div>

          {tab === "library" && (
            <div>
              <p className="hint">填这台交付台 Linux 机体上的路径，不是浏览器那台电脑的盘符。登记不会移动文件。</p>
              <p className="hint">
                单文件权重（如 SAM 3）填到文件，例如 <span className="mono path-example">/data/weights/sam3.pt</span>。整目录权重（如 π0.5）填目录。
              </p>
              <div className="form-grid">
                <div className="field">
                  <label htmlFor="local-path">本机路径</label>
                  <input
                    id="local-path"
                    className="input"
                    value={localPath}
                    onChange={(event) => setLocalPath(event.target.value)}
                    placeholder="/data/weights/sam3.pt"
                    spellCheck={false}
                  />
                </div>
                <div className="field">
                  <label htmlFor="local-note">备注</label>
                  <input id="local-note" className="input" value={localNote} onChange={(event) => setLocalNote(event.target.value)} />
                </div>
              </div>
              <div className="toolbar" style={{ marginTop: 10 }}>
                <button type="button" className="btn" onClick={() => void reload().catch((reason: Error) => setError(reason.message))}>
                  刷新本地库
                </button>
                <button type="button" className="btn primary" onClick={() => void onRegisterPath()}>
                  登记路径
                </button>
              </div>
              <AssetList assets={grouped.library} onDelete={onDeleteAsset} />
            </div>
          )}

          {tab === "upload" && (
            <div>
              <label
                className="drop"
                onDragOver={(event) => event.preventDefault()}
                onDrop={(event) => {
                  event.preventDefault();
                  void onUpload(event.dataTransfer.files[0]);
                }}
              >
                <span>把文件拖到这里，或点击选择</span>
                <span className="hint">上传后存在这台机体的 data/library/uploads</span>
                <input className="drop-input" type="file" onChange={(event) => void onUpload(event.target.files?.[0])} />
              </label>
              {progress != null && <p className="hint">上传中 {Math.round(progress * 100)}%</p>}
              <AssetList assets={grouped.upload} onDelete={onDeleteAsset} />
            </div>
          )}

          {tab === "s3" && (
            <div>
              <p className="hint">只记录地址，不在这里下载。检查连通会用这台机体上的访问凭证，密钥不会写进模型包。</p>
              <div className="form-grid">
                <div className="field span-2">
                  <label htmlFor="s3-uri">S3 地址</label>
                  <input id="s3-uri" className="input" value={s3Uri} onChange={(event) => setS3Uri(event.target.value)} spellCheck={false} />
                </div>
                <div className="field">
                  <label htmlFor="s3-endpoint">Endpoint，MinIO 才需要</label>
                  <input
                    id="s3-endpoint"
                    className="input"
                    value={s3Endpoint}
                    onChange={(event) => setS3Endpoint(event.target.value)}
                    placeholder="http://127.0.0.1:9000"
                    spellCheck={false}
                  />
                </div>
                <div className="field">
                  <label htmlFor="s3-region">Region</label>
                  <input id="s3-region" className="input" value={s3Region} onChange={(event) => setS3Region(event.target.value)} />
                </div>
                <div className="field span-2">
                  <label htmlFor="s3-note">备注</label>
                  <input id="s3-note" className="input" value={s3Note} onChange={(event) => setS3Note(event.target.value)} />
                </div>
              </div>
              <div className="toolbar" style={{ marginTop: 10 }}>
                <button type="button" className="btn" onClick={() => void onVerifyS3()}>
                  检查能否访问
                </button>
                <button type="button" className="btn primary" onClick={() => void onRegisterS3()}>
                  登记地址
                </button>
              </div>
              <AssetList assets={grouped.s3} onDelete={onDeleteAsset} />
            </div>
          )}
        </section>
      )}
    </div>
  );
}

function AssetList({ assets, onDelete }: { assets: WeightAsset[]; onDelete: (asset: WeightAsset) => void }) {
  if (assets.length === 0) return <p className="empty">这里还没有权重。</p>;
  return (
    <table className="table">
      <tbody>
        {assets.map((asset) => (
          <tr key={asset.id}>
            <td>
              <span className={asset.missing ? "badge missing" : `badge ${asset.source}`}>{sourceLabel(asset.source)}</span>
            </td>
            <td>
              <div>{asset.name}</div>
              <div className="uri">{asset.uri}</div>
              {asset.note && <div className="hint">{asset.note}</div>}
            </td>
            <td>{asset.missing ? "路径失效" : formatBytes(asset.sizeBytes)}</td>
            <td>
              {asset.removable && (
                <button type="button" className="btn small danger" onClick={() => onDelete(asset)}>
                  {asset.source === "upload" ? "删除" : "取消登记"}
                </button>
              )}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
