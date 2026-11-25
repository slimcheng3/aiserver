import { useEffect, useRef, useState } from "react";
import { api, uploadWeight } from "../api";
import { flow, formatBytes, protocolLabel, sourceLabel } from "../format";
import type { EngineForm, MetaInfo, ModelDef, RuntimeInfo, WeightAsset } from "../types";

type WeightTab = "library" | "upload" | "s3";

type RuntimeChoice = RuntimeInfo;

function choicesFor(model: ModelDef, runtimes: RuntimeInfo[]): RuntimeChoice[] {
  return runtimes.filter((runtime) => runtime.kind !== "base" && runtime.models.includes(model.id));
}

const LOCKED_FLAGS = new Set(["--model", "--host", "--port"]);

function splitCommand(parts: string[]): { locked: string[]; extra: string[] } {
  const locked: string[] = [];
  const extra: string[] = [];
  let index = 0;
  while (index < parts.length) {
    const token = parts[index];
    if (LOCKED_FLAGS.has(token)) {
      locked.push(token);
      index += 1;
      if (index < parts.length && !parts[index].startsWith("-")) {
        locked.push(parts[index]);
        index += 1;
      }
      continue;
    }
    extra.push(token);
    index += 1;
  }
  return { locked, extra };
}

function engineFromRuntime(model: ModelDef, runtime: RuntimeChoice): EngineForm {
  const { locked, extra } = splitCommand(runtime.command);
  const flagStyle = Boolean(runtime.command[0]?.startsWith("--"));
  return {
    image: runtime.image,
    codeRef: runtime.codeRef,
    command: (flagStyle ? locked : runtime.command).join(" "),
    commandExtra: flagStyle ? extra.join(" ") : "",
    port: runtime.port,
    healthPath: runtime.healthPath,
    protocol: runtime.protocol,
    gpu: model.resources.gpu,
    gpuMemoryGi: model.resources.gpuMemoryGi,
    cpu: model.resources.cpu,
    memoryGi: model.resources.memoryGi,
    envText: Object.entries(model.env)
      .map(([key, value]) => `${key}=${value}`)
      .join("\n"),
  };
}

function parseCommand(input: string): string[] {
  const trimmed = input.trim();
  if (!trimmed) return [];
  if (trimmed.startsWith("[")) {
    const parsed = JSON.parse(trimmed) as unknown;
    if (!Array.isArray(parsed) || parsed.some((item) => typeof item !== "string")) {
      throw new Error("命令 JSON 需要是字符串数组");
    }
    return parsed;
  }
  return trimmed.split(/\s+/);
}

function parseEnv(text: string): Record<string, string> {
  const env: Record<string, string> = {};
  for (const line of text.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const index = trimmed.indexOf("=");
    if (index <= 0) throw new Error(`环境变量需要写成 KEY=VALUE：${trimmed}`);
    env[trimmed.slice(0, index).trim()] = trimmed.slice(index + 1);
  }
  return env;
}

export function NewPackagePage({ initialModel }: { initialModel: string | null }) {
  const [models, setModels] = useState<ModelDef[]>([]);
  const [runtimes, setRuntimes] = useState<RuntimeInfo[]>([]);
  const [assets, setAssets] = useState<WeightAsset[]>([]);
  const [meta, setMeta] = useState<MetaInfo | null>(null);
  const [modelId, setModelId] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [version, setVersion] = useState("1.0.0");
  const [description, setDescription] = useState("");
  const [bindings, setBindings] = useState<Record<string, string>>({});
  const [quants, setQuants] = useState<Record<string, string>>({});
  const [engine, setEngine] = useState<EngineForm | null>(null);
  const [tab, setTab] = useState<WeightTab>("library");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [saving, setSaving] = useState(false);
  const [progress, setProgress] = useState<number | null>(null);
  const [s3Uri, setS3Uri] = useState("s3://");
  const [s3Endpoint, setS3Endpoint] = useState("");
  const [s3Region, setS3Region] = useState("");
  const [s3Note, setS3Note] = useState("");
  const [localPath, setLocalPath] = useState("");
  const [localNote, setLocalNote] = useState("");
  const appliedModel = useRef<string | null>(null);

  useEffect(() => {
    Promise.all([api.models(), api.runtimes(), api.weights(), api.meta()])
      .then(([modelList, runtimeList, weightList, info]) => {
        setModels(modelList);
        setRuntimes(runtimeList);
        setAssets(weightList);
        setMeta(info);
      })
      .catch((reason: Error) => setError(reason.message));
  }, []);

  useEffect(() => {
    if (modelId || models.length === 0) return;
    const wanted = initialModel && models.some((model) => model.id === initialModel) ? initialModel : models[0].id;
    setModelId(wanted);
  }, [models, modelId, initialModel]);

  useEffect(() => {
    if (!modelId || appliedModel.current === modelId) return;
    const selected = models.find((item) => item.id === modelId);
    if (!selected) return;
    const options = choicesFor(selected, runtimes);
    if (options.length === 0) return;
    appliedModel.current = modelId;
    setEngine(engineFromRuntime(selected, options[0]));
    setBindings({});
    setQuants(defaultQuants(selected, options[0]));
  }, [modelId, models, runtimes]);

  const model = models.find((item) => item.id === modelId) ?? null;
  const runtimeOptions = model ? choicesFor(model, runtimes) : [];
  const runtime = runtimeOptions.find((item) => item.image === engine?.image) ?? runtimeOptions[0] ?? null;

  async function refreshAssets(assignId?: string) {
    const list = await api.weights();
    setAssets(list);
    if (!assignId || !model) return;
    setBindings((current) => {
      const next = { ...current };
      const empty = model.artifacts.find((artifact) => !next[artifact.role]);
      if (empty) next[empty.role] = assignId;
      return next;
    });
  }

  async function onUpload(file: File | undefined) {
    if (!file) return;
    setError("");
    setNotice("");
    setProgress(0);
    try {
      const asset = await uploadWeight(file, setProgress);
      await refreshAssets(asset.id);
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
      const asset = await api.registerS3({ uri: s3Uri, endpoint: s3Endpoint, region: s3Region, note: s3Note });
      await refreshAssets(asset.id);
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
      const asset = await api.registerPath({ path: localPath, note: localNote });
      await refreshAssets(asset.id);
      setNotice("已登记本机路径。");
      setLocalPath("");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "登记失败");
    }
  }

  async function onDeleteAsset(asset: WeightAsset) {
    const message =
      asset.source === "upload" ? `删除已上传的 ${asset.name}？` : `取消登记 ${asset.name}？磁盘上的文件不会删。`;
    if (!window.confirm(message)) return;
    setError("");
    try {
      await api.deleteWeight(asset.id);
      setBindings((current) => {
        const next = { ...current };
        for (const role of Object.keys(next)) {
          if (next[role] === asset.id) delete next[role];
        }
        return next;
      });
      await refreshAssets();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "删除失败");
    }
  }

  async function onSubmit() {
    if (!model || !engine) return;
    setError("");
    setSaving(true);
    try {
      const weights = model.artifacts
        .filter((artifact) => bindings[artifact.role])
        .map((artifact) => {
          const binding: { role: string; assetId: string; quant?: string } = {
            role: artifact.role,
            assetId: bindings[artifact.role],
          };
          if (artifact.quantized) {
            const quant = quants[artifact.role];
            if (!quant || !runtime || !runtime.weightQuants.includes(quant as (typeof runtime.weightQuants)[number])) {
              throw new Error(`请标明这个运行时支持的量化类型：${artifact.label}`);
            }
            binding.quant = quant;
          }
          return binding;
        });
      const missing = model.artifacts.filter((artifact) => artifact.required && !bindings[artifact.role]);
      if (!name.trim()) throw new Error("请填写包名称");
      if (missing.length) throw new Error(`还缺必填权重：${missing.map((item) => item.label).join("、")}`);
      const created = await api.createPackage({
        modelId: model.id,
        name,
        version,
        description,
        weights,
        engine: {
          image: engine.image,
          codeRef: engine.codeRef,
          command: [...parseCommand(engine.command), ...parseCommand(engine.commandExtra)],
          port: Number(engine.port),
          healthPath: engine.healthPath,
          protocol: engine.protocol,
          resources: {
            gpu: Number(engine.gpu),
            gpuMemoryGi: Number(engine.gpuMemoryGi),
            cpu: Number(engine.cpu),
            memoryGi: Number(engine.memoryGi),
          },
          env: parseEnv(engine.envText),
        },
      });
      location.hash = `#/packages/${created.manifest.metadata.id}`;
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "生成失败");
      setSaving(false);
    }
  }

  if (!model || !engine || !runtime) {
    return (
      <div className="page">
        <p className="muted">{error || "正在读取模型目录…"}</p>
      </div>
    );
  }

  const grouped = {
    library: assets.filter((asset) => asset.origin === "library" || asset.origin === "registered"),
    upload: assets.filter((asset) => asset.origin === "upload"),
    s3: assets.filter((asset) => asset.origin === "s3"),
  };

  return (
    <div className="page">
      <p className="page-kicker">把模型和参数收成一份清单</p>
      <h1 className="page-title">装配 {model.displayName}</h1>
      <p className="page-lead">{model.summary}</p>
      <p className="hint">{flow(model.interfaces.inputs, model.interfaces.outputs)} · {model.vendor}</p>
      {error && <p className="error">{error}</p>}
      {notice && <p className="ok">{notice}</p>}
      <div className="composer">
        <div>
          <section className="section">
            <h2 className="section-title">模型</h2>
            <div className="choice-row">
              {models.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  className={item.id === model.id ? "choice on" : "choice"}
                  onClick={() => setModelId(item.id)}
                >
                  <span>{item.displayName}</span>
                  <small>{item.task}</small>
                </button>
              ))}
            </div>
          </section>

          <section className="section">
            <h2 className="section-title">包信息</h2>
            <div className="form-grid">
              <div className="field">
                <label htmlFor="pkg-name">名称</label>
                <input id="pkg-name" className="input" value={name} onChange={(event) => setName(event.target.value)} placeholder="例如 车间分割-白天" />
              </div>
              <div className="field">
                <label htmlFor="pkg-version">版本</label>
                <input id="pkg-version" className="input" value={version} onChange={(event) => setVersion(event.target.value)} />
              </div>
              <div className="field span-2">
                <label htmlFor="pkg-desc">说明</label>
                <textarea id="pkg-desc" className="textarea" value={description} onChange={(event) => setDescription(event.target.value)} />
              </div>
            </div>
          </section>

          <section className="section">
            <h2 className="section-title">参数</h2>
            <p className="hint">
              包里只写地址，不复制大文件。本地库目录：<span className="mono">{meta?.localLibrary ?? "…"}</span>
            </p>
            {model.artifacts.map((artifact) => {
              const selected = assets.find((asset) => asset.id === bindings[artifact.role]);
              const preferMismatch =
                selected &&
                ((artifact.prefer === "directory" && selected.kind === "file") ||
                  (artifact.prefer === "file" && selected.kind === "directory"));
              return (
                <div className="slot" key={artifact.role}>
                  <div className="slot-head">
                    <strong>
                      {artifact.label}
                      {artifact.required ? "" : " · 可选"}
                    </strong>
                  </div>
                  <p className="hint">{artifact.hint}</p>
                  <select
                    className="select"
                    value={bindings[artifact.role] ?? ""}
                    onChange={(event) =>
                      setBindings((current) => ({ ...current, [artifact.role]: event.target.value }))
                    }
                  >
                    <option value="">未选择</option>
                    {assets.map((asset) => (
                      <option key={asset.id} value={asset.id} disabled={asset.missing}>
                        {sourceLabel(asset.source)} · {asset.name}
                        {asset.missing ? " · 路径失效" : ""}
                      </option>
                    ))}
                  </select>
                  {selected && <p className="uri">{selected.uri}</p>}
                  {artifact.quantized && (
                    <div className="field" style={{ marginTop: 8 }}>
                      <label htmlFor={`quant-${artifact.role}`}>量化类型</label>
                      <select
                        id={`quant-${artifact.role}`}
                        className="select"
                        value={quants[artifact.role] ?? ""}
                        onChange={(event) =>
                          setQuants((current) => ({ ...current, [artifact.role]: event.target.value }))
                        }
                      >
                        {runtime.weightQuants.map((quant) => (
                          <option key={quant} value={quant}>
                            {quant}
                          </option>
                        ))}
                      </select>
                    </div>
                  )}
                  {preferMismatch && <p className="warn">这个角色通常是{artifact.prefer === "directory" ? "目录" : "单文件"}，当前选的是另一种。服务认就可以继续。</p>}
                </div>
              );
            })}

            <div className="tabs" style={{ marginTop: 16 }}>
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
                    <input id="local-path" className="input" value={localPath} onChange={(event) => setLocalPath(event.target.value)} placeholder="/data/weights/sam3.pt" spellCheck={false} />
                  </div>
                  <div className="field">
                    <label htmlFor="local-note">备注</label>
                    <input id="local-note" className="input" value={localNote} onChange={(event) => setLocalNote(event.target.value)} />
                  </div>
                </div>
                <div className="toolbar" style={{ marginTop: 10 }}>
                  <button type="button" className="btn" onClick={() => void refreshAssets()}>
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
                    <input id="s3-endpoint" className="input" value={s3Endpoint} onChange={(event) => setS3Endpoint(event.target.value)} placeholder="http://127.0.0.1:9000" spellCheck={false} />
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

          <section className="section">
            <h2 className="section-title">运行时</h2>
            <p className="hint">选用模型后会带上对应的运行时。这份模型有多份运行时时，在这里选。镜像、端口、协议跟着运行时。官方引擎的启动选项可以在下面改，会写进模型包。</p>
            <RuntimeField
              options={runtimeOptions}
              image={engine.image}
              onChange={(next) => {
                setEngine(engineFromRuntime(model, next));
                setQuants(defaultQuants(model, next));
              }}
            />
            <p className="card-image">{engine.image}</p>
            <p className="hint">
              {protocolLabel(engine.protocol)} · 容器内端口 {engine.port}
            </p>
            {engine.command.trim().startsWith("--") ? (
              <>
                <div className="field" style={{ marginTop: 12 }}>
                  <label htmlFor="command-locked">内置启动参数</label>
                  <p id="command-locked" className="readonly-command">
                    {engine.command}
                  </p>
                  <p className="hint">模型路径、监听地址和端口跟容器约定，装配时不能改。</p>
                </div>
                <div className="field" style={{ marginTop: 12 }}>
                  <label htmlFor="command-extra">可改启动参数</label>
                  <textarea
                    id="command-extra"
                    className="textarea"
                    value={engine.commandExtra}
                    onChange={(event) => setEngine({ ...engine, commandExtra: event.target.value })}
                    placeholder={"--gpu-memory-utilization 0.5"}
                    spellCheck={false}
                  />
                  <p className="hint">空格分隔，跟官方 vLLM 选项一样。例如 --gpu-memory-utilization 0.5、--max-model-len 2048。</p>
                </div>
              </>
            ) : (
              <p className="hint">{engine.command || "未记录启动命令"}</p>
            )}
            <p className="hint">服务代码 {engine.codeRef || "未记录"}。只记录位置，不复制进模型包。</p>
            <p className="hint">健康检查 {engine.healthPath}</p>
            <p className="hint">
              支持的模型：
              {runtime.models
                .map((id) => models.find((item) => item.id === id)?.displayName ?? id)
                .join("、")}
            </p>
            <p className="hint">支持的权重：{runtime.weightQuants.join("、")}</p>
            <div className="res-grid" style={{ marginTop: 12 }}>
              <NumberField label="GPU 数量" value={engine.gpu} onChange={(value) => setEngine({ ...engine, gpu: value })} />
              <NumberField label="显存 GiB" value={engine.gpuMemoryGi} onChange={(value) => setEngine({ ...engine, gpuMemoryGi: value })} />
              <NumberField label="CPU" value={engine.cpu} onChange={(value) => setEngine({ ...engine, cpu: value })} />
              <NumberField label="内存 GiB" value={engine.memoryGi} onChange={(value) => setEngine({ ...engine, memoryGi: value })} />
            </div>
            <div className="field" style={{ marginTop: 12 }}>
              <label htmlFor="env">环境变量</label>
              <textarea
                id="env"
                className="textarea"
                value={engine.envText}
                onChange={(event) => setEngine({ ...engine, envText: event.target.value })}
                placeholder={"# 一行一个\nHF_HUB_OFFLINE=1"}
                spellCheck={false}
              />
            </div>
          </section>
        </div>

        <aside className="preview">
          <h2>这一份包</h2>
          <dl>
            <dt>名称</dt>
            <dd>{name.trim() || "还没写名称"}</dd>
            <dt>模型</dt>
            <dd>
              {model.displayName} · {version}
            </dd>
            <dt>运行时</dt>
            <dd>{engine.image || "—"}</dd>
            <dt>启动</dt>
            <dd>
              {[engine.command, engine.commandExtra].filter(Boolean).join(" ") || "—"} · {engine.protocol} · {engine.port}
            </dd>
            <dt>参数</dt>
            <dd>
              {model.artifacts.map((artifact) => {
                const selected = assets.find((asset) => asset.id === bindings[artifact.role]);
                return (
                  <div key={artifact.role}>
                    {artifact.label}：{selected ? selected.name : artifact.required ? "未选" : "不绑"}
                    {artifact.quantized && selected ? ` · ${quants[artifact.role] ?? ""}` : ""}
                  </div>
                );
              })}
            </dd>
            <dt>资源</dt>
            <dd>
              GPU {engine.gpu} · {engine.gpuMemoryGi} GiB
            </dd>
          </dl>
          <button type="button" className="btn primary block" disabled={saving} onClick={() => void onSubmit()}>
            {saving ? "正在写入…" : "生成模型包"}
          </button>
        </aside>
      </div>
    </div>
  );
}

function defaultQuants(model: ModelDef, runtime: RuntimeInfo): Record<string, string> {
  const fallback = runtime.weightQuants.includes("fp16") ? "fp16" : runtime.weightQuants[0] ?? "";
  const next: Record<string, string> = {};
  for (const artifact of model.artifacts) {
    if (artifact.quantized) next[artifact.role] = fallback;
  }
  return next;
}

function RuntimeField({
  options,
  image,
  onChange,
}: {
  options: RuntimeChoice[];
  image: string;
  onChange: (runtime: RuntimeChoice) => void;
}) {
  if (options.length < 2) return null;
  return (
    <div className="field">
      <label htmlFor="runtime">运行时</label>
      <select
        id="runtime"
        className="select"
        value={image}
        onChange={(event) => {
          const picked = options.find((item) => item.image === event.target.value);
          if (picked) onChange(picked);
        }}
      >
        {options.map((item) => (
          <option key={item.image} value={item.image}>
            {item.kind === "user" ? `${item.name} · ${item.image}` : item.image}
          </option>
        ))}
      </select>
    </div>
  );
}

function NumberField({ label, value, onChange }: { label: string; value: number; onChange: (value: number) => void }) {
  return (
    <div className="field">
      <label>{label}</label>
      <input className="input" type="number" min={0} value={value} onChange={(event) => onChange(Number(event.target.value))} />
    </div>
  );
}

function AssetList({ assets, onDelete }: { assets: WeightAsset[]; onDelete: (asset: WeightAsset) => void }) {
  if (assets.length === 0) return <p className="empty">这里还没有参数。</p>;
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
