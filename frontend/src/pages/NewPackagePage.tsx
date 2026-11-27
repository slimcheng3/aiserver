import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import { flow, protocolLabel, sourceLabel } from "../format";
import type { EngineForm, ModelDef, RuntimeInfo, WeightAsset } from "../types";

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
    accelerators: model.resources.accelerators ?? model.resources.gpu ?? 1,
    acceleratorMemoryGi: model.resources.acceleratorMemoryGi ?? model.resources.gpuMemoryGi ?? 16,
    cpu: model.resources.cpu,
    memoryGi: model.resources.memoryGi,
    envRows: envRowsFrom(model.env),
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

function envRowsFrom(env: Record<string, string>): { key: string; value: string }[] {
  const rows = Object.entries(env).map(([key, value]) => ({ key, value }));
  return rows.length ? rows : [{ key: "", value: "" }];
}

function envFromRows(rows: { key: string; value: string }[]): Record<string, string> {
  const env: Record<string, string> = {};
  for (const row of rows) {
    const key = row.key.trim();
    if (!key && !row.value.trim()) continue;
    if (!key) throw new Error("环境变量需要填写名称");
    if (key.includes("=") || /\s/.test(key)) throw new Error(`环境变量名不合法：${key}`);
    env[key] = row.value;
  }
  return env;
}

export function NewPackagePage({ initialModel }: { initialModel: string | null }) {
  const [models, setModels] = useState<ModelDef[]>([]);
  const [runtimes, setRuntimes] = useState<RuntimeInfo[]>([]);
  const [assets, setAssets] = useState<WeightAsset[]>([]);
  const [modelId, setModelId] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [version, setVersion] = useState("1.0.0");
  const [bindings, setBindings] = useState<Record<string, string>>({});
  const [quants, setQuants] = useState<Record<string, string>>({});
  const [engine, setEngine] = useState<EngineForm | null>(null);
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);
  const appliedModel = useRef<string | null>(null);

  useEffect(() => {
    Promise.all([api.models(), api.runtimes(), api.weights()])
      .then(([modelList, runtimeList, weightList]) => {
        setModels(modelList);
        setRuntimes(runtimeList);
        setAssets(weightList);
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

  useEffect(() => {
    if (!model || !runtime) return;
    setQuants((current) => {
      let changed = false;
      const next = { ...current };
      for (const artifact of model.artifacts) {
        if (!artifact.quantized) continue;
        const asset = assets.find((item) => item.id === bindings[artifact.role]);
        const marked = asset ? quantMarkedOn(asset, runtime.weightQuants) : null;
        const wanted = marked ?? defaultQuant(runtime);
        if (wanted && next[artifact.role] !== wanted) {
          next[artifact.role] = wanted;
          changed = true;
        }
      }
      return changed ? next : current;
    });
  }, [assets, bindings, model, runtime]);

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
              throw new Error(`请标明这个运行时支持的权重精度：${artifact.label}`);
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
        description: "",
        weights,
        engine: {
          image: engine.image,
          codeRef: engine.codeRef,
          command: [...parseCommand(engine.command), ...parseCommand(engine.commandExtra)],
          port: Number(engine.port),
          healthPath: engine.healthPath,
          protocol: engine.protocol,
          resources: {
            accelerators: Number(engine.accelerators),
            acceleratorMemoryGi: Number(engine.acceleratorMemoryGi),
            cpu: Number(engine.cpu),
            memoryGi: Number(engine.memoryGi),
          },
          env: envFromRows(engine.envRows),
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

  return (
    <div className="page">
      <p className="page-kicker">把模型和参数收成一份清单</p>
      <h1 className="page-title">装配 {model.displayName}</h1>
      <p className="page-lead">{model.summary}</p>
      <p className="hint">{flow(model.interfaces.inputs, model.interfaces.outputs)} · {model.vendor}</p>
      {error && <p className="error">{error}</p>}
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
            </div>
          </section>

          <section className="section">
            <h2 className="section-title">参数</h2>
            <p className="hint">
              包里只写地址，不复制大文件。还没有这份权重时，先去<a href="#/weights">权重</a>页登记。量化写出的新目录也会出现在这份列表里。
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
                      <label htmlFor={`quant-${artifact.role}`}>权重精度</label>
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
                      {selected && quantMarkedOn(selected, runtime.weightQuants) ? (
                        <p className="hint">这份目录是 {quants[artifact.role]}，精度已按目录填上。</p>
                      ) : runtime.image === "local/vllm-engine:latest" ? (
                        <p className="hint">
                          普通目录默认 fp16。量化目录在<a href="#/quant">量化</a>页生成，选中后精度跟着目录走。
                        </p>
                      ) : null}
                    </div>
                  )}
                  {preferMismatch && <p className="warn">这个角色通常是{artifact.prefer === "directory" ? "目录" : "单文件"}，当前选的是另一种。服务认就可以继续。</p>}
                </div>
              );
            })}
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
              {protocolLabel(engine.protocol)} · 容器内端口 {engine.port} · 容器内权重 {runtime.weightsPath || "/weights/checkpoint"}
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
            <p className="hint">支持的权重：{runtime.weightQuants.join("、")}</p>
            <div className="res-grid" style={{ marginTop: 12 }}>
              <NumberField label="加速卡" value={engine.accelerators} onChange={(value) => setEngine({ ...engine, accelerators: value })} />
              <NumberField label="卡上内存 GiB" value={engine.acceleratorMemoryGi} onChange={(value) => setEngine({ ...engine, acceleratorMemoryGi: value })} />
              <NumberField label="CPU" value={engine.cpu} onChange={(value) => setEngine({ ...engine, cpu: value })} />
              <NumberField label="内存 GiB" value={engine.memoryGi} onChange={(value) => setEngine({ ...engine, memoryGi: value })} />
            </div>
            <p className="hint">会写入模型包。加速卡数量表示要几张卡（英伟达、昇腾等）。当前本机部署在数量大于 0 时仍加 --gpus all。卡上内存是声明，不会自动变成 --gpu-memory-utilization。</p>
            <div className="field" style={{ marginTop: 12 }}>
              <span>环境变量</span>
              <div className="env-rows">
                {engine.envRows.map((row, index) => (
                  <div className="env-row" key={index}>
                    <input
                      className="input"
                      value={row.key}
                      onChange={(event) => {
                        const envRows = engine.envRows.map((item, itemIndex) =>
                          itemIndex === index ? { ...item, key: event.target.value } : item,
                        );
                        setEngine({ ...engine, envRows });
                      }}
                      placeholder="名称"
                      spellCheck={false}
                      aria-label={`环境变量名称 ${index + 1}`}
                    />
                    <input
                      className="input"
                      value={row.value}
                      onChange={(event) => {
                        const envRows = engine.envRows.map((item, itemIndex) =>
                          itemIndex === index ? { ...item, value: event.target.value } : item,
                        );
                        setEngine({ ...engine, envRows });
                      }}
                      placeholder="值"
                      spellCheck={false}
                      aria-label={`环境变量值 ${index + 1}`}
                    />
                    <button
                      type="button"
                      className="btn small"
                      onClick={() => {
                        const envRows = engine.envRows.filter((_, itemIndex) => itemIndex !== index);
                        setEngine({ ...engine, envRows: envRows.length ? envRows : [{ key: "", value: "" }] });
                      }}
                    >
                      删除
                    </button>
                  </div>
                ))}
              </div>
              <button
                type="button"
                className="btn small"
                onClick={() => setEngine({ ...engine, envRows: [...engine.envRows, { key: "", value: "" }] })}
              >
                添加
              </button>
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
              加速卡 {engine.accelerators} · {engine.acceleratorMemoryGi} GiB · CPU {engine.cpu} · {engine.memoryGi} GiB
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

function defaultQuant(runtime: RuntimeInfo): string {
  return runtime.weightQuants.includes("fp16") ? "fp16" : runtime.weightQuants[0] ?? "";
}

function quantMarkedOn(asset: WeightAsset, allowed: string[]): string | null {
  const note = asset.note?.match(/^([A-Za-z0-9]+)\s+量化自/);
  if (note && allowed.includes(note[1].toLowerCase())) return note[1].toLowerCase();
  const name = asset.name.toLowerCase();
  return allowed.find((quant) => name.endsWith(`-${quant}`)) ?? null;
}

function defaultQuants(model: ModelDef, runtime: RuntimeInfo): Record<string, string> {
  const fallback = defaultQuant(runtime);
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

function NumberField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="field">
      <span>{label}</span>
      <input
        className="input"
        type="number"
        min={0}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}
