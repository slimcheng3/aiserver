import { useEffect, useState } from "react";
import { api } from "../api";
import { protocolLabel } from "../format";
import type { EnvironmentInfo, ModelDef, QuantType, RuntimeInfo } from "../types";

const QUANTS: QuantType[] = ["fp16", "bf16", "int8", "awq", "gptq", "fp8"];

const NOTE: Record<string, string> = {
  "local/sam3-engine:latest": "推理代码在 engines/sam3/upstream，服务入口在 engines/sam3。权重不进镜像。",
  "local/genpose2-engine:latest": "模型代码在 D:\\playgound\\genpose2，服务代码在 engines/genpose2。",
  "local/vllm-engine:latest": "推理代码在 vLLM 镜像里。权重是带配置和 tokenizer 的模型目录。",
  "local/flashrt-engine:latest": "用 FlashRT 部署 π0.5。计算核来自本机 FlashRT 目录，不通过 requirements.txt 安装。",
  "local/base-cuda128-torch:latest": "只有 CUDA 12.8、Python 和 PyTorch。新运行时从这张镜像长出来。",
};

export function EnginesPage() {
  const [models, setModels] = useState<ModelDef[]>([]);
  const [runtimes, setRuntimes] = useState<RuntimeInfo[]>([]);
  const [env, setEnv] = useState<EnvironmentInfo | null>(null);
  const [building, setBuilding] = useState<Record<string, boolean>>({});
  const [error, setError] = useState("");

  function reload() {
    return Promise.all([api.models(), api.runtimes(), api.environment(), api.builds("")]).then(
      ([modelList, runtimeList, environment, jobs]) => {
        setModels(modelList);
        setRuntimes(runtimeList);
        setEnv(environment);
        const active: Record<string, boolean> = {};
        for (const job of jobs) {
          if (job.status === "running" || job.status === "queued") active[job.image] = true;
        }
        setBuilding(active);
      },
    );
  }

  useEffect(() => {
    reload().catch((reason: Error) => setError(reason.message));
  }, []);

  return (
    <div className="page">
      <p className="page-kicker">同一台机体</p>
      <h1 className="page-title">运行时</h1>
      <p className="page-lead">
        内置运行时由交付台提供镜像和服务代码。需要额外的 Python 包时，用基础镜像加上自己的 serve.py 和 requirements.txt 构建一份新运行时。镜像已经在本机时不再构建。
      </p>
      {error && <p className="error">{error}</p>}
      {env && (
        <p className="hint">
          Docker {env.docker.ok ? "可用" : "不可用"}
          {env.device === "cuda" ? " · 检测到 NVIDIA GPU" : ""}
          {env.device === "ascend" ? " · 检测到昇腾" : ""}
          {env.device === "none" ? " · 没有检测到 NVIDIA 或昇腾" : ""}
          。目录里的显存数字仍是手写的。
        </p>
      )}
      <section className="runtime-block">
        <h2 className="section-title">内置运行时</h2>
        <p className="page-lead">交付台维护镜像和服务代码。这里不能改其中的文件，也不能通过 requirements.txt 往里加包。</p>
        <div className="grid-models">
          {runtimes
            .filter((runtime) => runtime.kind === "builtin")
            .map((runtime) => (
              <RuntimeCard key={runtime.image} runtime={runtime} models={models} env={env} building={building} />
            ))}
        </div>
      </section>

      <section className="runtime-block">
        <h2 className="section-title">自定义运行时</h2>
        <p className="page-lead">目录里的加载方式不够用时，从下面的基础镜像上传 serve.py 和 requirements.txt。新镜像单独保存，内置运行时保持原样。</p>
        <div className="grid-models">
          {runtimes
            .filter((runtime) => runtime.kind === "base")
            .map((runtime) => (
              <RuntimeCard key={runtime.image} runtime={runtime} models={models} env={env} building={building} />
            ))}
          {runtimes
            .filter((runtime) => runtime.kind === "user")
            .map((runtime) => (
              <RuntimeCard key={runtime.image} runtime={runtime} models={models} env={env} building={building} />
            ))}
        </div>
        {runtimes.every((runtime) => runtime.kind !== "user") && <p className="empty">还没有自定义运行时。</p>}
        <NewRuntimeForm onCreated={() => reload().catch((reason: Error) => setError(reason.message))} />
      </section>
    </div>
  );
}

function RuntimeCard({
  runtime,
  models,
  env,
  building,
}: {
  runtime: RuntimeInfo;
  models: ModelDef[];
  env: EnvironmentInfo | null;
  building: Record<string, boolean>;
}) {
  const ready = Boolean(env?.images[runtime.image]);
  const active = Boolean(building[runtime.image]);
  const kindLabel = runtime.kind === "base" ? "基础镜像" : runtime.kind === "user" ? "自定义" : "内置";
  return (
    <article className="model-card" data-kind={runtime.kind} data-tone={toneFor(runtime, models)}>
      <div className="card-top">
        <span className={`badge ${runtime.kind}`}>{kindLabel}</span>
        <span className={`badge ${ready ? "ready" : active ? "starting" : "stopped"}`}>
          {ready ? "本机已有" : active ? "构建中" : "未构建"}
        </span>
      </div>
      <h2 className="card-name">{runtime.kind === "user" ? runtime.name : runtime.image}</h2>
      {runtime.kind === "user" && <p className="mono">{runtime.image}</p>}
      <p className="hint">
        {protocolLabel(runtime.protocol)} · 容器内端口 {runtime.port} · {runtime.command.join(" ")}
      </p>
      <p className="hint">{NOTE[runtime.image] ?? "服务代码随这份运行时保存，父镜像是基础镜像。"}</p>
      {runtime.codeRef && <p className="hint">服务代码 {runtime.codeRef}</p>}
      {runtime.models.length > 0 && (
        <p className="hint">
          支持的模型：
          {runtime.models.map((id, index) => (
            <span key={id}>
              {index > 0 ? "、" : ""}
              <a href={`#/new?model=${id}`}>{models.find((model) => model.id === id)?.displayName ?? id}</a>
            </span>
          ))}
        </p>
      )}
      {runtime.weightQuants.length > 0 && <p className="hint">支持的权重：{runtime.weightQuants.join("、")}</p>}
      {runtime.kind !== "base" && !ready && (
        <div className="card-actions">
          <a className="btn primary" href={`#/build?image=${encodeURIComponent(runtime.image)}`}>
            {active ? "构建中" : runtime.kind === "user" ? "构建这份运行时" : "第一次构建"}
          </a>
        </div>
      )}
    </article>
  );
}

function toneFor(runtime: RuntimeInfo, models: ModelDef[]): string {
  const model = models.find((item) => runtime.models.includes(item.id));
  return model?.tone ?? "ink";
}

function NewRuntimeForm({ onCreated }: { onCreated: () => void }) {
  const [name, setName] = useState("");
  const [modelsText, setModelsText] = useState("");
  const [quants, setQuants] = useState<QuantType[]>(["fp16"]);
  const [serve, setServe] = useState<File | null>(null);
  const [requirements, setRequirements] = useState<File | null>(null);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [saving, setSaving] = useState(false);

  function toggle(list: string[], value: string, on: boolean): string[] {
    return on ? [...list, value] : list.filter((item) => item !== value);
  }

  async function submit() {
    if (!serve) {
      setError("请上传 serve.py");
      return;
    }
    setSaving(true);
    setError("");
    setNotice("");
    try {
      const modelIds = modelsText
        .split(/[\n,，]/)
        .map((item) => item.trim())
        .filter(Boolean);
      if (modelIds.length === 0) {
        setError("请填写这份运行时可以加载的模型");
        setSaving(false);
        return;
      }
      const body = new FormData();
      body.set("name", name);
      body.set("models", modelIds.join(","));
      body.set("weightQuants", quants.join(","));
      body.set("serve", serve);
      if (requirements) body.set("requirements", requirements);
      const created = await api.createRuntime(body);
      setNotice(`已保存 ${created.name}。构建这份运行时的时候，如果本机还没有基础镜像，会先构建基础镜像。`);
      setName("");
      setModelsText("");
      setServe(null);
      setRequirements(null);
      onCreated();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "保存失败");
    } finally {
      setSaving(false);
    }
  }

  return (
    <section className="section">
      <h2 className="section-title">从基础镜像新建</h2>
      <p className="hint">
        serve.py 需要实现 load、infer 和 main，并按基础镜像的 HTTP、8080、/health 对外服务。不确定怎么写时，先下载参考样例，再改里面的 load 和 infer。requirements.txt 里的包要钉死版本。不能在这里更换 PyTorch 或 CUDA。权重不要打进镜像。
      </p>
      {error && <p className="error">{error}</p>}
      {notice && <p className="ok">{notice}</p>}
      <div className="form-grid">
        <div className="field">
          <label htmlFor="runtime-name">名称</label>
          <input id="runtime-name" className="input" value={name} onChange={(event) => setName(event.target.value)} />
        </div>
        <div className="field">
          <div className="label-line">
            <label htmlFor="runtime-serve">serve.py</label>
            <a className="text-link" href="/samples/serve.py" download="serve.py">
              下载参考样例
            </a>
          </div>
          <input id="runtime-serve" className="input" type="file" accept=".py,text/x-python" onChange={(event) => setServe(event.target.files?.[0] ?? null)} />
        </div>
        <div className="field span-2">
          <label htmlFor="runtime-req">requirements.txt，可以不传</label>
          <input id="runtime-req" className="input" type="file" accept=".txt,text/plain" onChange={(event) => setRequirements(event.target.files?.[0] ?? null)} />
        </div>
      </div>
      <div className="field" style={{ marginTop: 12 }}>
        <label htmlFor="runtime-models">这份运行时可以加载哪些模型</label>
        <textarea
          id="runtime-models"
          className="textarea"
          value={modelsText}
          onChange={(event) => setModelsText(event.target.value)}
          placeholder={"一行一个模型 id\nsam3\nyolo11"}
          spellCheck={false}
        />
        <p className="hint">目录里已有的可以写 sam3、genpose2、pi05、qwen，也可以直接写新的模型 id。</p>
      </div>
      <p className="hint">能直接读的权重精度</p>
      <div className="choice-row">
        {QUANTS.map((quant) => (
          <label key={quant} className={quants.includes(quant) ? "choice on" : "choice"}>
            <input
              type="checkbox"
              checked={quants.includes(quant)}
              onChange={(event) => setQuants(toggle(quants, quant, event.target.checked) as QuantType[])}
            />
            <span>{quant}</span>
          </label>
        ))}
      </div>
      <div className="toolbar" style={{ marginTop: 12 }}>
        <button type="button" className="btn primary" disabled={saving} onClick={() => void submit()}>
          {saving ? "正在保存…" : "保存运行时"}
        </button>
      </div>
    </section>
  );
}
