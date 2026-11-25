import { useEffect, useState } from "react";
import { api } from "../api";
import { protocolLabel } from "../format";
import type { EnvironmentInfo, ModelDef, QuantType, RuntimeInfo } from "../types";

const QUANTS: QuantType[] = ["fp16", "bf16", "int8", "awq", "gptq", "fp8"];

const NOTE: Record<string, string> = {
  "local/sam3-engine:latest": "推理代码在 engines/sam3/upstream，服务入口在 engines/sam3。权重不进镜像。",
  "local/genpose2-engine:latest": "推理代码在 engines/genpose2/upstream。构建时编译 PointNet2，DINOv2 骨干打进镜像；Score/Energy/Scale 三个 pth 不进镜像。",
  "local/vllm-engine:latest": "官方 vllm-openai 镜像。OpenAI 兼容接口，端口 8000；权重是带配置和 tokenizer 的模型目录，不进镜像。",
  "local/flashrt-engine:latest": "推理代码在 engines/flashrt/upstream。构建时按本机 GPU/NPU 编译内核（4090=sm89，Orin=sm87，昇腾走 NPU 脚本）。权重不进镜像。",
  "local/base-cuda128-torch:latest": "只有 CUDA 12.8、Python 和 PyTorch。新运行时从这张镜像长出来。",
};

export function EnginesPage() {
  const [models, setModels] = useState<ModelDef[]>([]);
  const [runtimes, setRuntimes] = useState<RuntimeInfo[]>([]);
  const [env, setEnv] = useState<EnvironmentInfo | null>(null);
  const [building, setBuilding] = useState<Record<string, boolean>>({});
  const [jobs, setJobs] = useState<import("../types").BuildJob[]>([]);
  const [error, setError] = useState("");

  function reload() {
    return Promise.all([api.models(), api.runtimes(), api.environment(), api.builds("")]).then(
      ([modelList, runtimeList, environment, buildJobs]) => {
        setModels(modelList);
        setRuntimes(runtimeList);
        setEnv(environment);
        setJobs(buildJobs);
        const active: Record<string, boolean> = {};
        for (const job of buildJobs) {
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
        内置运行时由交付台提供镜像和服务代码。需要额外的 Python 包时，用基础镜像加上自己的 serve.py 和 requirements.txt 构建一份新运行时。登记基础镜像时，本机没有的会从仓库拉取。镜像已经在本机时，普通构建会跳过；服务代码改过之后，用重新构建再打一次。本机已经有的运行时可以下载成压缩包，里面是 runtime.tar 和 runtime.json。另一台机器解压后用 docker load -i runtime.tar 导入。权重不在这个包里。
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
              <RuntimeCard
                key={runtime.image}
                runtime={runtime}
                models={models}
                env={env}
                building={building}
                onChanged={() => reload().catch((reason: Error) => setError(reason.message))}
              />
            ))}
          {runtimes
            .filter((runtime) => runtime.kind === "user")
            .map((runtime) => (
              <RuntimeCard key={runtime.image} runtime={runtime} models={models} env={env} building={building} />
            ))}
        </div>
        {runtimes.every((runtime) => runtime.kind !== "user") && <p className="empty">还没有自定义运行时。</p>}
        <NewBaseForm
          pulling={jobs.find((job) => job.action === "pull" && (job.status === "running" || job.status === "queued")) ?? null}
          onCreated={() => reload().catch((reason: Error) => setError(reason.message))}
        />
        <NewRuntimeForm
          bases={runtimes.filter((runtime) => runtime.kind === "base")}
          onCreated={() => reload().catch((reason: Error) => setError(reason.message))}
        />
      </section>
    </div>
  );
}

function RuntimeCard({
  runtime,
  models,
  env,
  building,
  onChanged,
}: {
  runtime: RuntimeInfo;
  models: ModelDef[];
  env: EnvironmentInfo | null;
  building: Record<string, boolean>;
  onChanged?: () => void;
}) {
  const ready = Boolean(env?.images[runtime.image]);
  const active = Boolean(building[runtime.image]);
  const [rebuilding, setRebuilding] = useState(false);
  const [cardError, setCardError] = useState("");
  const kindLabel = runtime.kind === "base" ? "基础镜像" : runtime.kind === "user" ? "自定义" : "内置";
  const registeredBase = runtime.kind === "base" && !runtime.codeRef;
  const buildHref = `#/build?image=${encodeURIComponent(runtime.image)}`;

  async function rebuild() {
    setRebuilding(true);
    setCardError("");
    try {
      await api.startBuild(runtime.image, true);
      location.hash = buildHref;
    } catch (reason) {
      setCardError(reason instanceof Error ? reason.message : "重新构建没有开始");
      setRebuilding(false);
    }
  }

  async function removeBase() {
    if (!window.confirm(`取消登记「${runtime.name}」？本机上的镜像文件会留着。`)) return;
    setCardError("");
    try {
      await api.deleteBase(runtime.image);
      onChanged?.();
    } catch (reason) {
      setCardError(reason instanceof Error ? reason.message : "取消登记失败");
    }
  }

  const hint =
    NOTE[runtime.image] ??
    (runtime.kind === "base"
      ? runtime.note || "登记的基础镜像。登记时如果本机没有，会从仓库拉取。交付台不会替它构建。"
      : runtime.baseImage
        ? `父镜像 ${runtime.baseImage}。服务代码随这份运行时保存。`
        : "服务代码随这份运行时保存，父镜像是基础镜像。");

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
      <p className="hint">{hint}</p>
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
      {(runtime.kind !== "base" || ready || registeredBase) && (
        <div className="card-actions">
          {registeredBase && (
            <button type="button" className="btn" onClick={() => void removeBase()}>
              取消登记
            </button>
          )}
          {runtime.kind !== "base" &&
            (active ? (
              <a className="btn" href={buildHref}>
                构建中
              </a>
            ) : ready ? (
              <button type="button" className="btn" disabled={rebuilding} onClick={() => void rebuild()}>
                {rebuilding ? "正在开始…" : "重新构建"}
              </button>
            ) : (
              <a className="btn primary" href={buildHref}>
                {runtime.kind === "user" ? "构建这份运行时" : "第一次构建"}
              </a>
            ))}
          {ready && (
            <a className="btn tar-download" href={`/api/runtimes/tar?image=${encodeURIComponent(runtime.image)}`}>
              下载
            </a>
          )}
        </div>
      )}
      {cardError && <p className="error">{cardError}</p>}
    </article>
  );
}

function toneFor(runtime: RuntimeInfo, models: ModelDef[]): string {
  const model = models.find((item) => runtime.models.includes(item.id));
  return model?.tone ?? "ink";
}

function NewBaseForm({ pulling, onCreated }: { pulling: import("../types").BuildJob | null; onCreated: () => void }) {
  const [name, setName] = useState("");
  const [image, setImage] = useState("");
  const [note, setNote] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [saving, setSaving] = useState(false);

  async function waitForPull(job: import("../types").BuildJob, target: string) {
    let current = job;
    while (current.status === "running" || current.status === "queued") {
      setNotice(current.logs[current.logs.length - 1] ?? `正在从仓库拉取 ${target}`);
      await new Promise((resolve) => setTimeout(resolve, 1000));
      const listed = await api.builds(target);
      current = listed.find((item) => item.id === current.id) ?? current;
    }
    if (current.status === "failed") {
      const detail = [...current.logs].reverse().find((line) => line && line !== "拉取失败");
      throw new Error(detail ?? "拉取失败");
    }
  }

  useEffect(() => {
    if (!pulling) return;
    let stop = false;
    setSaving(true);
    setError("");
    waitForPull(pulling, pulling.image)
      .then(() => {
        if (stop) return;
        setNotice(`已登记 ${pulling.image}。新建运行时时可以选它作为父镜像。`);
        onCreated();
      })
      .catch((reason: Error) => {
        if (!stop) setError(reason.message);
      })
      .finally(() => {
        if (!stop) setSaving(false);
      });
    return () => {
      stop = true;
    };
  }, [pulling?.id]);

  async function submit() {
    setSaving(true);
    setError("");
    setNotice("");
    try {
      const created = await api.registerBase({ name, image, note });
      if (created.job && created.job.status !== "ready" && created.job.status !== "failed") {
        await waitForPull(created.job, image.trim());
      }
      if (created.job?.status === "failed") {
        throw new Error(created.job.logs[created.job.logs.length - 1] ?? "拉取失败");
      }
      setNotice("已登记。新建运行时时可以选它作为父镜像。");
      setName("");
      setImage("");
      setNote("");
      onCreated();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "登记失败");
    } finally {
      setSaving(false);
    }
  }

  return (
    <section className="section">
      <h2 className="section-title">登记基础镜像</h2>
      <p className="hint">镜像名要带标签。本机已经有的会直接登记；没有的会按这个名字从仓库拉取，例如 nvcr.io/nvidia/pytorch:24.08-py3。拉取可能较久，完成前请留在此页。</p>
      {error && <p className="error">{error}</p>}
      {notice && <p className="ok">{notice}</p>}
      <div className="form-grid">
        <div className="field">
          <label htmlFor="base-name">名称</label>
          <input id="base-name" className="input" value={name} onChange={(event) => setName(event.target.value)} placeholder="例如 CUDA 12.4" />
        </div>
        <div className="field">
          <label htmlFor="base-image">镜像名</label>
          <input
            id="base-image"
            className="input"
            value={image}
            onChange={(event) => setImage(event.target.value)}
            placeholder="nvcr.io/nvidia/pytorch:24.08-py3"
            spellCheck={false}
          />
        </div>
        <div className="field span-2">
          <label htmlFor="base-note">说明，可以不填</label>
          <input id="base-note" className="input" value={note} onChange={(event) => setNote(event.target.value)} />
        </div>
      </div>
      <div className="toolbar" style={{ marginTop: 12 }}>
        <button type="button" className="btn" disabled={saving} onClick={() => void submit()}>
          {saving ? "正在登记或拉取…" : "登记"}
        </button>
      </div>
    </section>
  );
}

function NewRuntimeForm({ bases, onCreated }: { bases: RuntimeInfo[]; onCreated: () => void }) {
  const [name, setName] = useState("");
  const [baseImage, setBaseImage] = useState("");
  const [modelsText, setModelsText] = useState("");
  const [quants, setQuants] = useState<QuantType[]>(["fp16"]);
  const [serve, setServe] = useState<File | null>(null);
  const [requirements, setRequirements] = useState<File | null>(null);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [saving, setSaving] = useState(false);
  const selectedBase = bases.some((item) => item.image === baseImage) ? baseImage : (bases[0]?.image ?? "");

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
      body.set("baseImage", selectedBase);
      body.set("serve", serve);
      if (requirements) body.set("requirements", requirements);
      const created = await api.createRuntime(body);
      const parent = bases.find((item) => item.image === selectedBase);
      setNotice(
        parent?.codeRef
          ? `已保存 ${created.name}。构建时如果本机还没有 ${selectedBase}，会先构建这张基础镜像。`
          : `已保存 ${created.name}。构建时使用父镜像 ${selectedBase}。`,
      );
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
          <label htmlFor="runtime-base">基础镜像</label>
          <select id="runtime-base" className="select" value={selectedBase} onChange={(event) => setBaseImage(event.target.value)}>
            {bases.map((item) => (
              <option key={item.image} value={item.image}>
                {item.name === item.image ? item.image : `${item.name} · ${item.image}`}
              </option>
            ))}
          </select>
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
