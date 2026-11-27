import { useEffect, useState } from "react";
import { api } from "../api";
import { sourceLabel } from "../format";
import type { EnvironmentInfo, ModelDef, QuantJob, QuantType, RuntimeInfo, WeightAsset } from "../types";

const OFFLINE: Record<string, { quants: QuantType[]; kind: "file" | "directory" }> = {
  "local/vllm-engine:latest": { quants: ["fp8", "awq", "gptq"], kind: "directory" },
  "local/sam3-engine:latest": { quants: ["fp16", "bf16", "int8"], kind: "file" },
};

export function QuantPage() {
  const [models, setModels] = useState<ModelDef[]>([]);
  const [runtimes, setRuntimes] = useState<RuntimeInfo[]>([]);
  const [assets, setAssets] = useState<WeightAsset[]>([]);
  const [jobs, setJobs] = useState<QuantJob[]>([]);
  const [device, setDevice] = useState<EnvironmentInfo["device"] | undefined>(undefined);
  const [runtimeImage, setRuntimeImage] = useState("");
  const [assetId, setAssetId] = useState("");
  const [quant, setQuant] = useState<QuantType | "">("");
  const [calibration, setCalibration] = useState("");
  const [picked, setPicked] = useState<string | null>(null);
  const [ready, setReady] = useState<WeightAsset | null>(null);
  const [error, setError] = useState("");
  const [posting, setPosting] = useState(false);
  const [loaded, setLoaded] = useState(false);

  function applyJobs(list: QuantJob[]) {
    setJobs(list);
  }

  useEffect(() => {
    Promise.all([api.models(), api.runtimes(), api.weights(), api.environment(), api.quants()])
      .then(([modelList, runtimeList, weightList, environment, jobList]) => {
        setModels(modelList);
        setRuntimes(runtimeList);
        setAssets(weightList);
        setDevice(environment.device);
        applyJobs(jobList);
        const capable = runtimeList.filter((runtime) => canQuantize(runtime, environment.device));
        setRuntimeImage((current) => current || capable[0]?.image || "");
        setLoaded(true);
      })
      .catch((reason: Error) => {
        setError(reason.message);
        setLoaded(true);
      });
  }, []);

  const capable = runtimes.filter((runtime) => canQuantize(runtime, device));
  const runtime = capable.find((item) => item.image === runtimeImage) ?? capable[0] ?? null;
  const profile = runtime ? OFFLINE[runtime.image] : undefined;
  const options = profile?.quants ?? [];
  const selectedQuant = options.includes(quant as QuantType) ? quant : options[0] ?? "";
  const modelIds = runtime?.models ?? [];
  const weightKind = profile?.kind ?? "directory";
  const candidates = assets.filter(
    (asset) => asset.kind === weightKind && !asset.missing && asset.localPath && modelIds.some((id) => assetMatchesModel(asset, id)),
  );
  const selected = candidates.find((asset) => asset.id === assetId) ?? null;
  const busy = posting || jobs.some((job) => job.status === "running" || job.status === "queued");
  const shownId = picked ?? jobs.find((job) => job.status === "running" || job.status === "queued")?.id ?? null;

  useEffect(() => {
    if (!busy || posting) return;
    let stop = false;
    const timer = window.setInterval(() => {
      api
        .quants()
        .then((list) => {
          if (!stop) applyJobs(list);
        })
        .catch((reason: Error) => {
          if (!stop) setError(reason.message);
        });
    }, 2000);
    return () => {
      stop = true;
      window.clearInterval(timer);
    };
  }, [busy, posting]);

  async function start() {
    if (!selected || !selectedQuant) return;
    setError("");
    setReady(null);
    setPosting(true);
    try {
      const result = await api.startQuant(
        selected.id,
        selectedQuant,
        needsCalibration(selectedQuant) ? calibration : "",
        runtime?.image ?? "",
      );
      if (result.asset) {
        setReady(result.asset);
        setAssets((current) => (current.some((item) => item.id === result.asset?.id) ? current : [result.asset as WeightAsset, ...current]));
      }
      if (result.job) {
        setPicked(result.job.id);
        setJobs((current) => [result.job as QuantJob, ...current.filter((item) => item.id !== result.job?.id)]);
      }
      const list = await api.quants();
      applyJobs(list);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "量化没有开始");
    } finally {
      setPosting(false);
    }
  }

  return (
    <div className="page">
      <p className="page-kicker">写成新权重</p>
      <h1 className="page-title">量化</h1>
      <p className="page-lead">
        选一份已经登记的权重，用运行时写成新文件或新目录。原权重留在原地。完成后到装配页自己选中这份新权重，再生成模型包。
      </p>
      {error && <p className="error">{error}</p>}
      {!loaded ? (
        <p className="muted">正在读取…</p>
      ) : capable.length === 0 ? (
        <p className="empty">这台机器上还没有能离线量化的运行时。Qwen 可以写成 fp8、awq、gptq。SAM 3 可以写成 fp16、bf16、int8。</p>
      ) : (
        <section className="section">
          <div className="field">
            <label htmlFor="quant-runtime">运行时</label>
            <select
              id="quant-runtime"
              className="select"
              value={runtime?.image ?? ""}
              onChange={(event) => {
                setRuntimeImage(event.target.value);
                setQuant("");
                setAssetId("");
              }}
            >
              {capable.map((item) => (
                <option key={item.image} value={item.image}>
                  {runtimeLabel(item, models)}
                </option>
              ))}
            </select>
          </div>
          <div className="field" style={{ marginTop: 12 }}>
            <label htmlFor="quant-weight">{weightKind === "file" ? "权重文件" : "模型目录"}</label>
            <select id="quant-weight" className="select" value={selected?.id ?? ""} onChange={(event) => setAssetId(event.target.value)}>
              <option value="">未选择</option>
              {candidates.map((asset) => (
                <option key={asset.id} value={asset.id}>
                  {sourceLabel(asset.source)} · {asset.name}
                  {asset.note ? ` · ${asset.note}` : ""}
                </option>
              ))}
            </select>
            {selected && <p className="uri">{selected.uri}</p>}
            <p className="hint">只显示{linkedModelNames(runtime, models)}的{weightKind === "file" ? "文件" : "目录"}。</p>
            {candidates.length === 0 && (
              <p className="hint">
                还没有{linkedModelNames(runtime, models)}的本机{weightKind === "file" ? "文件" : "目录"}。先到<a href="#/weights">权重</a>页登记。
              </p>
            )}
          </div>
          <div className="field" style={{ marginTop: 12 }}>
            <label htmlFor="quant-precision">精度</label>
            <select
              id="quant-precision"
              className="select"
              value={selectedQuant}
              onChange={(event) => setQuant(event.target.value as QuantType)}
            >
              {options.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
            <p className="hint">{quantHint(selectedQuant)}</p>
          </div>
          {needsCalibration(selectedQuant) && (
            <div className="field" style={{ marginTop: 12 }}>
              <label htmlFor="quant-calibration">校准文本</label>
              <textarea
                id="quant-calibration"
                className="textarea"
                value={calibration}
                onChange={(event) => setCalibration(event.target.value)}
                placeholder="留空使用内置的中英文短句。自己贴时，空行分段，每段一条。"
              />
              <p className="hint">awq 和 gptq 用这些文本看激活。至少 8 条，留空则用内置短句。</p>
            </div>
          )}
          <div className="toolbar">
            <button type="button" className="btn primary" disabled={!selected || !selectedQuant || busy} onClick={() => void start()}>
              {busy ? "正在量化…" : "开始量化"}
            </button>
          </div>
          {ready && runtime && (
            <p className="hint">
              已登记 {ready.name}。{ready.note ? `${ready.note}。` : ""}
              <a href={assembleHref(runtime)}>去装配</a>
            </p>
          )}
        </section>
      )}
      <section className="section">
        <h2 className="section-title">记录</h2>
        {!loaded ? null : jobs.length === 0 ? <p className="empty">还没有量化记录。</p> : null}
        {jobs.map((job) => {
          const owner = runtimes.find((item) => item.image === job.image);
          return (
            <div className="slot" key={job.id}>
              <div className="slot-head">
                <strong>
                  {basename(job.source)} · {job.quant}
                </strong>
                <span className={`badge ${badgeClass(job.status)}`}>{statusLabel(job.status)}</span>
              </div>
              {job.status === "failed" && <p className="error">{failureDetail(job)}</p>}
              {job.status === "ready" && job.output && <p className="uri">{job.output}</p>}
              {shownId === job.id && job.logs.length > 0 && (
                <ol className="log-list">
                  {job.logs.map((line, index) => (
                    <li key={`${job.id}-${index}`}>{line}</li>
                  ))}
                </ol>
              )}
              <div className="toolbar">
                {job.logs.length > 0 && shownId !== job.id && (
                  <button type="button" className="btn small" onClick={() => setPicked(job.id)}>
                    查看日志
                  </button>
                )}
                {job.status === "ready" && (
                  <a className="btn small" href={assembleHref(owner)}>
                    去装配
                  </a>
                )}
              </div>
            </div>
          );
        })}
      </section>
    </div>
  );
}

function assetMatchesModel(asset: WeightAsset, modelId: string): boolean {
  const id = modelId.toLowerCase();
  if (markerMatches(asset.modelType ?? "", id)) return true;
  const segments = (asset.localPath ?? "").split(/[/\\]/).filter(Boolean);
  return segments.some((segment) => markerMatches(segment, id));
}

function markerMatches(value: string, modelId: string): boolean {
  const name = value.toLowerCase();
  if (!name.startsWith(modelId)) return false;
  const next = name.charAt(modelId.length);
  return next === "" || !/[a-z]/.test(next);
}

function linkedModelNames(runtime: RuntimeInfo | null, models: ModelDef[]): string {
  const names = (runtime?.models ?? []).map((id) => models.find((model) => model.id === id)?.displayName ?? id);
  return names.filter(Boolean).join("、") || "当前模型";
}

function quantHint(quant: string): string {
  const load = "会把模型读进显存，第一次还会在 vLLM 镜像里安装量化库。";
  if (quant === "awq") return `当前写出 awq 的 4 bit 目录。用校准文本找出重要通道再量化。${load}`;
  if (quant === "gptq") return `当前写出 gptq 的 4 bit 目录。用校准文本估计每一层的误差再量化。${load}`;
  if (quant === "int8") return "当前写出 int8 的 checkpoint。线性层按 int8 相乘，卷积仍是 fp16。不用校准文本。";
  if (quant === "fp16" || quant === "bf16") return `当前写出 ${quant} 的 checkpoint。部署后按这个精度放进显存，不用校准文本。`;
  return `当前写出 fp8 目录。只看权重，不用校准文本。${load}`;
}

function canQuantize(runtime: RuntimeInfo, device: EnvironmentInfo["device"] | undefined): boolean {
  if (runtime.kind === "base" || !OFFLINE[runtime.image]) return false;
  if (device === "ascend") return runtime.hardware === "ascend";
  if (device === "cuda") return runtime.hardware === "nvidia";
  return true;
}

function runtimeLabel(runtime: RuntimeInfo, models: ModelDef[]): string {
  const names = runtime.models.map((id) => models.find((item) => item.id === id)?.displayName ?? id).filter(Boolean);
  return `${names.join("、") || runtime.name} · ${runtime.image}`;
}

function assembleHref(runtime: RuntimeInfo | undefined): string {
  const model = runtime?.models[0];
  return model ? `#/new?model=${encodeURIComponent(model)}` : "#/new";
}

function needsCalibration(quant: string): boolean {
  return quant === "awq" || quant === "gptq";
}

function basename(path: string | undefined): string {
  if (!path) return "模型目录";
  const parts = path.split(/[/\\]/).filter(Boolean);
  return parts[parts.length - 1] || "模型目录";
}

function statusLabel(status: QuantJob["status"]): string {
  if (status === "ready") return "已登记";
  if (status === "failed") return "失败";
  if (status === "queued") return "排队";
  return "量化中";
}

function badgeClass(status: QuantJob["status"]): string {
  if (status === "ready") return "ready";
  if (status === "failed") return "stopped";
  return "starting";
}

function failureDetail(job: QuantJob): string {
  const skip = new Set(["量化失败", "Run 'docker run --help' for more information"]);
  const line = [...job.logs].reverse().find((item) => item && !skip.has(item.trim()));
  return line || "量化失败";
}
