import type { BuildJob, Deployment, EnvironmentInfo, MetaInfo, ModelDef, PackageView, RuntimeInfo, WeightAsset } from "./types";

export function errorMessage(text: string, fallback: string): string {
  try {
    const data = JSON.parse(text) as { detail?: unknown };
    if (typeof data.detail === "string") return data.detail;
    if (Array.isArray(data.detail)) {
      const parts = data.detail
        .map((item) => {
          if (item && typeof item === "object" && "msg" in item) {
            return String((item as { msg: unknown }).msg);
          }
          return "";
        })
        .filter(Boolean);
      if (parts.length) return parts.join("；");
    }
  } catch {
    /* 响应不是 JSON */
  }
  return text || fallback;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    throw new Error(errorMessage(await response.text(), response.statusText));
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export const api = {
  meta: () => request<MetaInfo>("/api/meta"),
  models: () => request<ModelDef[]>("/api/models"),
  weights: () => request<WeightAsset[]>("/api/weights"),
  packages: () => request<PackageView[]>("/api/packages"),
  package: (id: string) => request<PackageView>(`/api/packages/${id}`),
  createPackage: (body: unknown) =>
    request<PackageView>("/api/packages", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  deletePackage: (id: string) => request<{ ok: boolean }>(`/api/packages/${id}`, { method: "DELETE" }),
  registerS3: (body: { uri: string; endpoint: string; region: string; note: string }) =>
    request<WeightAsset>("/api/weights/s3", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  verifyS3: (body: { uri: string; endpoint: string; region: string }) =>
    request<{ ok: boolean; sizeBytes: number | null; detail: string | null }>("/api/weights/s3/verify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  registerPath: (body: { path: string; note: string }) =>
    request<WeightAsset>("/api/weights/local-path", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  deleteWeight: (id: string) =>
    request<{ ok: boolean }>(`/api/weights?id=${encodeURIComponent(id)}`, { method: "DELETE" }),
  runtimes: () => request<RuntimeInfo[]>("/api/runtimes"),
  createRuntime: (body: FormData) => request<RuntimeInfo>("/api/runtimes", { method: "POST", body }),
  registerBase: (body: { name: string; image: string; note: string }) =>
    request<{ base: RuntimeInfo | null; job: BuildJob | null }>("/api/bases", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  deleteBase: (image: string) =>
    request<{ ok: boolean }>(`/api/bases?image=${encodeURIComponent(image)}`, { method: "DELETE" }),
  environment: () => request<EnvironmentInfo>("/api/environment"),
  builds: (image: string) => request<BuildJob[]>(`/api/builds?image=${encodeURIComponent(image)}`),
  startBuild: (image: string, force = false) =>
    request<BuildJob>("/api/builds", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image, force }),
    }),
  deploys: () => request<Deployment[]>("/api/deploys"),
  deploy: (packageId: string, hostPort?: number) =>
    request<Deployment>("/api/deploys", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ packageId, hostPort: hostPort ?? null }),
    }),
  stopDeploy: (packageId: string) => request<Deployment>(`/api/deploys/${packageId}`, { method: "DELETE" }),
  deployHealth: (packageId: string) => request<{ ok: boolean; endpoint: string; body: Record<string, unknown> }>(`/api/deploys/${packageId}/health`),
  infer: (packageId: string, body: Record<string, unknown>) =>
    request<Record<string, unknown>>(`/api/deploys/${packageId}/infer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  chat: (packageId: string, body: { prompt: string; max_tokens?: number; temperature?: number }) =>
    request<Record<string, unknown>>(`/api/deploys/${packageId}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
};

export function uploadWeight(file: File, onProgress: (ratio: number) => void): Promise<WeightAsset> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/weights/upload");
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress(event.loaded / event.total);
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(JSON.parse(xhr.responseText) as WeightAsset);
        return;
      }
      reject(new Error(errorMessage(xhr.responseText, "上传失败")));
    };
    xhr.onerror = () => reject(new Error("上传失败"));
    const data = new FormData();
    data.append("file", file);
    xhr.send(data);
  });
}
