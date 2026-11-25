export type Protocol = "http" | "websocket" | "grpc";
export type WeightSource = "local" | "upload" | "s3";
export type WeightKind = "file" | "directory" | "remote";

export type QuantType = "fp16" | "bf16" | "int8" | "awq" | "gptq" | "fp8";

export interface ArtifactSpec {
  role: string;
  label: string;
  required: boolean;
  prefer: "file" | "directory" | "any";
  hint: string;
  quantized: boolean;
}

export interface Resources {
  gpu: number;
  gpuMemoryGi: number;
  cpu: number;
  memoryGi: number;
}

export interface ModelDef {
  id: string;
  displayName: string;
  vendor: string;
  task: string;
  tone: string;
  summary: string;
  interfaces: { inputs: string[]; outputs: string[] };
  engine: {
    image: string;
    codeRef: string;
    command: string[];
    port: number;
    healthPath: string;
    protocol: Protocol;
    models: string[];
    weightQuants: QuantType[];
  };
  resources: Resources;
  env: Record<string, string>;
  artifacts: ArtifactSpec[];
}

export interface WeightAsset {
  id: string;
  source: WeightSource;
  origin: "library" | "registered" | "upload" | "s3";
  kind: WeightKind;
  name: string;
  uri: string;
  removable: boolean;
  missing: boolean;
  sizeBytes: number | null;
  fileCount: number | null;
  sha256: string | null;
  relativePath: string | null;
  localPath: string | null;
  endpoint: string | null;
  region: string | null;
  bucket: string | null;
  key: string | null;
  note: string | null;
  createdAt: string | null;
}

export interface WeightEntry {
  role: string;
  path?: string;
  quant?: string | null;
  label?: string;
  source?: WeightSource;
  kind?: WeightKind;
  name?: string;
  uri?: string;
  sizeBytes?: number | null;
  localPath?: string | null;
}

export interface Manifest {
  apiVersion: string;
  kind: "ModelPackage";
  metadata: {
    id: string;
    name: string;
    version: string;
    description?: string;
    createdAt?: string;
  };
  model: {
    id: string;
    displayName?: string;
    interfaces?: { inputs: string[]; outputs: string[] };
  };
  engine: {
    image: string;
    command: string[];
    port: number;
    protocol: Protocol;
    codeRef?: string;
    healthPath?: string;
    env?: Record<string, string>;
    resources?: Resources;
    models?: string[];
    weightQuants?: string[];
  };
  weights: WeightEntry[];
}

export interface PackageView {
  absoluteRoot: string;
  manifest: Manifest;
}

export interface MetaInfo {
  dataDir: string;
  localLibrary: string;
  catalogPath: string;
  packageDir: string;
}

export interface RuntimeInfo {
  id: string;
  name: string;
  image: string;
  kind: "builtin" | "user" | "base";
  codeRef: string;
  command: string[];
  port: number;
  healthPath: string;
  protocol: Protocol;
  models: string[];
  weightQuants: QuantType[];
  baseImage: string;
  note?: string;
  createdAt: string;
}

export interface BuildJob {
  id: string;
  image: string;
  action?: "pull";
  status: "queued" | "running" | "ready" | "failed";
  logs: string[];
  createdAt: string;
  updatedAt: string;
}

export interface Deployment {
  packageId: string;
  name: string;
  target: "local";
  status: "running" | "prepared" | "stopped" | "failed";
  image: string;
  protocol: Protocol | string;
  containerPort: number;
  hostPort: number | null;
  endpoint: string | null;
  pipeline: string | null;
  command: string[];
  logs: string[];
}

export interface EnvironmentInfo {
  docker: { ok: boolean; detail: string };
  nvidia: { ok: boolean; detail: string };
  ascend: { ok: boolean; detail: string };
  device: "cuda" | "ascend" | "none";
  images: Record<string, boolean>;
}

export interface EngineForm {
  image: string;
  codeRef: string;
  command: string;
  commandExtra: string;
  port: number;
  healthPath: string;
  protocol: Protocol;
  gpu: number;
  gpuMemoryGi: number;
  cpu: number;
  memoryGi: number;
  envText: string;
}
