const MODALITY: Record<string, string> = {
  image: "图像",
  video: "视频",
  text: "文本",
  mask: "掩码",
  pose: "6D 位姿",
  action: "动作",
  proprio: "本体状态",
  depth: "深度",
};

export function modality(id: string): string {
  return MODALITY[id] ?? id;
}

export function flow(inputs: string[], outputs: string[]): string {
  return `${inputs.map(modality).join("、")} → ${outputs.map(modality).join("、")}`;
}

export function formatBytes(size: number | null): string {
  if (size == null) return "大小未知";
  if (size < 1024) return `${size} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let value = size / 1024;
  let index = 0;
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024;
    index += 1;
  }
  const digits = value >= 10 ? 0 : 1;
  return `${value.toFixed(digits)} ${units[index]}`;
}

export function formatTime(iso: string | null): string {
  if (!iso) return "—";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleString("zh-CN", { hour12: false });
}

export function sourceLabel(source: string): string {
  if (source === "local") return "本地";
  if (source === "upload") return "上传";
  if (source === "s3") return "S3";
  return source;
}

export function protocolLabel(protocol: string): string {
  if (protocol === "http") return "HTTP";
  if (protocol === "websocket") return "WebSocket";
  if (protocol === "grpc") return "gRPC";
  return protocol;
}
