# 模型包清单

`manifest.json` 是一份模型包的清单。它只说明用哪个模型、哪张镜像、怎么启动、权重在哪。权重文件、镜像和服务代码都不在这个文件里，也不在清单所在的目录里。

每份包一个目录：`data/packages/<id>/manifest.json`。`<id>` 形如 `pkg_` 加 12 位十六进制，和 `metadata.id` 相同。

## 例子

π0.5 用自带服务进程，协议是 WebSocket：

```json
{
  "apiVersion": "aiserver.modelpackage/v1",
  "kind": "ModelPackage",
  "metadata": {
    "id": "pkg_ff3f9f58bf48",
    "name": "pi0.5-service",
    "version": "1.0.0",
    "createdAt": "2026-11-24T09:40:57.024855+00:00"
  },
  "model": { "id": "pi05" },
  "engine": {
    "image": "local/flashrt-engine:latest",
    "command": ["python", "serve.py"],
    "port": 8000,
    "protocol": "websocket"
  },
  "weights": [
    { "role": "checkpoint", "quant": "fp16", "path": "/home/kim/slim_models/pi05_libero" }
  ]
}
```

Qwen 的镜像是 vLLM，启动命令是它自己的参数，权重角色是 `model_dir`：

```json
{
  "model": { "id": "qwen" },
  "engine": {
    "image": "local/vllm-engine:latest",
    "command": ["--model", "/weights/checkpoint", "--host", "0.0.0.0", "--port", "8000", "--gpu-memory-utilization", "0.5"],
    "port": 8000,
    "protocol": "http"
  },
  "weights": [
    { "role": "model_dir", "quant": "fp16", "path": "/home/kim/slim_models/qwen/Qwen2.5-1.5B-Instruct" }
  ]
}
```

## 字段

| 字段 | 含义 |
|---|---|
| `apiVersion` | 固定 `aiserver.modelpackage/v1` |
| `kind` | 固定 `ModelPackage` |
| `metadata.id` | 包编号 |
| `metadata.name` | 包名，装配时填写 |
| `metadata.version` | 包版本，默认 `1.0.0` |
| `metadata.createdAt` | 生成时间，ISO 8601 |
| `model.id` | 模型编号，对应 `backend/catalog/models.yaml` 里的 `id`，例如 `sam3`、`genpose2`、`pi05`、`qwen` |
| `engine.image` | 运行时镜像名。部署时这张镜像必须已经在本机 |
| `engine.command` | 容器启动命令。多数运行时是 `python serve.py`；vLLM 是一串以 `--` 开头的参数 |
| `engine.port` | 容器内端口。宿主机端口在部署时另定，留空则自动分配 |
| `engine.protocol` | `http` 或 `websocket`。π0.5 是 `websocket`，其余当前模型是 `http` |
| `weights[].role` | 权重角色。SAM 3、GenPose2、π0.5 是 `checkpoint`；Qwen 是 `model_dir`。可选的附加文件用别的角色，例如 `cad` |
| `weights[].quant` | 量化或精度：`fp16`、`bf16`、`int8`、`awq`、`gptq`、`fp8`。必须是该运行时声明能读的一种。不加量化标记的角色没有这个字段 |
| `weights[].path` | 权重位置。本机文件或目录用绝对路径；对象存储用 `s3://bucket/key` |

显示名、输入输出、厂商、显存、代码目录不写进清单。页面需要这些说明时，按 `model.id` 去模型目录查。

## 部署时怎么用

本机部署把 `path` 只读挂进容器。目录挂到 `/weights/checkpoint`；单个文件挂到 `/weights/<文件名>`，所在目录挂到 `/weights`。主权重以外的角色挂到 `/weights/roles/<角色名>`。

`path` 以 `s3://` 开头时，不能部署到本机，因为容器挂的是本机路径。

清单不记 GPU 数量。部署时默认使用一块 GPU。健康检查路径未写入清单时，按 `/health`。
